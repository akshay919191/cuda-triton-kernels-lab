#include "../common/common_helper.cuh"
#include "private_helper.cuh"

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <vector>

#define PADDING 8
#define fPAD 1

// x            : [B, H, S, D]       FP32 / FP16 / BF16
// position_ids : [B, S]             int32 
// inv_freq     : [rotary_dim / 2]   FP32
// output       : [B, H, S, D]       same dtype as x

template<int base , int Br>
__global__ void build_rope_cache_nopos(
    float* cos_cache,        // maxseqlen , rotatary_dim / 2
    float* sin_cache,       
    int max_seq_len,
    int rotatary_dim
){
    int tid = threadIdx.x;
    const int tileid = blockIdx.x;

    int halfrot   = rotatary_dim / 2;

    extern __shared__ char smem[];
    char* ptr = smem;

    /// theta = maxseqlen outer prod W(omega)

    float* inv_freq  = reinterpret_cast<float*>(ptr);
    ptr += (halfrot) * sizeof(float);    // [rotatary_dim / 2 + 1]

    float* seqlenitr = reinterpret_cast<float*>(ptr);
    ptr += max_seq_len * sizeof(float);

    float* buff = reinterpret_cast<float*>(ptr);
    ptr += max_seq_len * sizeof(float);


    for(int i = tid ; i < max_seq_len ; i += blockDim.x) seqlenitr[i] = i; __syncthreads()

    for(int i = tid ; i < halfrot ; i += blockDim.x) inv_freq[i] = rsqrtf(powf(base , 2 * i / rotatary_dim)); __syncthreads();

    /// now we have L and W , theta is L outer W 
    
    /// we do Br , so multiple blocks can be launched and be used parallely
    for(int i = tid ; i < Br * halfrot ; i += blockDim.x){
        int r = i / halfrot;
        int c = i % halfrot;

        int global_row = tileid * Br + r;
        if(global_row >= max_seq_len) continue;

        sin_cache[global_row * halfrot + c] = sinf(seqlenitr[global_row] * inv_freq[c]);
        cos_cache[global_row * halfrot + c] = cosf(seqlenitr[global_row] * inv_freq[c]);
    }
    /// shape is [seqlen , halfrot] , now either you can repeat looping , or make [seqlen , rotatory_dim]
}

template<typename scaler_t , int Br>
__global__ void ropefwd_kernel(
    const float* __restrict__ matrix, ///[batch , numhead , seqlen , headdim]
    const float* __restrict__ sin_cached,
    const float* __restrict__ cos_cached,
          float* __restrict__ output,
    int max_seq_len, int rotatary_dim
    int headdim , int numhead
)
{
    int tid = threadIdx.x; int warpid = tid >> 5; int lane = tid & 31;

    /*
    Args:
        input is a matrix  , eg [1,2,3,4,5,6]
    return:
        [-4,-5,-6,1,2,3] * sin + [1,2,3,4,5,6] * cos
    */

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * max_seq_len * headdim +
                           (long long)headid  * max_seq_len * headdim;

    const int rowstride = headdim + fPAD;
    const int halfdim   = headdim >> 1;
    const int halfrot   = rotatary_dim >> 1;

    extern __shared__ char smem[];
    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    float* smemA = reinterpret_cast<float*>(ptr);
    ptr += Br * rowstride * sizeof(float);

    float* normal = reinterpret_cast<float*>(ptr);
    ptr += headdim * sizeof(float);

    float* rotate = reinterpret_cast<float*>(ptr);
    ptr += headdim * sizeof(float);

    float* sin_cache = reinterpret_cast<float*>(ptr);
    ptr += Br * headdim * sizeof(float);

    float* cos_cache = reinterpret_cast<float*>(ptr);
    ptr += Br * headdim * sizeof(float);

    scalar_t* INptr = matrix + base; 
    float*   outptr = output + base;

    for (int i = tid; i < Br * (headdim / 4); i += blockDim.x) {
        int r = i / (headdim / 4);
        int c = (i % (headdim / 4)) * 4;

        int src = (tileid * Br + r) * halfrot + (c % halfrot);
        int dst = r * headdim + c;

        *reinterpret_cast<float4*>(sin_cache + dst) =
            *reinterpret_cast<const float4*>(sin_cached + src);

        *reinterpret_cast<float4*>(cos_cache + dst) =
            *reinterpret_cast<const float4*>(cos_cached + src);
    }

    __syncthreads();

    copy_to_float_smem<scalar_t, Br>(
        INptr,
        smemA,
        rowstride,
        tileid,
        max_seq_len,
        headdim
    );

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");

    __syncthreads();

    for(int row = warpid ; row < Br ; row += blockDim.x >> 5){
        for(int i = lane ; i < headdim ; i += 32)
        {
            normal[i] = smemA[row * rowstride + i];
            int rotateIDX = (i < halfdim) ? halfdim + i : i & halfdim;
            rotate[i] = (i < headdim) ? -smemA[row * rowstride + rotateIDX] : smemA[row * rowstride + rotateIDX];
        }
        __syncthreads();


        for(int i = lane ; i < headdim ; i += 32){
            int globalrow = tileid * Br + row;
            if(globalrow >= max_seq_len) continue;

            outptr[global_row * headdim + i] = normal[i] * cos_cache[row * headdim + i] + rotate[i] * sin_cache[row * headdim + i];
        }
        __syncthreads();
    }
    
}
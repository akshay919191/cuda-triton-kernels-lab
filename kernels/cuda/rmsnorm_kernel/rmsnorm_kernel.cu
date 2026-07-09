#include "../common/common_helper.cuh"  /// common among most of kernels
#include "private_helper.cuh"           /// specific for this 
/// includes needed
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <cmath>
#define PADDING 8
#define FLOAT4(x)  (*reinterpret_cast<float4*>(&(x)))
#define CFLOAT4(x) (*reinterpret_cast<const float4*>(&(x)))
/*
mean of the whole data , wrt to col --- row wise
*/
template<int Br , int seqlen , int headdim , int numhead>
__global__ void rmsfwd_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    float eps , const __half* __restrict__ gammaGlobal
)
{
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warpid = tid >> 5; 
    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;
        /// as it is used in attentions in transformers so we need to use batchid and all
    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                                (long long)headid * headdim * seqlen;
    const __half* INptr  = input + base;
    __half* outptr = output + base;
        /// now allocate the mem
    extern __shared__ char smen_raw[];
    char* ptr = smen_raw;
    const int rowStride = headdim + PADDING;   /// real row width padded, used for every Asmem index below
    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );
    /// for A

    __half* Asmem = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    /// for B    for gamma
    __half* gamma = reinterpret_cast<__half*>(ptr);
    ptr += (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    /// due to multi warp , we will store the mean here 
    float* result = reinterpret_cast<float*>(ptr);

    ptr += Br * sizeof(float);

    float* denoSmem = reinterpret_cast<float*>(ptr);   // [Br], one slot per row

    ptr += Br * sizeof(float);
        /// launch with number of rows actually needed so no loop
        /// const int rowitr = tileid;   use tile id instead of new var
    int vec_elems = headdim / 8;  

    for (int i = tid; i < vec_elems; i += blockDim.x) {
        FLOAT4(gamma[i * 8]) = CFLOAT4(gammaGlobal[i * 8]);
    }

    for (int i = vec_elems * 8 + tid; i < headdim; i += blockDim.x) {
        gamma[i] = gammaGlobal[i];
    }
    __syncthreads();
    cpasynccopyRMS<Br , seqlen , headdim>(
        INptr,
        Asmem,
        headdim + PADDING,
        tileid
    );
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
        /// loaded the whole Br * headim 
        /// now we need that reduce but but but , we need x ** 2 mean not x ; x is the whole row
    multiWarpReductionSUM_RMS(Asmem , result , headdim , Br);    //// it gives us sum not mean so divide by total numbers in a row means headdim

    __syncthreads();

    for (int row = tid; row < Br; row += blockDim.x)
        denoSmem[row] = sqrtf(result[row] / static_cast<float>(headdim) + eps);

    __syncthreads(); 
        /// now each row mean is in result , can be accesed by index
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
    int row = i / headdim;
    int col = i % headdim;
    float deno = denoSmem[row];   // shared read, no sqrtf here anymore
            Asmem[row * rowStride + col] = __float2half(
                (__half2float(Asmem[row * rowStride + col]) / deno) * __half2float(gamma[col])
            );
        }
    __syncthreads();
        /// write the normalized tile back to global — same row/col this tile owns,
        /// shifted by tileid * Br to land at the right block of seqlen. vectorized
        /// float4 so it's 8 halves a store instead of 1, same trick as the gamma load.
    for (int i = tid; i < Br * (headdim / 8); i += blockDim.x)
    {
    int row = i / (headdim / 8);
    int c8  = i % (headdim / 8);
    int globalRow = tileid * Br + row;
    if (globalRow >= seqlen) continue;   /// last tile can be partial if seqlen % Br != 0
            FLOAT4(outptr[globalRow * headdim + c8 * 8]) = CFLOAT4(Asmem[row * rowStride + c8 * 8]);
        }
}

template<int Br , int seqlen , int headdim , int numhead>
__global__ void rmsbwd_kernel(
    const __half* __restrict__ dl_final_out,
    const __half* __restrict__ input,
    const __half* __restrict__ gammaGlobal,
          __half* __restrict__ dl_dx,
          float eps
)
{
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warpid = tid >> 5; 
    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    extern __shared__ char smem[];
    char* ptr = smem;

    /*
    we need space for DO , dl_final , input , result(for rms deno saving)
    */

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31
    );

    __half* dl_in = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31
    );
    
    __half* dl_final = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31
    );
    
    __half* dl_O = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    //// we need atleast 2-3 element wise multiplication so we will write it down in our helper header
    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );
    /// for B    for gamma
    __half* gamma = reinterpret_cast<__half*>(ptr);

    ptr += (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );
    
    float* result = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    float* summat = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                                (long long)headid * headdim * seqlen;
    const __half* INptr  = input + base;
    const __half* Fptr   = dl_final_out + base;
          __half* out    = dl_dx + base;
    

    int vec_elems = headdim / 8;  // Number of full 8-element chunks

    for (int i = tid; i < vec_elems; i += blockDim.x) {
        FLOAT4(gamma[i * 8]) = CFLOAT4(gammaGlobal[i * 8]);
    }

    for (int i = vec_elems * 8 + tid; i < headdim; i += blockDim.x) {
        gamma[i] = gammaGlobal[i];
    }

    cpasynccopyRMS<Br , seqlen , headdim>(
        INptr,
        dl_in,
        headdim + PADDING,
        tileid
    );
    asm volatile("cp.async.commit_group;\n");

    cpasynccopyRMS<Br , seqlen , headdim>(
        Fptr,
        dl_final,
        headdim + PADDING,
        tileid
    );
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    multiWarpReductionSUM_RMS(dl_in , result , headdim , Br);
    __syncthreads();

    for (int row = tid; row < Br; row += blockDim.x)
        result[row] = sqrtf(result[row] / static_cast<float>(headdim) + eps);

    __syncthreads(); 

    elementwisemultiply<Br , headdim , 1>(dl_final , gamma , dl_O);
    __syncthreads();

    /// now we need only dl/dx  

    //// now d_final hold the summation
    elementwisemultiply<Br , headdim , 1>(dl_O , dl_in , dl_final);
    __syncthreads();

    multiWarpReductionSUM_RMS(dl_final , summat , headdim , Br);
    __syncthreads();

    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        float val = result[r];
        dl_O[r * headdim + c]  = __float2half(__half2float(dl_O[r * headdim + c]) / val);
        dl_in[r * headdim + c] = __float2half(__half2float(dl_in[r * headdim + c]) / (val * val * val));
    }
    __syncthreads();


    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim; int c = i % headdim;
        out[r * headdim + c] = dl_O[r * headdim + c] - (dl_in[r * headdim + c] * summat[r]);
    }
}



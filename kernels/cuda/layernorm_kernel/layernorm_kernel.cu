#include "../common/common_helper.cuh"
#include "private_helper.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <cmath>
#define PADDING 8

template<int Br>
__global__ void layernormfwd_kernel(
    const __half* __restrict__ input,
    const float*  __restrict__ gamma,
    const float*  __restrict__ betaa,
          __half* __restrict__ output,
    float eps , const int headdim , int seqlen , int numhead
){
    int tid = threadIdx.x;

    const int rowstride = headdim + PADDING;
    const int batchid   = blockIdx.x;
    const int headid    = blockIdx.y;
    const int tileid    = blockIdx.z;

    long long base = (long long)batchid * numhead * headdim * seqlen + 
                     (long long)headid  * headdim * seqlen;
    
    const __half* INptr = input + base;
          __half* outptr= output + base;

    extern __shared__ char smem[];
    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * rowstride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* gam = reinterpret_cast<float*>(ptr);
    ptr += headdim * sizeof(float);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* beta = reinterpret_cast<float*>(ptr);
    ptr += headdim * sizeof(float);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* res1 = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    float* res2 = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    cpasynccopy<Br>(INptr , smemA , rowstride , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");

    __syncthreads();

    for(int i = tid ; i < headdim ; i += blockDim.x)
    {
        gam[i] = gamma[i]; beta[i] = betaa[i];
    }
    __syncthreads();

    multiWarpReductionMEAN(smemA , res1 , headdim , rowstride , Br);
    __syncthreads();

    multiWarpReductionSIGMA2(smemA , res1 , res2 , headdim , rowstride , Br);
    __syncthreads();

    //// now we have sigma square and U and x , gamma and beta
    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int global_row = (tileid * Br + r);
        if (global_row >= seqlen) continue;

        int smemidx = r * rowstride + c;
        float val = gam[r] * (__half2float(smemA[smemidx]) - res1[r]) / (sqrtf(res2[r] + eps)) + beta[r];
        outptr[global_row * headdim + c] = val;

    }
}




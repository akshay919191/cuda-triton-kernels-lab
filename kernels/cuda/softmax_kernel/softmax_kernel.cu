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
__global__ void softmaxfwd_kernel(
    const __half* __restrict__ input,
          __half* __restrict__ output,
    const int headdim , const int seqlen , const int numhead
)
{
    int tid = threadIdx.x;
    const int rowstride = PADDING + headdim;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                           (long long)headid  * headdim * seqlen;

    const __half* INptr  = input + base;
          __half* outptr = output + base;
    
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

    __half* res = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    
}
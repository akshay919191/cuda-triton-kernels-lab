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
#define FLOAT4(x)  (*reinterpret_cast<float4*>(&(x)))
#define CFLOAT4(x) (*reinterpret_cast<const float4*>(&(x)))


template<int Br>
__global__ void Silufwd_kernel(
    const __half* __restrict__ input,
          __half* __restrict__ output,
    int seqlen , int headdim , int numhead
)
{
    int tid = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * seqlen * headdim + 
                           (long long)headid * numhead * seqlen;

    __half* INptr  = input + base;
    __half* outptr = output + base;

    extern __shared__ char smem[];
    char* ptr = smem;

    const int rowstride = headdim + PADDING;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * rowstride * sizeof(__half);

    cpasynccopy<Br>(INptr , smemA , rowstride , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");

    
}
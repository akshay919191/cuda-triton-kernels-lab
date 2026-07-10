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

///
template<int Br>
__global__ void gelufwd_kernel(
    const __half* __restrict__ input,
          __half* __restrict__ output ,
    const int seqlen , const int headdim , const int numhead
)
{
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warpid = tid >> 5; 
    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                                (long long)headid * headdim * seqlen;

    const __half* INptr  = input + base;
    __half* outptr = output + base;

    extern __shared__ char smem[];

    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    cpasynccopygelu<Br>(INptr , smemA , headdim + PADDING , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");

    __syncthreads();

    performGelu<Br>(smemA , headdim + PADDING , headdim);
    __syncthreads();

    /// now the activated terms are in smemA , save it globally
    for (int i = tid; i < Br * (headdim / 8); i += blockDim.x)
    {
    int row = i / (headdim / 8);
    int c8  = i % (headdim / 8);
    int globalRow = tileid * Br + row;
    if (globalRow >= seqlen) continue;   
    FLOAT4(outptr[globalRow * headdim + c8 * 8]) = CFLOAT4(smemA[row * rowStride + c8 * 8]);
    }

    for (int i = headdim * Br + tid; i < Br * headdim; i += blockDim.x)
    {
        int row = i / headdim;
        int col = i % headdim;

        if (col < headdim)
            continue;   
        int globalRow = tileid * Br + row;
        if (globalRow >= seqlen)
            continue;

        outptr[globalRow * headdim + col] =
            smemA[row * rowStride + col];
    }
}

template<int Br>
__global__ void gelubwd_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ dl_dy,
          __half* __restrict__ output ,
    const int seqlen , const int headdim , const int numhead
)
{
    int tid = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base    = (long long)batchid * numhead * seqlen * headdim +
                        (long long)headid  * seqlen * headdim;

    const __half* INptr  = input  + base;
    const __half* prevv  = dl_dy  + base;
    
          __half* outptr = output + base;

        
    extern __shared__ char smem[];
    
    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* upstream = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    cpasynccopygelu<Br>(prevv , upstream , headdim + PADDING , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");

    cpasynccopygelu<Br>(INptr , smemA , headdim + PADDING , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n");

    __syncthreads();

    performGelubck<Br>(smemA , upstream , headdim + PADDING , headdim);

    for (int i = headdim * Br + tid; i < Br * headdim; i += blockDim.x)
    {
        int row = i / headdim;
        int col = i % headdim;

        if (col < headdim)
            continue;   
        int globalRow = tileid * Br + row;
        if (globalRow >= seqlen)
            continue;

        outptr[globalRow * headdim + col] =
            smemA[row * rowStride + col];
    }
}
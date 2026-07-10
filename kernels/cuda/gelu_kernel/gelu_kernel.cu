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
    const __half* __restrict__ output ,
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

    
}
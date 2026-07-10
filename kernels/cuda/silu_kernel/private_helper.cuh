#pragma once

#include "../common/common_helper.cuh"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cuda.h>
#include <math.h>
#include <float.h>

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

template<int Br>
__device__ __forceinline__ void doSILU(
    __half* __restrict__ data,
    int seqlen , int headdim , int rowstride
)
{
    int tid = threadIdx.x;
    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim; 
        int c = i % headdim;

        int smemidx = r * rowstride + c;
        if(smemidx >= seqlen) break;

        data[smemidx] = __float2half(doSILU(__half2float(data[smemidx])));
    }
}
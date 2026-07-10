#pragma once

#include "../common/common_helper.cuh"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

__device__ __forceinline__ float silu_fwd_float(float x)
{
    float s = 1.0f / (1.0f + expf(-x));
    return x * s;
}

__device__ __forceinline__ float silu_bwd_float(float x)
{
    float s = 1.0f / (1.0f + expf(-x));

    // d/dx [x * sigmoid(x)]
    // = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    return s * (1.0f + x * (1.0f - s));
}


template<int Br>
__device__ __forceinline__ void doSILU(
    __half* __restrict__ data,
    int rowStride,
    int headdim
)
{
    int tid = threadIdx.x;

    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int idx = r * rowStride + c;

        float x = __half2float(data[idx]);
        float s = 1.0f / (1.0f + expf(-x));
        float y = x * s;

        data[idx] = __float2half(y);
    }
}


template<int Br>
__device__ __forceinline__ void doSILUbck(
    __half* __restrict__ data,
    const __half* __restrict__ upstream,
    int rowStride,
    int headdim
)
{
    int tid = threadIdx.x;

    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int idx = r * rowStride + c;

        float x  = __half2float(data[idx]);
        float dy = __half2float(upstream[idx]);

        float s = 1.0f / (1.0f + expf(-x));
        float grad = s * (1.0f + x * (1.0f - s));

        data[idx] = __float2half(dy * grad);
    }
}
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

#include <math.h>

__device__ __forceinline__ float U(float x)
{
    return 0.7978845608028654f * (x + 0.044715f * x * x * x);
}

__device__ __forceinline__ float DU(float x)
{
    return 0.7978845608028654f * (1.0f + 0.134145f * x * x);
}

__device__ __forceinline__ float GELU(float x)
{
    return 0.5f * x * (1.0f + tanhf(U(x)));
}

__device__ __forceinline__ float GELU_BWD(float x)
{
    float ux = U(x);
    float t  = tanhf(ux);

    return 0.5f * (1.0f + t)
         + 0.5f * x * (1.0f - t * t) * DU(x);
}


template<int Br>
__device__ __forceinline__ void performGelu(
    __half* __restrict__ data,
    const int stride,
    const int headdim
)
{
    int tid = threadIdx.x;

    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int idx = r * stride + c;

        float x = __half2float(data[idx]);
        float y = GELU(x);

        data[idx] = __float2half(y);
    }
}


template<int Br>
__device__ __forceinline__ void performGelubck(
    __half* __restrict__ data,
    const __half* __restrict__ dl_dy,
    const int stride,
    const int headdim
)
{
    int tid = threadIdx.x;

    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int idx = r * stride + c;

        float x  = __half2float(data[idx]);
        float dy = __half2float(dl_dy[idx]);

        float grad = GELU_BWD(x);

        data[idx] = __float2half(dy * grad);
    }
}
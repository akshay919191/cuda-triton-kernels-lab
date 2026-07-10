#pragma once

#include "../common/common_helper.cuh"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <float.h>


__device__ __forceinline__ float warp_reduce_sum_softmax(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__device__ __forceinline__ float warp_reduce_max_softmax(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}


__device__ __forceinline__
void multiWarpReductionSUM_softmax_plain(
    const __half* __restrict__ input,
    float* __restrict__ out,
    int cols,
    int rowStride,
    int Br
)
{
    int tid      = threadIdx.x;
    int lane     = tid & 31;
    int warpid   = tid >> 5;
    int numWarps = blockDim.x >> 5;

    for (int row = warpid; row < Br; row += numWarps)
    {
        float localSum = 0.0f;

        for (int c = lane; c < cols; c += 32)
        {
            localSum += __half2float(input[row * rowStride + c]);
        }

        float sum = warp_reduce_sum_softmax(localSum);

        if (lane == 0) {
            out[row] = sum;
        }
    }
}


// stable softmax over each row
template<int Br>
__device__ __forceinline__ void dosoftmax(
    __half* __restrict__ data,
    float* __restrict__ rowMax,
    float* __restrict__ rowSum,
    int headdim,
    int seqlen,
    int rowstride
)
{
    int tid      = threadIdx.x;
    int lane     = tid & 31;
    int warpid   = tid >> 5;
    int numWarps = blockDim.x >> 5;

    // row max
    for (int row = warpid; row < Br; row += numWarps)
    {
        float localMax = -FLT_MAX;

        for (int c = lane; c < headdim; c += 32)
        {
            float x = __half2float(data[row * rowstride + c]);
            localMax = fmaxf(localMax, x);
        }

        float m = warp_reduce_max_softmax(localMax);

        if (lane == 0) {
            rowMax[row] = m;
        }
    }

    __syncthreads();

    for (int row = warpid; row < Br; row += numWarps)
    {
        float m = rowMax[row];
        float localSum = 0.0f;

        for (int c = lane; c < headdim; c += 32)
        {
            int idx = row * rowstride + c;

            float x = __half2float(data[idx]);
            float e = expf(x - m);

            data[idx] = __float2half(e);
            localSum += e;
        }

        float s = warp_reduce_sum_softmax(localSum);

        if (lane == 0) {
            rowSum[row] = s;
        }
    }

    __syncthreads();

    // 3. normalize
    for (int row = warpid; row < Br; row += numWarps)
    {
        float invSum = 1.0f / rowSum[row];

        for (int c = lane; c < headdim; c += 32)
        {
            int idx = row * rowstride + c;

            float e = __half2float(data[idx]);
            data[idx] = __float2half(e * invSum);
        }
    }

    __syncthreads();
}
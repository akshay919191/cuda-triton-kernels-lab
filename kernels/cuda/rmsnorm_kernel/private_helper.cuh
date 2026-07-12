#pragma once

#include "../common/common_helper.cuh"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <stdint.h>
#include <math.h>
#include <float.h>




__device__ __forceinline__
void multiWarpReductionSUM_RMS(
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
        const __half* rowPtr = input + row * rowStride;

        float localSum = 0.0f;

        for (int c = lane; c < cols; c += WARP_SIZE)
        {
            float x = __half2float(rowPtr[c]);
            localSum += x * x;
        }

        localSum = reducesum(localSum);

        if (lane == 0) {
            out[row] = localSum;
        }
    }
}


__device__ __forceinline__
void multiWarpReductionSUM_plain_RMS(
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
        const __half* rowPtr = input + row * rowStride;

        float localSum = 0.0f;

        for (int c = lane; c < cols; c += WARP_SIZE)
        {
            localSum += __half2float(rowPtr[c]);
        }

        localSum = reducesum(localSum);

        if (lane == 0) {
            out[row] = localSum;
        }
    }
}


__device__ __forceinline__
void elementwisemultiply_runtime(
    const __half* __restrict__ inputA,
    const __half* __restrict__ inputB,
          __half* __restrict__ output,
    int M,
    int N,
    int K
)
{
    int tid = threadIdx.x;

    for (int i = tid; i < M * N; i += blockDim.x)
    {
        int r = i / N;
        int c = i % N;

        float a;
        float b;

        if (K == 1) {
            a = __half2float(inputA[r * N + c]);
            b = __half2float(inputB[c]);
        } else if (N == 1) {
            a = __half2float(inputA[c]);
            b = __half2float(inputB[r * N + c]);
        } else {
            a = __half2float(inputA[r * N + c]);
            b = __half2float(inputB[r * N + c]);
        }

        output[r * N + c] = __float2half(a * b);
    }
}


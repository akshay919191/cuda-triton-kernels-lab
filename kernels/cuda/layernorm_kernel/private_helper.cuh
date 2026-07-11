#pragma once

#include "../common/common_helper.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <float.h>


#pragma once

#include "../common/common_helper.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cuda.h>
#include <math.h>
#include <float.h>


__device__ __forceinline__
void multiWarpReductionMEAN(
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
            out[row] = localSum / (float)cols;
        }
    }
}


__device__ __forceinline__
void multiWarpReductionSIGMA2(
    const __half* __restrict__ input,
    float* __restrict__ mean,
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

        float localVar = 0.0f;
        float m = mean[row];

        for (int c = lane; c < cols; c += WARP_SIZE)
        {
            float x = __half2float(rowPtr[c]);
            float d = x - m;
            localVar += d * d;
        }

        localVar = reducesum(localVar);

        if (lane == 0) {
            out[row] = localVar / (float)cols;
        }
    }
}
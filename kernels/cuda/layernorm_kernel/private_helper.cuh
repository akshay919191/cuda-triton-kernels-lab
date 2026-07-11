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

        float localsum = 0.0f;

        const half2* rowPtr2 = reinterpret_cast<const half2*>(rowPtr);
        int cols2 = cols >> 1;

        for (int i = lane; i < cols2; i += WARP_SIZE)  /// warp size from common helper
        {
            half2 h = rowPtr2[i];
            float2 f = __half22float2(h);

            localsum += f.x;
            localsum += f.y;
        }

        localsum = reducesum(localsum);

        if (lane == 0)
            out[row] = localsum / cols;
    }
}

__device__ __forceinline__
void multiWarpReductionSIGMA2(
    const __half* __restrict__ input,
    float* __restrict__ res,
    float* __restrict__ out,
    int cols,
    int rowStride,
    int Br
){
    int tid      = threadIdx.x;
    int lane     = tid & 31;
    int warpid   = tid >> 5;
    int numWarps = blockDim.x >> 5;

    for(int row = warpid ; row < Br ; row += numWarps)
    {
        const __half* rowptr = input + row * rowStride;

        float localsigma = 0.f;

        const half2* rowptr2 = reinterpret_cast<half2*>(rowptr);
        int col2 = cols >> 1;

        for(int i = lane ; i < col2 ; i += WARP_SIZE)
        {
            half2 h = rowptr2[i];
            float2 f = __half22float2(h);

            localsigma += (f.x - res[row]) * (f.x - res[row]);
            localsigma += (f.y - res[row]) * (f.y - res[row]);
        }
        localsigma = reducesum(localsigma);
        if(lane == 0) out[row] = localsigma / cols;
    }

}
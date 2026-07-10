#pragma once

#include "../common/common_helper.cuh"


#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

/// we will force it for softmax
__device__ __forceinline__
void multiWarpReduction_softmax(
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

        for (int i = lane; i < cols2; i += WARP_SIZE)
        {
            half2 h = rowPtr2[i];
            float2 f = __half22float2(h);

            localsum += expf(f.x - out[row]);
            localsum += expf(f.y - out[row]);
        }

        localsum = reducesum(localsum);

        if (lane == 0)
            out[row] = localsum;
    }
}

template<int Br>
__device__ __shared__ void dosoftmax(
    __half* __restrict__ data,
    float*  __restrict__ rsult1,
    float*  __restrict__ rsult2,
    int headdim , int seqlen , int rowstride
)
{
    int tid = threadIdx.x;

    //// we will store all max in rsult , and then overwrite it with actual each row sum using this multiWarpReductionMax_half2
    multiWarpReductionMax_half2(data , rsult1 , headdim , rowstride , Br);
    __syncthreads();
    
    for(int i = tid ; i < Br ; i += blockDim.x) rsult2[i] = rsult1[i]; __syncthreads();

    //// now this function gives sum for each row with numerical stablity
    multiWarpReduction_softmax(data , rsult1 , headdim , rowstride , Br);
    __syncthreads();

    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int smemidx = r * rowstride + c;

        data[smemidx] = __float2half((__half2float(data[smemidx]) - rsult2[r]) / rsult1[r]);
    }

}
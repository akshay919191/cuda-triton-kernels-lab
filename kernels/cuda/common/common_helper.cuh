#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err) cuda_check(err, __FILE__, __LINE__)

/// sum reduce for a single 
__device__ __forceinline__
float reducesum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);

    return val;
}

__device__ __forceinline__
void multiWarpReductionSUM_half2(
    const __half* __restrict__ input,
    float* __restrict__ out,
    int cols)
{
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warpid = tid >> 5;

    const __half* rowPtr = input + warpid * cols;

    float localsum = 0;

    const half2* rowPtr2 = reinterpret_cast<const half2*>(rowPtr);

    int cols2 = cols >> 1;          // cols / 2

    for (int i = lane; i < cols2; i += WARP_SIZE)
    {
        half2 h = rowPtr2[i];

        float2 f = __half22float2(h);

        localsum += f.x;
        localsum += f.y;
    }

    localsum = reducesum(localsum);

    if (lane == 0)
        out[warpid] = localsum;
}


//// max reduce shfl
__device__ __forceinline__
float warpReduceMax(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__device__ __forceinline__
void multiWarpReductionMax_half2(
    const __half* __restrict__ input,
    float* __restrict__ out,
    int cols)
{
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warpid = tid >> 5;

    const __half* rowPtr = input + warpid * cols;

    float localMax = -FLT_MAX;

    const half2* rowPtr2 = reinterpret_cast<const half2*>(rowPtr);

    int cols2 = cols >> 1;          // cols / 2

    for (int i = lane; i < cols2; i += WARP_SIZE)
    {
        half2 h = rowPtr2[i];

        float2 f = __half22float2(h);

        localMax = fmaxf(localMax, f.x);
        localMax = fmaxf(localMax, f.y);
    }

    localMax = warpReduceMax(localMax);

    if (lane == 0)
        out[warpid] = localMax;
}


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

/// for loading we will use cpsync
template<int rows , int cols>
__device__ __forceinline__ void cpasynccopy(
    const __half* __restrict__ input,
          __half* __restrict__ out,    /// 4 * N  ---shared mem
          int stride,
          int itr
)
{
    int tid = threadIdx.x;

    /// iterations calc
    constexpr int numperitr = 8; /// 16 byte means 8 halfs(each half 2 byte)
    constexpr int totalitr  = (4 * cols) / numperitr;

    /// this works on 16 byte alignment

    for(int i = tid ; i < totalitr ; i += blockDim.x)
    {
        int logical_offset = i * numperitr;

        int logical_row    = logical_offset / cols;
        int logical_col    = logical_offset % cols;

        int actual_row     = logical_row + itr * 4;
        int actual_col     = logical_col;

        uint32_t smenaddr  = out + (logical_row * stride + logical_col) * sizeof(__half);

        bool isvalid = (actual_row < rows) && 
                        (actual_col + 7 < cols);

        const __half* global_src = 
            isvalid ? input + (size_t)actual_row * cols + actual_col : input;

        int predicate = isvalid ? 1 : 0;

        /*

            asm volatile(
            "{\n"
            "  .reg .pred p;\n"                                             set a predicate 
            "  .reg .u32 z;\n"                                              set a z with uint 32 dtype
            "  mov.u32 z, 0;\n"                                             zero all the z
            "  setp.ne.b32 p, %2, 0;\n"                                     now compare with actual condition 
            "  @p  cp.async.cg.shared.global [%0], [%1], 16;\n"             if p is true , then do this 
            "  @!p st.shared.v4.b32 [%0], {z, z, z, z};\n"                  if p is false , then fill with 0
            "}\n"
            :
            : "r"(smenaddr), "l"(global_src), "r"(predicate)
            : "memory"
        );

        */

        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  .reg .u32 z;\n"
            "  mov.u32 z, 0;\n"
            "  setp.ne.b32 p, %2, 0;\n"
            "  @p  cp.async.cg.shared.global [%0], [%1], 16;\n"
            "  @!p st.shared.v4.b32 [%0], {z, z, z, z};\n"
            "}\n"
            :
            : "r"(smenaddr), "l"(global_src), "r"(predicate)
            : "memory"
        );
    }
    
}
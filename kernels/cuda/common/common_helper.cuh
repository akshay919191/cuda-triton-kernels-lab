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

        float localMax = -FLT_MAX;

        const half2* rowPtr2 = reinterpret_cast<const half2*>(rowPtr);
        int cols2 = cols >> 1;

        for (int i = lane; i < cols2; i += WARP_SIZE)
        {
            half2 h = rowPtr2[i];
            float2 f = __half22float2(h);

            localMax = fmaxf(localMax, f.x); localMax = fmaxf(localMax, f.y);
        }

        localMax = warpReduceMax(localMax);

        if (lane == 0)
            out[row] = localsum;
    }
}


template<int Br>
__device__ __forceinline__ void cpasynccopy(
    const __half* __restrict__ input,
          __half* __restrict__ out,
          int stride,
          int itr,
          int rows,
          int cols
)
{
    int tid = threadIdx.x;

    constexpr int numperitr = 8; // 16 bytes = 8 halfs

    if ((cols & 7) == 0)
    {
        int vec_cols = cols / numperitr;
        int total_vec = Br * vec_cols;

        for (int i = tid; i < total_vec; i += blockDim.x)
        {
            int logical_row = i / vec_cols;
            int vec_id      = i % vec_cols;

            int logical_col = vec_id * numperitr;

            int actual_row = logical_row + itr * Br;
            int actual_col = logical_col;

            uint32_t smem_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(out + logical_row * stride + logical_col)
            );

            bool isvalid = actual_row < rows;

            const __half* global_src =
                isvalid ? input + (size_t)actual_row * cols + actual_col : input;

            int predicate = isvalid ? 1 : 0;

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
                : "r"(smem_addr), "l"(global_src), "r"(predicate)
                : "memory"
            );
        }
    }
    else
    {
        for (int i = tid; i < Br * cols; i += blockDim.x)
        {
            int logical_row = i / cols;
            int logical_col = i % cols;

            int actual_row = logical_row + itr * Br;

            if (actual_row < rows) {
                out[logical_row * stride + logical_col] =
                    input[(size_t)actual_row * cols + logical_col];
            } else {
                out[logical_row * stride + logical_col] = __float2half(0.0f);
            }
        }
    }
}
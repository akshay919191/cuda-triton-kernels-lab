#pragma once

#include "../common/common_helper.cuh"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <stdint.h>
#include <math.h>
#include <float.h>


template<int Br>
__device__ __forceinline__ void cpasynccopyRMS(
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


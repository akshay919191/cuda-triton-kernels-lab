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

#define U(x)  (0.7978845608028654f * ((x) + 0.044715f * (x) * (x) * (x)))
#define DU(x) (0.7978845608028654f * (1.0f + 0.134145f * (x) * (x)))

#define GELU(x) \
(0.5f * (x) * (1.0f + tanhf(U(x))))

#define GELU_BWD(x) \
(0.5f * (1.0f + tanhf(U(x))) + \
 0.5f * (x) * (1.0f - tanhf(U(x)) * tanhf(U(x))) * DU(x))


template<int Br>
__device__ __forceinline__ void cpasynccopygelu(
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
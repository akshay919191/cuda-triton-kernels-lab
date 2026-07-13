#pragma once

#include "../common/common_helper.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__device__ __forceinline__ float to_float(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value)
{
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(
    __nv_bfloat16 value
)
{
    return __bfloat162float(value);
}


template <typename scalar_t, int Br>
__device__ __forceinline__ void copy_to_float_smem(
    const scalar_t* __restrict__ input,
    float* __restrict__ out,
    int smem_stride,
    int itr,
    int rows,
    int cols
)
{
    const int tid = threadIdx.x;

    if constexpr (std::is_same_v<scalar_t, float>)
    {
        constexpr int elements_per_copy = 4;

        if ((cols % elements_per_copy) == 0)
        {
            const int vectors_per_row = cols / elements_per_copy;
            const int total_vectors = Br * vectors_per_row;

            for (
                int i = tid;
                i < total_vectors;
                i += blockDim.x
            )
            {
                const int logical_row = i / vectors_per_row;
                const int vector_id = i % vectors_per_row;

                const int logical_col =
                    vector_id * elements_per_copy;

                const int actual_row =
                    itr * Br + logical_row;

                const bool valid = actual_row < rows;

                float* shared_dst =
                    out +
                    logical_row * smem_stride +
                    logical_col;

                const float* global_src =
                    valid
                        ? input +
                          static_cast<size_t>(actual_row) * cols +
                          logical_col
                        : input;

                const uint32_t shared_addr =
                    static_cast<uint32_t>(
                        __cvta_generic_to_shared(shared_dst)
                    );

                const int predicate = valid ? 1 : 0;

                asm volatile(
                    "{\n"
                    "  .reg .pred p;\n"
                    "  .reg .u32 z;\n"
                    "  mov.u32 z, 0;\n"
                    "  setp.ne.b32 p, %2, 0;\n"
                    "  @p  cp.async.cg.shared.global "
                    "      [%0], [%1], 16;\n"
                    "  @!p st.shared.v4.b32 "
                    "      [%0], {z, z, z, z};\n"
                    "}\n"
                    :
                    : "r"(shared_addr),
                      "l"(global_src),
                      "r"(predicate)
                    : "memory"
                );
            }

            return;
        }
    }

    const int total_elements = Br * cols;

    for (
        int i = tid;
        i < total_elements;
        i += blockDim.x
    )
    {
        const int logical_row = i / cols;
        const int logical_col = i % cols;

        const int actual_row =
            itr * Br + logical_row;

        float value = 0.0f;

        if (actual_row < rows)
        {
            const size_t global_index =
                static_cast<size_t>(actual_row) * cols +
                logical_col;

            value = to_float<scalar_t>(
                input[global_index]
            );
        }

        out[
            logical_row * smem_stride +
            logical_col
        ] = value;
    }
}
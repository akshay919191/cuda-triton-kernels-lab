#pragma once

#include "../common/common_helper.cuh"

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
#include <cassert>

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
          int rows, int cols
)
{
    int tid = threadIdx.x;

    int numperitr = 8; // 16 bytes = 8 halfs
    int totalitr  = (Br * cols) / numperitr;

    for (int i = tid; i < totalitr; i += blockDim.x)
    {
        int logical_offset = i * numperitr;

        int logical_row = logical_offset / cols;
        int logical_col = logical_offset % cols;

        int actual_row = logical_row + itr * Br;  // FIXED: Br, not 4
        int actual_col = logical_col;

        uint32_t smem_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(out + logical_row * stride + logical_col)
        );

        bool isvalid = (actual_row < rows) && (actual_col + 7 < cols);

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


template<int Br>
__device__ __forceinline__ void performGelu(__half* __restrict__ data,
    const int stride , 
    const int headdim
){
    int tid = threadIdx.x;

    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        data[r * stride + c] = GELU(data[r * stride + c]);
    }
}

template<int Br>
__device__ __forceinline__ void performGelubck(__half* __restrict__ data,
    const float* __restrict__ dl_dy,
    const int stride , 
    const int headdim
){
    int tid = threadIdx.x;

    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        data[r * stride + c] = __half2float(GELU_BWD(data[r * stride + c])) * dl_dy[r * stride + c];
    }
}
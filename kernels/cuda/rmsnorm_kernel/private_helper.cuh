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

// Private helpers for rmsnorm kernel only.

template<int Br, int rows, int cols>
__device__ __forceinline__ void cpasynccopyRMS(
    const __half* __restrict__ input,
          __half* __restrict__ out,
          int stride,
          int itr
)
{
    int tid = threadIdx.x;

    constexpr int numperitr = 8; // 16 bytes = 8 halfs
    constexpr int totalitr  = (Br * cols) / numperitr;

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

        float localsum = 0.0f;

        const half2* rowPtr2 = reinterpret_cast<const half2*>(rowPtr);
        int cols2 = cols >> 1;

        for (int i = lane; i < cols2; i += WARP_SIZE)
        {
            half2 h = rowPtr2[i];
            float2 f = __half22float2(h);

            localsum += f.x * f.x;
            localsum += f.y * f.y;
        }

        localsum = reducesum(localsum);

        if (lane == 0)
            out[row] = localsum;
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

        float localsum = 0.0f;

        const half2* rowPtr2 = reinterpret_cast<const half2*>(rowPtr);
        int cols2 = cols >> 1;

        for (int i = lane; i < cols2; i += WARP_SIZE)
        {
            half2 h = rowPtr2[i];
            float2 f = __half22float2(h);

            localsum += f.x;
            localsum += f.y;
        }

        localsum = reducesum(localsum);

        if (lane == 0)
            out[row] = localsum;
    }
}


template<int M , int N , int K> /// this handles broadcastings too  (M * N) * (K * N) where K == 1  or (M * N) * (K * N) N == 1
__device__ __forceinline__ void elementwisemultiply(
    const __half* __restrict__ inputA,
    const __half* __restrict__ inputB,
          __half* __restrict__ output 
)
{
    int tid  = threadIdx.x;
    
    for(int i = tid ; i < M * N ; i += blockDim.x)
    {
        int r = i / N;
        int c = i % N;

        if (K == 1) {
            output[r * N + c] = inputA[r * N + c] * inputB[c];
        } else if(N == 1) {
            output[r * N * c] = inputB[r * N + c] * inputA[c];
        } else {
            /// means we are safe dims are equal but safely we will have a fallback using assert 
            assert((M == K) && "M does not equal K (K != 1, no broadcasting)");
            output[r * N + c] = inputA[r * N + c] * inputB[r * N + c];
        }
    }
}



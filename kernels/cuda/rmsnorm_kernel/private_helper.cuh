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

template<int Br , int rows , int cols>
__device__ __forceinline__ void cpasynccopyRMS(
    const __half* __restrict__ input,
          __half* __restrict__ out,    /// 4 * N  ---shared mem
          int stride,
          int itr
)
{
    int tid = threadIdx.x;

    /// iterations calc
    constexpr int numperitr = 8; /// 16 byte means 8 halfs(each half 2 byte)
    constexpr int totalitr  = (Br * cols) / numperitr;

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
            "  setp.ne.b32 p, %2, 0;\n"                                     now compare with actual condition store in p in  binary digit form
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


__device__ __forceinline__
void multiWarpReductionSUM_RMS(
    const __half* __restrict__ input,
    float* __restrict__ out,
    int cols , int Br)
{
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warpid = tid >> 5;

    #pragma unroll 2
    for(int bb = 0 ; bb < Br ; bb++)
    {
        const __half* rowPtr = input + (bb * 4 + warpid) * cols;

        float localsum = 0;

        const half2* rowPtr2 = reinterpret_cast<const half2*>(rowPtr);

        int cols2 = cols >> 1;          // cols / 2

        for (int i = lane; i < cols2; i += WARP_SIZE)
        {
            half2 h = rowPtr2[i];

            float2 f = __half22float2(h);

            localsum += f.x * f.x;
            localsum += f.y * f.y;
        }

        localsum = reducesum(localsum);

        if (lane == 0)
            out[bb * 4 + warpid] = localsum;
    }
}


template<int M , int N , int K> /// this handles broadcastings too  (M * N) * (K * N) where K == 1  or (M * N) * (K * N) N == 1
__device__ __forceinline__ void elementwisemultiply(
    const __half* __restrict__ inputA,
    const __half* __restrict__ inputB
)
{
    int tid  = threadIdx.x;
    
    for(int i = tid ; i < M * N ; i += blockDim.x)
    {
        int r = i / N;
        int c = i % N;

        if (K == 1) {
            inputA[r * N + c] = inputA[r * N + c] * inputB[c];
        } else if(N == 1) {
            inputB[r * N * c] = inputB[r * N + c] * inputA[c];
        } else {
            /// means we are safe dims are equal but safely we will have a fallback using assert 
            assert((M == K) && "M does not equal K (K != 1, no broadcasting)");
            inputA[r * N + c] = inputA[r * N + c] * inputB[r * N + c];
        }
    }
}



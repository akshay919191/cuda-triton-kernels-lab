#pragma once

#include "../common/common_helper.cuh"


#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

template<int Br>
__device__ __shared__ void dosoftmax(
    const __half* __restrict__ data,
          float*  __restrict__ rsult,
    int headdim , int seqlen , int rowstride
)
{
    int tid = threadIdx.x;

    //// we will store all max in rsult , and then overwrite it with actual each row sum using this multiWarpReductionMax_half2
    
}
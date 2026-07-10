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

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}
#include "../common/common_helper.cuh"
#include "private_helper.cuh"

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <vector>

#define PADDING 8

// TODO: write elementwise_fusion kernels here


std::vector<torch::Tensor> elementwise_fusion_forward_cuda(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");

    auto y = torch::empty_like(x);

    // TODO: launch kernel

    return {y};
}


std::vector<torch::Tensor> elementwise_fusion_backward_cuda(torch::Tensor dy, torch::Tensor x) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    TORCH_CHECK(dy.scalar_type() == torch::kFloat16, "dy must be float16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");

    auto dx = torch::empty_like(x);

    // TODO: launch backward kernel if needed

    return {dx};
}

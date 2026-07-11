#include <torch/extension.h>

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> layernorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
);

std::vector<torch::Tensor> layernorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward_cuda, "LayerNorm forward CUDA");
    m.def("backward", &layernorm_backward_cuda, "LayerNorm backward CUDA");
}
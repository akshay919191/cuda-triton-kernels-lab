#include <torch/extension.h>

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> silu_forward_cuda(
    torch::Tensor x
);

std::vector<torch::Tensor> silu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &silu_forward_cuda, "SILU forward CUDA");
    m.def("backward", &silu_backward_cuda, "SILU backward CUDA");
}
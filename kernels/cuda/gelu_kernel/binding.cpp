#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> gelu_forward_cuda(torch::Tensor x);

std::vector<torch::Tensor> gelu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward_cuda, "GELU forward CUDA");
    m.def("backward", &gelu_backward_cuda, "GELU backward CUDA");
}
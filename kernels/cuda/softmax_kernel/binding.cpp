#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> softmax_forward_cuda(
    torch::Tensor x
);

std::vector<torch::Tensor> softmax_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward_cuda, "Softmax forward CUDA");
    m.def("backward", &softmax_backward_cuda, "Softmax backward CUDA");
}
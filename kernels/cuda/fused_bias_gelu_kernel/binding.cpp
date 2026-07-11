#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_bias_gelu_forward_cuda(torch::Tensor x);

std::vector<torch::Tensor> fused_bias_gelu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_bias_gelu_forward_cuda, "fused_bias_gelu forward CUDA");
    m.def("backward", &fused_bias_gelu_backward_cuda, "fused_bias_gelu backward CUDA");
}

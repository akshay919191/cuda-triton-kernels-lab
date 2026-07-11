#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_bias_gelu_forward_cuda(
    torch::Tensor x,
    torch::Tensor bias
);

std::vector<torch::Tensor> fused_bias_gelu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_bias_gelu_forward_cuda, "Fused bias GELU forward CUDA");
    m.def("backward", &fused_bias_gelu_backward_cuda, "Fused bias GELU backward CUDA");
}
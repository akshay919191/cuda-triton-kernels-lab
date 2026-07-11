#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> elementwise_fusion_forward_cuda(torch::Tensor x);

std::vector<torch::Tensor> elementwise_fusion_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elementwise_fusion_forward_cuda, "elementwise_fusion forward CUDA");
    m.def("backward", &elementwise_fusion_backward_cuda, "elementwise_fusion backward CUDA");
}

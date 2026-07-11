#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_residual_rmsnorm_forward_cuda(torch::Tensor x);

std::vector<torch::Tensor> fused_residual_rmsnorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_residual_rmsnorm_forward_cuda, "fused_residual_rmsnorm forward CUDA");
    m.def("backward", &fused_residual_rmsnorm_backward_cuda, "fused_residual_rmsnorm backward CUDA");
}

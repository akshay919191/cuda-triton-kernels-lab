#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_residual_rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    double eps
);

std::vector<torch::Tensor> fused_residual_rmsnorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_residual_rmsnorm_forward_cuda, "Fused residual RMSNorm forward CUDA");
    m.def("backward", &fused_residual_rmsnorm_backward_cuda, "Fused residual RMSNorm backward CUDA");
}
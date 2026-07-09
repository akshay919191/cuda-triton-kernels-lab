#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
);

std::vector<torch::Tensor> rmsnorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor inv_rms,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward_cuda, "RMSNorm forward CUDA");
    m.def("backward", &rmsnorm_backward_cuda, "RMSNorm backward CUDA");
}
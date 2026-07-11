#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> rope_forward_cuda(torch::Tensor x);

std::vector<torch::Tensor> rope_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rope_forward_cuda, "rope forward CUDA");
    m.def("backward", &rope_backward_cuda, "rope backward CUDA");
}

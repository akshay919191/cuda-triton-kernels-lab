#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace py = pybind11;


std::vector<torch::Tensor> build_rope_cache_cuda(
    const torch::Tensor& reference,
    int64_t max_seq_len,
    int64_t rotary_dim,
    double base
);


torch::Tensor rope_forward_cuda(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset
);


torch::Tensor rope_backward_cuda(
    const torch::Tensor& grad_out,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "build_cache",
        &build_rope_cache_cuda,
        py::arg("reference"),
        py::arg("max_seq_len"),
        py::arg("rotary_dim"),
        py::arg("base") = 10000.0
    );

    m.def(
        "forward",
        &rope_forward_cuda,
        py::arg("input"),
        py::arg("position_ids"),
        py::arg("cos_cache"),
        py::arg("sin_cache"),
        py::arg("rotary_dim"),
        py::arg("position_offset") = 0
    );

    m.def(
        "backward",
        &rope_backward_cuda,
        py::arg("grad_out"),
        py::arg("position_ids"),
        py::arg("cos_cache"),
        py::arg("sin_cache"),
        py::arg("rotary_dim"),
        py::arg("position_offset") = 0
    );
}
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace py = pybind11;

/// this is LLM generated

// Defined inside rope_cuda.cu
std::vector<torch::Tensor> build_rope_cache_cuda(
    const torch::Tensor& reference,
    int64_t cache_len,
    int64_t rotary_dim,
    double base
);


// Defined inside rope_cuda.cu
torch::Tensor rope_forward_cuda(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset,
    int64_t Br
);


// Defined inside rope_cuda.cu
torch::Tensor rope_backward_cuda(
    const torch::Tensor& grad_out,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset,
    int64_t Br
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA implementation of rotary positional embeddings";

    m.def(
        "build_cache",
        &build_rope_cache_cuda,
        py::arg("reference"),
        py::arg("cache_len"),
        py::arg("rotary_dim"),
        py::arg("base") = 10000.0,
        R"doc(
Build the FP32 RoPE caches.

Returns:
    cos_cache: [cache_len, rotary_dim / 2]
    sin_cache: [cache_len, rotary_dim / 2]
    inv_freq:  [rotary_dim / 2]
)doc"
    );

    m.def(
        "forward",
        &rope_forward_cuda,
        py::arg("input"),
        py::arg("position_ids"),
        py::arg("cos_cache"),
        py::arg("sin_cache"),
        py::arg("rotary_dim"),
        py::arg("position_offset") = 0,
        py::arg("Br") = 32,
        R"doc(
Apply RoPE forward.

input:
    [B, H, S, D], float32/float16/bfloat16

position_ids:
    [B, S], int32/int64, or None

When position_ids is None:
    position = position_offset + sequence_index
)doc"
    );

    m.def(
        "backward",
        &rope_backward_cuda,
        py::arg("grad_out"),
        py::arg("position_ids"),
        py::arg("cos_cache"),
        py::arg("sin_cache"),
        py::arg("rotary_dim"),
        py::arg("position_offset") = 0,
        py::arg("Br") = 32,
        R"doc(
Apply inverse RoPE to obtain the input gradient.

grad_out:
    [B, H, S, D], float32/float16/bfloat16

position_ids:
    Must represent the same positions used in forward,
    or be None with the same position_offset.
)doc"
    );
}
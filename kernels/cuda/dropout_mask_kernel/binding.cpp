#include <torch/extension.h>

torch::Tensor dropout_cuda(
    torch::Tensor x,
    double p,
    uint64_t seed,
    uint64_t offset
);

torch::Tensor dropout(
    torch::Tensor x,
    double p,
    uint64_t seed,
    uint64_t offset
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(p >= 0.0 && p < 1.0, "p must be in [0, 1)");

    return dropout_cuda(x, p, seed, offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dropout",
        &dropout,
        "CUDA Dropout"
    );
}
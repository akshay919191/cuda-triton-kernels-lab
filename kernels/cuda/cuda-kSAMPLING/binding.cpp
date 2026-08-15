#include <torch/extension.h>
#include <vector>

// CUDA launcher declarations
void ksmapling_cuda(
    torch::Tensor vocab,
    int64_t K,
    torch::Tensor out_vals,
    torch::Tensor out_idxs
);

torch::Tensor topk_cuda(
    torch::Tensor vocab,
    int64_t K
) {
    TORCH_CHECK(vocab.is_cuda(), "vocab must be CUDA");
    TORCH_CHECK(vocab.scalar_type() == torch::kFloat32,
                "vocab must be float32");
    TORCH_CHECK(vocab.dim() == 2,
                "vocab must have shape [batch, vocab_size]");
    TORCH_CHECK(vocab.is_contiguous(),
                "vocab must be contiguous");

    const auto batch = vocab.size(0);
    const auto vocab_size = vocab.size(1);

    TORCH_CHECK(K > 0, "K must be > 0");
    TORCH_CHECK(K <= vocab_size, "K must be <= vocab size");

    auto out_vals = torch::empty(
        {batch, K},
        vocab.options()
    );

    auto out_idxs = torch::empty(
        {batch, K},
        vocab.options().dtype(torch::kInt32)
    );

    ksmapling_cuda(
        vocab,
        K,
        out_vals,
        out_idxs
    );

    return torch::stack({out_vals, out_idxs.to(torch::kFloat32)}, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "topk",
        &topk_cuda,
        "CUDA Top-K"
    );
}
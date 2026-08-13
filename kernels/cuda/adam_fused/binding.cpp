#include <torch/extension.h>

void adam_fused_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step
);

void adam_fused(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step
) {
    TORCH_CHECK(param.is_cuda(), "param must be a CUDA tensor");
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(exp_avg.is_cuda(), "exp_avg must be a CUDA tensor");
    TORCH_CHECK(exp_avg_sq.is_cuda(), "exp_avg_sq must be a CUDA tensor");

    adam_fused_cuda(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "adam_fused",
        &adam_fused,
        "Fused AdamW CUDA"
    );
}
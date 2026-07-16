import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "kernels" / "cuda"

KERNEL_PATHS = [
    CUDA_DIR / "rmsnorm_kernel",
    CUDA_DIR / "layernorm_kernel",
    CUDA_DIR / "softmax_kernel",
    CUDA_DIR / "gelu_kernel",
    CUDA_DIR / "silu_kernel",
    CUDA_DIR / "fused_bias_gelu_kernel",
    CUDA_DIR / "fused_residual_rmsnorm_kernel",
]

for p in KERNEL_PATHS:
    sys.path.insert(0, str(p))


import rmsnorm_cuda
import layernorm_cuda
import softmax_cuda
import gelu_cuda
import silu_cuda
import fused_bias_gelu_cuda
import fused_residual_rmsnorm_cuda


EPS = 1e-5
FAILURES = []


def first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def check_finite(name, t, kernel, shape):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero()[0].tolist()
        raise AssertionError(
            f"{kernel} {shape} failed finite check for {name}. "
            f"First bad index: {bad}, value={t[tuple(bad)].item()}"
        )


def check_close(name, custom, ref, kernel, shape, atol, rtol):
    custom_f = custom.float()
    ref_f = ref.float()

    diff = (custom_f - ref_f).abs()
    max_err = diff.max().item()
    idx = diff.argmax().item()

    custom_val = custom_f.flatten()[idx].item()
    ref_val = ref_f.flatten()[idx].item()

    ok = torch.allclose(custom_f, ref_f, atol=atol, rtol=rtol)

    if not ok:
        raise AssertionError(
            f"{kernel} {shape} failed {name}\n"
            f"  max_err={max_err}\n"
            f"  idx={idx}\n"
            f"  custom={custom_val}\n"
            f"  ref={ref_val}\n"
            f"  atol={atol}, rtol={rtol}"
        )

    return max_err


def run_case(kernel, shape, fn):
    try:
        msg = fn()
        print(f"  PASS {shape} | {msg}")
    except Exception as e:
        print(f"  FAIL {shape} | {e}")
        FAILURES.append((kernel, shape, str(e)))


def rand_half(shape, scale=1.0):
    return (torch.randn(*shape, device="cuda", dtype=torch.float16) * scale).contiguous()


def ref_rmsnorm(x, gamma):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + EPS)
    return x / rms * gamma


def test_rmsnorm_shape(B, H, N, D):
    shape = (B, H, N, D)

    x = rand_half(shape)
    gamma = rand_half((D,))

    dy = rand_half(shape)

    y = first(rmsnorm_cuda.forward(x, gamma, EPS))
    check_finite("forward", y, "RMSNorm", shape)

    x_ref = x.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)

    y_ref_fp32 = ref_rmsnorm(x_ref, gamma_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, "RMSNorm", shape, 5e-2, 5e-2)

    out = rmsnorm_cuda.backward(dy, x, gamma, EPS)
    dx = out[0]
    dgamma = out[1]

    y_ref_fp32.backward(dy.float())

    dx_ref = x_ref.grad
    dgamma_ref = gamma_ref.grad

    check_finite("dx", dx, "RMSNorm", shape)
    check_finite("dgamma", dgamma, "RMSNorm", shape)

    dx_err = check_close("dx", dx, dx_ref, "RMSNorm", shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, dgamma_ref, "RMSNorm", shape, 5e-1, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dgamma={dg_err:.4g}"



def test_layernorm_shape(B, H, N, D):
    shape = (B, H, N, D)

    x = rand_half(shape)
    gamma = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    beta = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    y = first(layernorm_cuda.forward(x, gamma, beta, EPS))
    check_finite("forward", y, "LayerNorm", shape)

    x_ref = x.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)

    y_ref_fp32 = F.layer_norm(
        x_ref,
        normalized_shape=(D,),
        weight=gamma_ref,
        bias=beta_ref,
        eps=EPS,
    )

    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, "LayerNorm", shape, 5e-2, 5e-2)

    out = layernorm_cuda.backward(dy, x, gamma, beta, EPS)
    dx = out[0]
    dgamma = out[1]
    dbeta = out[2]

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, "LayerNorm", shape)
    check_finite("dgamma", dgamma, "LayerNorm", shape)
    check_finite("dbeta", dbeta, "LayerNorm", shape)

    dx_err = check_close("dx", dx, x_ref.grad, "LayerNorm", shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, "LayerNorm", shape, 5e-1, 8e-2)
    db_err = check_close("dbeta", dbeta, beta_ref.grad, "LayerNorm", shape, 5e-1, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dgamma={dg_err:.4g}, dbeta={db_err:.4g}"



def test_softmax_shape(B, H, N, D):
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(softmax_cuda.forward(x))
    check_finite("forward", y, "Softmax", shape)

    row_sum_err = (y.float().sum(dim=-1) - 1.0).abs().max().item()
    if row_sum_err > 5e-2:
        raise AssertionError(f"Softmax {shape} row sum failed: max row_sum_err={row_sum_err}")

    x_ref = x.detach().float().requires_grad_(True)

    y_ref_fp32 = F.softmax(x_ref, dim=-1)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, "Softmax", shape, 5e-2, 5e-2)

    out = softmax_cuda.backward(dy, x)
    dx = out[0]

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, "Softmax", shape)

    dx_err = check_close("dx", dx, x_ref.grad, "Softmax", shape, 8e-2, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, row_sum={row_sum_err:.4g}"



def test_gelu_shape(B, H, N, D):
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(gelu_cuda.forward(x))
    check_finite("forward", y, "GELU", shape)

    x_ref = x.detach().float().requires_grad_(True)

    y_ref_fp32 = F.gelu(x_ref, approximate="none")
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, "GELU", shape, 5e-2, 5e-2)

    out = gelu_cuda.backward(dy, x)
    dx = out[0]

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, "GELU", shape)

    dx_err = check_close("dx", dx, x_ref.grad, "GELU", shape, 6e-2, 6e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}"



def test_silu_shape(B, H, N, D):
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(silu_cuda.forward(x))
    check_finite("forward", y, "SiLU", shape)

    x_ref = x.detach().float().requires_grad_(True)

    y_ref_fp32 = F.silu(x_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, "SiLU", shape, 5e-2, 5e-2)

    out = silu_cuda.backward(dy, x)
    dx = out[0]

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, "SiLU", shape)

    dx_err = check_close("dx", dx, x_ref.grad, "SiLU", shape, 6e-2, 6e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}"



def test_fused_bias_gelu_shape(B, H, N, D):
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    bias = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    y = first(fused_bias_gelu_cuda.forward(x, bias))
    check_finite("forward", y, "FusedBiasGELU", shape)

    x_ref = x.detach().float().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)

    y_ref_fp32 = F.gelu(x_ref + bias_ref, approximate="none")
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, "FusedBiasGELU", shape, 5e-2, 5e-2)

    out = fused_bias_gelu_cuda.backward(dy, x, bias)
    dx = out[0]
    dbias = out[1]

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, "FusedBiasGELU", shape)
    check_finite("dbias", dbias, "FusedBiasGELU", shape)

    dx_err = check_close("dx", dx, x_ref.grad, "FusedBiasGELU", shape, 6e-2, 6e-2)
    db_err = check_close("dbias", dbias, bias_ref.grad, "FusedBiasGELU", shape, 5e-1, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dbias={db_err:.4g}"



def ref_fused_residual_rmsnorm(x, residual, gamma):
    z = x + residual
    rms = torch.sqrt(torch.mean(z * z, dim=-1, keepdim=True) + EPS)
    return z / rms * gamma


def test_fused_residual_rmsnorm_shape(B, H, N, D):
    shape = (B, H, N, D)

    x = rand_half(shape)
    residual = rand_half(shape)
    gamma = rand_half((D,))
    dy = rand_half(shape)

    y = first(fused_residual_rmsnorm_cuda.forward(x, residual, gamma, EPS))
    check_finite("forward", y, "FusedResidualRMSNorm", shape)

    x_ref = x.detach().float().requires_grad_(True)
    residual_ref = residual.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)

    y_ref_fp32 = ref_fused_residual_rmsnorm(x_ref, residual_ref, gamma_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, "FusedResidualRMSNorm", shape, 5e-2, 5e-2)

    out = fused_residual_rmsnorm_cuda.backward(dy, x, residual, gamma, EPS)
    dx = out[0]
    dresidual = out[1]
    dgamma = out[2]

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, "FusedResidualRMSNorm", shape)
    check_finite("dresidual", dresidual, "FusedResidualRMSNorm", shape)
    check_finite("dgamma", dgamma, "FusedResidualRMSNorm", shape)

    dx_err = check_close("dx", dx, x_ref.grad, "FusedResidualRMSNorm", shape, 8e-2, 8e-2)
    dr_err = check_close("dresidual", dresidual, residual_ref.grad, "FusedResidualRMSNorm", shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, "FusedResidualRMSNorm", shape, 5e-1, 8e-2)

    dx_dr_err = (dx.float() - dresidual.float()).abs().max().item()
    if dx_dr_err != 0.0:
        raise AssertionError(
            f"FusedResidualRMSNorm {shape} failed dx == dresidual check. "
            f"max_err={dx_dr_err}"
        )

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dres={dr_err:.4g}, dgamma={dg_err:.4g}"


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    torch.manual_seed(0)

    shapes_d8 = [
        (1, 1, 1, 8),
        (1, 1, 1, 64),
        (1, 1, 17, 64),
        (1, 1, 31, 128),
        (1, 4, 31, 64),
        (2, 4, 63, 64),
        (1, 1, 128, 64),
        (1, 1, 256, 64),
        (2, 1, 512, 128),
        (1, 2, 64, 256),
    ]

    shapes_any = [
        (1, 1, 1, 1),
        (1, 1, 1, 17),
        (1, 1, 17, 10),
        (1, 1, 31, 33),
        (1, 1, 63, 77),
        (2, 4, 31, 33),
        (1, 1, 128, 64),
        (1, 4, 128, 512),
        (1, 2, 64, 768),
    ]

    print("\n========== RMSNorm ==========")
    for s in shapes_d8:
        run_case("RMSNorm", s, lambda s=s: test_rmsnorm_shape(*s))

    print("\n========== LayerNorm ==========")
    for s in shapes_any:
        run_case("LayerNorm", s, lambda s=s: test_layernorm_shape(*s))

    print("\n========== Softmax ==========")
    for s in shapes_any + [(1, 1, 32, 1024)]:
        run_case("Softmax", s, lambda s=s: test_softmax_shape(*s))

    print("\n========== GELU exact ==========")
    for s in shapes_any:
        run_case("GELU", s, lambda s=s: test_gelu_shape(*s))

    print("\n========== SiLU ==========")
    for s in shapes_any:
        run_case("SiLU", s, lambda s=s: test_silu_shape(*s))

    print("\n========== Fused bias + GELU exact ==========")
    for s in shapes_any:
        run_case("FusedBiasGELU", s, lambda s=s: test_fused_bias_gelu_shape(*s))

    print("\n========== Fused residual + RMSNorm ==========")
    for s in shapes_d8:
        run_case("FusedResidualRMSNorm", s, lambda s=s: test_fused_residual_rmsnorm_shape(*s))

    print("\n========== SUMMARY ==========")

    if len(FAILURES) == 0:
        print("ALL CUDA KERNEL TESTS PASSED")
        return

    print(f"{len(FAILURES)} TESTS FAILED:\n")
    for i, (kernel, shape, reason) in enumerate(FAILURES, 1):
        print(f"{i}. {kernel} {shape}")
        print(reason)
        print()

    raise SystemExit(1)


if __name__ == "__main__":
    main()
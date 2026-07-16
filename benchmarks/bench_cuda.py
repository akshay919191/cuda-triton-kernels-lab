import sys
import math
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
    CUDA_DIR / "rope_kernel",
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
import rope_cuda


EPS = 1e-5
FAILURES = []


def first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def rand_half(shape, scale=1.0):
    return (torch.randn(*shape, device="cuda", dtype=torch.float16) * scale).contiguous()


def sync():
    torch.cuda.synchronize()


def bench_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()

    sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    sync()

    return start.elapsed_time(end) / iters


def choose_iters(numel):
    if numel <= 8192:
        return 300
    if numel <= 131072:
        return 150
    if numel <= 1048576:
        return 80
    return 40


def gbps(bytes_moved, ms):
    if ms <= 0:
        return 0.0
    return bytes_moved / (ms * 1e-3) / 1e9


def check_finite(name, t, kernel, shape):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero()[0].tolist()
        raise AssertionError(
            f"{kernel} {shape} failed finite check for {name}. "
            f"first_bad_index={bad}, value={t[tuple(bad)].item()}"
        )


def check_close(name, custom, ref, kernel, shape, atol, rtol):
    custom_f = custom.float()
    ref_f = ref.float()

    if custom_f.shape != ref_f.shape:
        raise AssertionError(
            f"{kernel} {shape} failed {name}: shape mismatch. "
            f"custom={tuple(custom_f.shape)}, ref={tuple(ref_f.shape)}"
        )

    diff = (custom_f - ref_f).abs()
    max_err = diff.max().item()
    idx = diff.argmax().item()

    custom_val = custom_f.flatten()[idx].item()
    ref_val = ref_f.flatten()[idx].item()

    if not torch.allclose(custom_f, ref_f, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{kernel} {shape} failed {name}\n"
            f"  max_err={max_err}\n"
            f"  flat_idx={idx}\n"
            f"  custom={custom_val}\n"
            f"  ref={ref_val}\n"
            f"  atol={atol}, rtol={rtol}"
        )

    return max_err


def check_not_mutated(name, before, after, kernel, shape):
    err = (before.float() - after.float()).abs().max().item()
    if err != 0.0:
        raise AssertionError(
            f"{kernel} {shape} mutated {name}. max_change={err}"
        )


def run_correctness_case(kernel, shape, fn):
    try:
        msg = fn()
        print(f"  CORRECT PASS {shape} | {msg}")
    except Exception as e:
        print(f"  CORRECT FAIL {shape} | {e}")
        FAILURES.append((kernel, shape, "correctness", str(e)))


def run_bench_case(kernel, shape, fn):
    try:
        msg = fn()
        print(f"  BENCH {shape} | {msg}")
    except Exception as e:
        print(f"  BENCH FAIL {shape} | {e}")
        FAILURES.append((kernel, shape, "benchmark", str(e)))


def print_header(name):
    print(f"\n========== {name} ==========")


# ============================================================
# RMSNorm
# ============================================================

def ref_rmsnorm(x, gamma):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + EPS)
    return x / rms * gamma


def correctness_rmsnorm(B, H, N, D):
    kernel = "RMSNorm"
    shape = (B, H, N, D)

    x = rand_half(shape)
    gamma = rand_half((D,))
    dy = rand_half(shape)

    y = first(rmsnorm_cuda.forward(x, gamma, EPS))
    sync()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)

    y_ref_fp32 = ref_rmsnorm(x_ref, gamma_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = rmsnorm_cuda.backward(dy, x, gamma, EPS)
    dx, dgamma = out[0], out[1]
    sync()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)
    check_finite("dgamma", dgamma, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, kernel, shape, 5e-1, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dgamma={dg_err:.4g}"


def bench_rmsnorm(B, H, N, D):
    shape = (B, H, N, D)
    elems = B * H * N * D
    iters = choose_iters(elems)

    x = rand_half(shape)
    gamma = rand_half((D,))
    dy = rand_half(shape)

    x_ref = x.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)
    y_ref = ref_rmsnorm(x_ref, gamma_ref)

    custom_fwd = lambda: first(rmsnorm_cuda.forward(x, gamma, EPS))
    custom_bwd = lambda: rmsnorm_cuda.backward(dy, x, gamma, EPS)

    torch_fwd = lambda: ref_rmsnorm(x.float(), gamma.float()).to(torch.float16)
    torch_bwd = lambda: torch.autograd.grad(
        y_ref,
        (x_ref, gamma_ref),
        grad_outputs=dy.float(),
        retain_graph=True,
    )

    cf = bench_ms(custom_fwd, iters=iters)
    tf = bench_ms(torch_fwd, iters=iters)
    cb = bench_ms(custom_bwd, iters=iters)
    tb = bench_ms(torch_bwd, iters=iters)

    fwd_bytes = elems * 4
    bwd_bytes = elems * 6 + D * 4

    return (
        f"fwd_custom={cf:.4f}ms, fwd_torch={tf:.4f}ms, speedup={tf/cf:.2f}x, "
        f"bwd_custom={cb:.4f}ms, bwd_torch={tb:.4f}ms, speedup={tb/cb:.2f}x, "
        f"GB/s_fwd={gbps(fwd_bytes, cf):.1f}, GB/s_bwd={gbps(bwd_bytes, cb):.1f}"
    )


# ============================================================
# LayerNorm
# ============================================================

def correctness_layernorm(B, H, N, D):
    kernel = "LayerNorm"
    shape = (B, H, N, D)

    x = rand_half(shape)
    gamma = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    beta = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    y = first(layernorm_cuda.forward(x, gamma, beta, EPS))
    sync()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)

    y_ref_fp32 = F.layer_norm(x_ref, (D,), gamma_ref, beta_ref, EPS)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = layernorm_cuda.backward(dy, x, gamma, beta, EPS)
    dx, dgamma, dbeta = out[0], out[1], out[2]
    sync()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)
    check_finite("dgamma", dgamma, kernel, shape)
    check_finite("dbeta", dbeta, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, kernel, shape, 5e-1, 8e-2)
    db_err = check_close("dbeta", dbeta, beta_ref.grad, kernel, shape, 5e-1, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dgamma={dg_err:.4g}, dbeta={db_err:.4g}"


def bench_layernorm(B, H, N, D):
    shape = (B, H, N, D)
    elems = B * H * N * D
    iters = choose_iters(elems)

    x = rand_half(shape)
    gamma = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    beta = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    x_ref = x.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    y_ref = F.layer_norm(x_ref, (D,), gamma_ref, beta_ref, EPS)

    custom_fwd = lambda: first(layernorm_cuda.forward(x, gamma, beta, EPS))
    custom_bwd = lambda: layernorm_cuda.backward(dy, x, gamma, beta, EPS)

    torch_fwd = lambda: F.layer_norm(x.float(), (D,), gamma, beta, EPS).to(torch.float16)
    torch_bwd = lambda: torch.autograd.grad(
        y_ref,
        (x_ref, gamma_ref, beta_ref),
        grad_outputs=dy.float(),
        retain_graph=True,
    )

    cf = bench_ms(custom_fwd, iters=iters)
    tf = bench_ms(torch_fwd, iters=iters)
    cb = bench_ms(custom_bwd, iters=iters)
    tb = bench_ms(torch_bwd, iters=iters)

    fwd_bytes = elems * 4 + D * 8
    bwd_bytes = elems * 6 + D * 8

    return (
        f"fwd_custom={cf:.4f}ms, fwd_torch={tf:.4f}ms, speedup={tf/cf:.2f}x, "
        f"bwd_custom={cb:.4f}ms, bwd_torch={tb:.4f}ms, speedup={tb/cb:.2f}x, "
        f"GB/s_fwd={gbps(fwd_bytes, cf):.1f}, GB/s_bwd={gbps(bwd_bytes, cb):.1f}"
    )


# ============================================================
# Softmax
# ============================================================

def correctness_softmax(B, H, N, D):
    kernel = "Softmax"
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(softmax_cuda.forward(x))
    sync()

    check_finite("forward", y, kernel, shape)

    row_sum_err = (y.float().sum(dim=-1) - 1.0).abs().max().item()
    if row_sum_err > 5e-2:
        raise AssertionError(f"row sum failed. max_row_sum_err={row_sum_err}")

    x_ref = x.detach().float().requires_grad_(True)

    y_ref_fp32 = F.softmax(x_ref, dim=-1)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = softmax_cuda.backward(dy, x)
    dx = out[0]
    sync()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 8e-2, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, row_sum={row_sum_err:.4g}"


def bench_softmax(B, H, N, D):
    shape = (B, H, N, D)
    elems = B * H * N * D
    iters = choose_iters(elems)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    x_ref = x.detach().float().requires_grad_(True)
    y_ref = F.softmax(x_ref, dim=-1)

    custom_fwd = lambda: first(softmax_cuda.forward(x))
    custom_bwd = lambda: softmax_cuda.backward(dy, x)

    torch_fwd = lambda: F.softmax(x.float(), dim=-1).to(torch.float16)
    torch_bwd = lambda: torch.autograd.grad(
        y_ref,
        x_ref,
        grad_outputs=dy.float(),
        retain_graph=True,
    )

    cf = bench_ms(custom_fwd, iters=iters)
    tf = bench_ms(torch_fwd, iters=iters)
    cb = bench_ms(custom_bwd, iters=iters)
    tb = bench_ms(torch_bwd, iters=iters)

    fwd_bytes = elems * 4
    bwd_bytes = elems * 6

    return (
        f"fwd_custom={cf:.4f}ms, fwd_torch={tf:.4f}ms, speedup={tf/cf:.2f}x, "
        f"bwd_custom={cb:.4f}ms, bwd_torch={tb:.4f}ms, speedup={tb/cb:.2f}x, "
        f"GB/s_fwd={gbps(fwd_bytes, cf):.1f}, GB/s_bwd={gbps(bwd_bytes, cb):.1f}"
    )


# ============================================================
# GELU exact
# ============================================================

def correctness_gelu(B, H, N, D):
    kernel = "GELU exact"
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(gelu_cuda.forward(x))
    sync()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)
    y_ref_fp32 = F.gelu(x_ref, approximate="none")
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = gelu_cuda.backward(dy, x)
    dx = out[0]
    sync()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 6e-2, 6e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}"


def bench_gelu(B, H, N, D):
    shape = (B, H, N, D)
    elems = B * H * N * D
    iters = choose_iters(elems)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    x_ref = x.detach().float().requires_grad_(True)
    y_ref = F.gelu(x_ref, approximate="none")

    custom_fwd = lambda: first(gelu_cuda.forward(x))
    custom_bwd = lambda: gelu_cuda.backward(dy, x)

    torch_fwd = lambda: F.gelu(x.float(), approximate="none").to(torch.float16)
    torch_bwd = lambda: torch.autograd.grad(
        y_ref,
        x_ref,
        grad_outputs=dy.float(),
        retain_graph=True,
    )

    cf = bench_ms(custom_fwd, iters=iters)
    tf = bench_ms(torch_fwd, iters=iters)
    cb = bench_ms(custom_bwd, iters=iters)
    tb = bench_ms(torch_bwd, iters=iters)

    fwd_bytes = elems * 4
    bwd_bytes = elems * 6

    return (
        f"fwd_custom={cf:.4f}ms, fwd_torch={tf:.4f}ms, speedup={tf/cf:.2f}x, "
        f"bwd_custom={cb:.4f}ms, bwd_torch={tb:.4f}ms, speedup={tb/cb:.2f}x, "
        f"GB/s_fwd={gbps(fwd_bytes, cf):.1f}, GB/s_bwd={gbps(bwd_bytes, cb):.1f}"
    )


# ============================================================
# SiLU
# ============================================================

def correctness_silu(B, H, N, D):
    kernel = "SiLU"
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(silu_cuda.forward(x))
    sync()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)
    y_ref_fp32 = F.silu(x_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = silu_cuda.backward(dy, x)
    dx = out[0]
    sync()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 6e-2, 6e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}"


def bench_silu(B, H, N, D):
    shape = (B, H, N, D)
    elems = B * H * N * D
    iters = choose_iters(elems)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    x_ref = x.detach().float().requires_grad_(True)
    y_ref = F.silu(x_ref)

    custom_fwd = lambda: first(silu_cuda.forward(x))
    custom_bwd = lambda: silu_cuda.backward(dy, x)

    torch_fwd = lambda: F.silu(x.float()).to(torch.float16)
    torch_bwd = lambda: torch.autograd.grad(
        y_ref,
        x_ref,
        grad_outputs=dy.float(),
        retain_graph=True,
    )

    cf = bench_ms(custom_fwd, iters=iters)
    tf = bench_ms(torch_fwd, iters=iters)
    cb = bench_ms(custom_bwd, iters=iters)
    tb = bench_ms(torch_bwd, iters=iters)

    fwd_bytes = elems * 4
    bwd_bytes = elems * 6

    return (
        f"fwd_custom={cf:.4f}ms, fwd_torch={tf:.4f}ms, speedup={tf/cf:.2f}x, "
        f"bwd_custom={cb:.4f}ms, bwd_torch={tb:.4f}ms, speedup={tb/cb:.2f}x, "
        f"GB/s_fwd={gbps(fwd_bytes, cf):.1f}, GB/s_bwd={gbps(bwd_bytes, cb):.1f}"
    )


# ============================================================
# Fused bias + GELU exact
# ============================================================

def correctness_fused_bias_gelu(B, H, N, D):
    kernel = "FusedBiasGELU exact"
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    bias = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    y = first(fused_bias_gelu_cuda.forward(x, bias))
    sync()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)

    y_ref_fp32 = F.gelu(x_ref + bias_ref, approximate="none")
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = fused_bias_gelu_cuda.backward(dy, x, bias)
    dx, dbias = out[0], out[1]
    sync()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)
    check_finite("dbias", dbias, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 6e-2, 6e-2)
    db_err = check_close("dbias", dbias, bias_ref.grad, kernel, shape, 5e-1, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dbias={db_err:.4g}"


def bench_fused_bias_gelu(B, H, N, D):
    shape = (B, H, N, D)
    elems = B * H * N * D
    iters = choose_iters(elems)

    x = rand_half(shape, scale=2.0)
    bias = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    x_ref = x.detach().float().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    y_ref = F.gelu(x_ref + bias_ref, approximate="none")

    custom_fwd = lambda: first(fused_bias_gelu_cuda.forward(x, bias))
    custom_bwd = lambda: fused_bias_gelu_cuda.backward(dy, x, bias)

    torch_fwd = lambda: F.gelu(x.float() + bias, approximate="none").to(torch.float16)
    torch_bwd = lambda: torch.autograd.grad(
        y_ref,
        (x_ref, bias_ref),
        grad_outputs=dy.float(),
        retain_graph=True,
    )

    cf = bench_ms(custom_fwd, iters=iters)
    tf = bench_ms(torch_fwd, iters=iters)
    cb = bench_ms(custom_bwd, iters=iters)
    tb = bench_ms(torch_bwd, iters=iters)

    fwd_bytes = elems * 6 + D * 4
    bwd_bytes = elems * 8 + D * 4

    return (
        f"fwd_custom={cf:.4f}ms, fwd_torch={tf:.4f}ms, speedup={tf/cf:.2f}x, "
        f"bwd_custom={cb:.4f}ms, bwd_torch={tb:.4f}ms, speedup={tb/cb:.2f}x, "
        f"GB/s_fwd={gbps(fwd_bytes, cf):.1f}, GB/s_bwd={gbps(bwd_bytes, cb):.1f}"
    )


# ============================================================
# Fused residual + RMSNorm
# ============================================================

def ref_fused_residual_rmsnorm(x, residual, gamma):
    z = x + residual
    rms = torch.sqrt(torch.mean(z * z, dim=-1, keepdim=True) + EPS)
    return z / rms * gamma


def correctness_fused_residual_rmsnorm(B, H, N, D):
    kernel = "FusedResidualRMSNorm"
    shape = (B, H, N, D)

    x = rand_half(shape)
    residual = rand_half(shape)
    gamma = rand_half((D,))
    dy = rand_half(shape)

    y = first(fused_residual_rmsnorm_cuda.forward(x, residual, gamma, EPS))
    sync()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)
    residual_ref = residual.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)

    y_ref_fp32 = ref_fused_residual_rmsnorm(x_ref, residual_ref, gamma_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = fused_residual_rmsnorm_cuda.backward(dy, x, residual, gamma, EPS)
    dx, dresidual, dgamma = out[0], out[1], out[2]
    sync()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)
    check_finite("dresidual", dresidual, kernel, shape)
    check_finite("dgamma", dgamma, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 8e-2, 8e-2)
    dr_err = check_close("dresidual", dresidual, residual_ref.grad, kernel, shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, kernel, shape, 5e-1, 8e-2)

    dx_dr_err = (dx.float() - dresidual.float()).abs().max().item()
    if dx_dr_err != 0.0:
        raise AssertionError(f"dx != dresidual. max_err={dx_dr_err}")

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dres={dr_err:.4g}, dgamma={dg_err:.4g}"


def bench_fused_residual_rmsnorm(B, H, N, D):
    shape = (B, H, N, D)
    elems = B * H * N * D
    iters = choose_iters(elems)

    x = rand_half(shape)
    residual = rand_half(shape)
    gamma = rand_half((D,))
    dy = rand_half(shape)

    x_ref = x.detach().float().requires_grad_(True)
    residual_ref = residual.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)
    y_ref = ref_fused_residual_rmsnorm(x_ref, residual_ref, gamma_ref)

    custom_fwd = lambda: first(fused_residual_rmsnorm_cuda.forward(x, residual, gamma, EPS))
    custom_bwd = lambda: fused_residual_rmsnorm_cuda.backward(dy, x, residual, gamma, EPS)

    torch_fwd = lambda: ref_fused_residual_rmsnorm(x.float(), residual.float(), gamma.float()).to(torch.float16)
    torch_bwd = lambda: torch.autograd.grad(
        y_ref,
        (x_ref, residual_ref, gamma_ref),
        grad_outputs=dy.float(),
        retain_graph=True,
    )

    cf = bench_ms(custom_fwd, iters=iters)
    tf = bench_ms(torch_fwd, iters=iters)
    cb = bench_ms(custom_bwd, iters=iters)
    tb = bench_ms(torch_bwd, iters=iters)

    fwd_bytes = elems * 6 + D * 2
    bwd_bytes = elems * 10 + D * 4

    return (
        f"fwd_custom={cf:.4f}ms, fwd_torch={tf:.4f}ms, speedup={tf/cf:.2f}x, "
        f"bwd_custom={cb:.4f}ms, bwd_torch={tb:.4f}ms, speedup={tb/cb:.2f}x, "
        f"GB/s_fwd={gbps(fwd_bytes, cf):.1f}, GB/s_bwd={gbps(bwd_bytes, cb):.1f}"
    )


# ============================================================
# RoPE
# ============================================================

def manual_rope_cache(max_seq_len, rotary_dim, device):
    half = rotary_dim // 2

    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    dim = torch.arange(half, device=device, dtype=torch.float32)

    inv_freq = 1.0 / (10000.0 ** (2.0 * dim / rotary_dim))
    freqs = torch.outer(pos, inv_freq)

    return torch.cos(freqs).float().contiguous(), torch.sin(freqs).float().contiguous()


def extension_rope_cache(reference, max_seq_len, rotary_dim):
    out = rope_cuda.build_cache(reference, max_seq_len, rotary_dim, 10000.0)

    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise AssertionError("build_cache should return [cos_cache, sin_cache]")

    return out[0].float().contiguous(), out[1].float().contiguous()


def ref_rope_forward(x, cos_cache, sin_cache, rotary_dim, position_ids=None, position_offset=0):
    B, H, N, D = x.shape
    half = rotary_dim // 2

    x = x.float()

    x_rot = x[..., :rotary_dim]
    x_tail = x[..., rotary_dim:]

    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:rotary_dim]

    if position_ids is None:
        pos = torch.arange(N, device=x.device, dtype=torch.long) + int(position_offset)
        cos = cos_cache[pos][None, None, :, :]
        sin = sin_cache[pos][None, None, :, :]
    else:
        pos = position_ids.long()
        cos = cos_cache[pos][:, None, :, :]
        sin = sin_cache[pos][:, None, :, :]

    y1 = x1 * cos.float() - x2 * sin.float()
    y2 = x1 * sin.float() + x2 * cos.float()

    y_rot = torch.cat([y1, y2], dim=-1)

    if rotary_dim < D:
        return torch.cat([y_rot, x_tail], dim=-1)

    return y_rot


def rope_case_tensors(B, H, N, D, rotary_dim, mode):
    x = rand_half((B, H, N, D))
    dy = rand_half((B, H, N, D))

    max_seq_len = max(N + 16, 32)
    cos, sin = extension_rope_cache(x, max_seq_len, rotary_dim)

    if mode == "none":
        position_ids = None
        position_offset = 0
    elif mode == "offset":
        position_ids = None
        position_offset = 3
    elif mode == "position_ids":
        base = torch.arange(N, device="cuda", dtype=torch.int32).unsqueeze(0).repeat(B, 1)
        batch_offsets = torch.arange(B, device="cuda", dtype=torch.int32).unsqueeze(1)
        position_ids = (base + batch_offsets).contiguous()
        position_offset = 0
    else:
        raise AssertionError(f"unknown RoPE mode={mode}")

    return x, dy, cos, sin, position_ids, position_offset, max_seq_len


def correctness_rope(B, H, N, D, rotary_dim, mode):
    kernel = "RoPE"
    shape = (B, H, N, D, rotary_dim, mode)

    if rotary_dim > D:
        raise AssertionError(f"rotary_dim={rotary_dim} > D={D}")
    if rotary_dim % 2 != 0:
        raise AssertionError(f"rotary_dim must be even, got {rotary_dim}")

    x, dy, cos, sin, position_ids, position_offset, max_seq_len = rope_case_tensors(
        B, H, N, D, rotary_dim, mode
    )

    cos_ref, sin_ref = manual_rope_cache(max_seq_len, rotary_dim, x.device)
    cos_err = check_close("cos_cache", cos, cos_ref, "RoPE build_cache", shape, 5e-5, 5e-5)
    sin_err = check_close("sin_cache", sin, sin_ref, "RoPE build_cache", shape, 5e-5, 5e-5)

    y = first(rope_cuda.forward(x, position_ids, cos, sin, rotary_dim, position_offset))
    sync()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)

    y_ref_fp32 = ref_rope_forward(
        x_ref,
        cos.detach(),
        sin.detach(),
        rotary_dim,
        position_ids=position_ids,
        position_offset=position_offset,
    )

    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    dx = first(rope_cuda.backward(dy, position_ids, cos, sin, rotary_dim, position_offset))
    sync()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 6e-2, 6e-2)

    if rotary_dim < D:
        tail_fwd_err = (y[..., rotary_dim:].float() - x[..., rotary_dim:].float()).abs().max().item()
        tail_bwd_err = (dx[..., rotary_dim:].float() - dy[..., rotary_dim:].float()).abs().max().item()

        if tail_fwd_err != 0.0:
            raise AssertionError(f"forward tail passthrough failed. err={tail_fwd_err}")
        if tail_bwd_err != 0.0:
            raise AssertionError(f"backward tail passthrough failed. err={tail_bwd_err}")
    else:
        tail_fwd_err = 0.0
        tail_bwd_err = 0.0

    return (
        f"cache_cos={cos_err:.2g}, cache_sin={sin_err:.2g}, "
        f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, "
        f"tail_fwd={tail_fwd_err:.4g}, tail_bwd={tail_bwd_err:.4g}"
    )


def bench_rope(B, H, N, D, rotary_dim, mode):
    shape = (B, H, N, D, rotary_dim, mode)
    elems = B * H * N * D
    iters = choose_iters(elems)

    x, dy, cos, sin, position_ids, position_offset, max_seq_len = rope_case_tensors(
        B, H, N, D, rotary_dim, mode
    )

    x_ref = x.detach().float().requires_grad_(True)
    y_ref = ref_rope_forward(
        x_ref,
        cos.detach(),
        sin.detach(),
        rotary_dim,
        position_ids=position_ids,
        position_offset=position_offset,
    )

    custom_cache = lambda: rope_cuda.build_cache(x, max_seq_len, rotary_dim, 10000.0)
    torch_cache = lambda: manual_rope_cache(max_seq_len, rotary_dim, x.device)

    custom_fwd = lambda: first(rope_cuda.forward(x, position_ids, cos, sin, rotary_dim, position_offset))
    custom_bwd = lambda: first(rope_cuda.backward(dy, position_ids, cos, sin, rotary_dim, position_offset))

    torch_fwd = lambda: ref_rope_forward(
        x.float(),
        cos,
        sin,
        rotary_dim,
        position_ids=position_ids,
        position_offset=position_offset,
    ).to(torch.float16)

    torch_bwd = lambda: torch.autograd.grad(
        y_ref,
        x_ref,
        grad_outputs=dy.float(),
        retain_graph=True,
    )

    cc = bench_ms(custom_cache, iters=max(30, iters // 4))
    tc = bench_ms(torch_cache, iters=max(30, iters // 4))

    cf = bench_ms(custom_fwd, iters=iters)
    tf = bench_ms(torch_fwd, iters=iters)

    cb = bench_ms(custom_bwd, iters=iters)
    tb = bench_ms(torch_bwd, iters=iters)

    fwd_bytes = elems * 4 + N * rotary_dim * 4
    bwd_bytes = elems * 4 + N * rotary_dim * 4

    return (
        f"cache_custom={cc:.4f}ms, cache_torch={tc:.4f}ms, speedup={tc/cc:.2f}x, "
        f"fwd_custom={cf:.4f}ms, fwd_torch={tf:.4f}ms, speedup={tf/cf:.2f}x, "
        f"bwd_custom={cb:.4f}ms, bwd_torch={tb:.4f}ms, speedup={tb/cb:.2f}x, "
        f"GB/s_fwd={gbps(fwd_bytes, cf):.1f}, GB/s_bwd={gbps(bwd_bytes, cb):.1f}"
    )


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    torch.manual_seed(0)

    shapes_d8_correct = [
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

    shapes_any_correct = [
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

    bench_shapes_norm = [
        (1, 1, 128, 64),
        (1, 4, 512, 128),
        (2, 4, 1024, 128),
    ]

    bench_shapes_any = [
        (1, 1, 128, 64),
        (1, 4, 512, 128),
        (2, 4, 1024, 128),
        (1, 4, 512, 512),
    ]

    rope_correct = [
        (1, 1, 1, 32, 32, "none"),
        (1, 1, 17, 32, 32, "none"),
        (1, 4, 31, 64, 64, "none"),
        (2, 4, 63, 64, 32, "none"),
        (1, 2, 128, 128, 64, "none"),
        (2, 1, 256, 128, 128, "none"),
        (1, 1, 17, 32, 32, "offset"),
        (2, 4, 63, 64, 32, "offset"),
        (1, 1, 17, 32, 32, "position_ids"),
        (2, 4, 63, 64, 32, "position_ids"),
    ]

    rope_bench = [
        (1, 4, 512, 64, 64, "none"),
        (2, 4, 1024, 128, 64, "none"),
        (2, 4, 1024, 128, 128, "position_ids"),
    ]

    print_header("RMSNorm correctness")
    for s in shapes_d8_correct:
        run_correctness_case("RMSNorm", s, lambda s=s: correctness_rmsnorm(*s))

    print_header("RMSNorm benchmark")
    for s in bench_shapes_norm:
        run_bench_case("RMSNorm", s, lambda s=s: bench_rmsnorm(*s))

    print_header("LayerNorm correctness")
    for s in shapes_any_correct:
        run_correctness_case("LayerNorm", s, lambda s=s: correctness_layernorm(*s))

    print_header("LayerNorm benchmark")
    for s in bench_shapes_any:
        run_bench_case("LayerNorm", s, lambda s=s: bench_layernorm(*s))

    print_header("Softmax correctness")
    for s in shapes_any_correct + [(1, 1, 32, 1024)]:
        run_correctness_case("Softmax", s, lambda s=s: correctness_softmax(*s))

    print_header("Softmax benchmark")
    for s in bench_shapes_any + [(1, 1, 512, 1024)]:
        run_bench_case("Softmax", s, lambda s=s: bench_softmax(*s))

    print_header("GELU exact correctness")
    for s in shapes_any_correct:
        run_correctness_case("GELU exact", s, lambda s=s: correctness_gelu(*s))

    print_header("GELU exact benchmark")
    for s in bench_shapes_any:
        run_bench_case("GELU exact", s, lambda s=s: bench_gelu(*s))

    print_header("SiLU correctness")
    for s in shapes_any_correct:
        run_correctness_case("SiLU", s, lambda s=s: correctness_silu(*s))

    print_header("SiLU benchmark")
    for s in bench_shapes_any:
        run_bench_case("SiLU", s, lambda s=s: bench_silu(*s))

    print_header("Fused bias + GELU exact correctness")
    for s in shapes_any_correct:
        run_correctness_case("FusedBiasGELU exact", s, lambda s=s: correctness_fused_bias_gelu(*s))

    print_header("Fused bias + GELU exact benchmark")
    for s in bench_shapes_any:
        run_bench_case("FusedBiasGELU exact", s, lambda s=s: bench_fused_bias_gelu(*s))

    print_header("Fused residual + RMSNorm correctness")
    for s in shapes_d8_correct:
        run_correctness_case("FusedResidualRMSNorm", s, lambda s=s: correctness_fused_residual_rmsnorm(*s))

    print_header("Fused residual + RMSNorm benchmark")
    for s in bench_shapes_norm:
        run_bench_case("FusedResidualRMSNorm", s, lambda s=s: bench_fused_residual_rmsnorm(*s))

    print_header("RoPE correctness")
    for s in rope_correct:
        run_correctness_case("RoPE", s, lambda s=s: correctness_rope(*s))

    print_header("RoPE benchmark")
    for s in rope_bench:
        run_bench_case("RoPE", s, lambda s=s: bench_rope(*s))

    print("\n========== SUMMARY ==========")

    if len(FAILURES) == 0:
        print("ALL CORRECTNESS + BENCHMARK CHECKS PASSED")
        return

    print(f"{len(FAILURES)} FAILURES:\n")

    for i, (kernel, shape, phase, reason) in enumerate(FAILURES, 1):
        print(f"{i}. {kernel} {shape} [{phase}]")
        print(reason)
        print()

    raise SystemExit(1)


if __name__ == "__main__":
    main()



import os
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

RUN_PROFILER_TABLES = os.environ.get("PROFILE_TABLES", "1") != "0"
PROFILE_ITERS = int(os.environ.get("PROFILE_ITERS", "10"))
BENCH_WARMUP = int(os.environ.get("BENCH_WARMUP", "30"))
BENCH_ITERS = int(os.environ.get("BENCH_ITERS", "100"))


BOUND_GUESS = {
    "RMSNorm": "reduction + memory bandwidth + dgamma atomic in backward",
    "LayerNorm": "mean/variance reductions + sync + dgamma/dbeta atomics",
    "Softmax": "max/sum reductions + exp + memory traffic",
    "GELU exact": "erf/exp special-function math + memory bandwidth",
    "SiLU": "exp/sigmoid special-function math + memory bandwidth",
    "FusedBiasGELU exact": "erf/exp math + dbias atomic + saved memory traffic from fusion",
    "FusedResidualRMSNorm": "RMS reduction + shared memory sync + dgamma atomic + extra dx/dres writes",
    "RoPE": "memory bandwidth + cos/sin cache reads + position indexing",
}


def first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def sync():
    torch.cuda.synchronize()


def rand_half(shape, scale=1.0):
    return (torch.randn(*shape, device="cuda", dtype=torch.float16) * scale).contiguous()


def gbps(bytes_moved, ms):
    if ms <= 0:
        return 0.0
    return bytes_moved / (ms * 1e-3) / 1e9


def time_cuda(fn, warmup=BENCH_WARMUP, iters=BENCH_ITERS):
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
            f"{kernel} {shape} {name} shape mismatch. "
            f"custom={tuple(custom_f.shape)}, ref={tuple(ref_f.shape)}"
        )

    diff = (custom_f - ref_f).abs()
    max_err = diff.max().item()
    idx = diff.argmax().item()

    if not torch.allclose(custom_f, ref_f, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{kernel} {shape} failed {name}\n"
            f"  max_err={max_err}\n"
            f"  flat_idx={idx}\n"
            f"  custom={custom_f.flatten()[idx].item()}\n"
            f"  ref={ref_f.flatten()[idx].item()}\n"
            f"  atol={atol}, rtol={rtol}"
        )

    return max_err


def profile_table(title, fn, iters=PROFILE_ITERS):
    if not RUN_PROFILER_TABLES:
        return

    try:
        from torch.profiler import profile, ProfilerActivity

        print(f"\n--- profiler table: {title} ---")

        for _ in range(5):
            fn()
        sync()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in range(iters):
                fn()

        sync()

        print(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=8,
            )
        )

    except Exception as e:
        print(f"\n--- profiler table skipped for {title}: {type(e).__name__}: {e} ---")


def print_kernel_header(name, shape):
    print("\n" + "=" * 90)
    print(f"{name} | shape={shape}")
    print("=" * 90)
    print(f"Likely bound: {BOUND_GUESS.get(name, 'unknown')}")


def print_result(name, fwd_custom, fwd_ref, bwd_custom, bwd_ref, fwd_gbps, bwd_gbps):
    print("\nSpeed:")
    print(f"  forward custom : {fwd_custom:.5f} ms")
    print(f"  forward ref    : {fwd_ref:.5f} ms")
    print(f"  forward speedup: {fwd_ref / fwd_custom:.2f}x")
    print(f"  forward approx : {fwd_gbps:.1f} GB/s")

    print(f"  backward custom : {bwd_custom:.5f} ms")
    print(f"  backward ref    : {bwd_ref:.5f} ms")
    print(f"  backward speedup: {bwd_ref / bwd_custom:.2f}x")
    print(f"  backward approx : {bwd_gbps:.1f} GB/s")


# ============================================================
# References
# ============================================================

def ref_rmsnorm(x, gamma):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + EPS)
    return x / rms * gamma


def ref_fused_residual_rmsnorm(x, residual, gamma):
    z = x + residual
    rms = torch.sqrt(torch.mean(z * z, dim=-1, keepdim=True) + EPS)
    return z / rms * gamma


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


def manual_rope_cache(max_seq_len, rotary_dim, device):
    half = rotary_dim // 2
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    dim = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (2.0 * dim / rotary_dim))
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs).float().contiguous(), torch.sin(freqs).float().contiguous()


# ============================================================
# Individual profiling blocks
# ============================================================

def profile_rmsnorm():
    name = "RMSNorm"
    shape = (2, 4, 1024, 128)
    B, H, N, D = shape
    print_kernel_header(name, shape)

    x = rand_half(shape)
    gamma = rand_half((D,))
    dy = rand_half(shape)

    y = first(rmsnorm_cuda.forward(x, gamma, EPS))
    x_ref = x.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)
    y_ref_fp32 = ref_rmsnorm(x_ref, gamma_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, name, shape, 5e-2, 5e-2)

    out = rmsnorm_cuda.backward(dy, x, gamma, EPS)
    dx, dgamma = out[0], out[1]
    y_ref_fp32.backward(dy.float())

    dx_err = check_close("dx", dx, x_ref.grad, name, shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, name, shape, 5e-1, 8e-2)

    print(f"Correctness: PASS | fwd={fwd_err:.4g}, dx={dx_err:.4g}, dgamma={dg_err:.4g}")

    x_ref_b = x.detach().float().requires_grad_(True)
    gamma_ref_b = gamma.detach().float().requires_grad_(True)
    y_ref_b = ref_rmsnorm(x_ref_b, gamma_ref_b)

    custom_fwd = lambda: first(rmsnorm_cuda.forward(x, gamma, EPS))
    custom_bwd = lambda: rmsnorm_cuda.backward(dy, x, gamma, EPS)
    ref_fwd = lambda: ref_rmsnorm(x.float(), gamma.float()).to(torch.float16)
    ref_bwd = lambda: torch.autograd.grad(y_ref_b, (x_ref_b, gamma_ref_b), grad_outputs=dy.float(), retain_graph=True)

    elems = B * H * N * D
    cf = time_cuda(custom_fwd)
    rf = time_cuda(ref_fwd)
    cb = time_cuda(custom_bwd)
    rb = time_cuda(ref_bwd)

    print_result(name, cf, rf, cb, rb, gbps(elems * 4, cf), gbps(elems * 6 + D * 4, cb))

    profile_table("RMSNorm custom forward", custom_fwd)
    profile_table("RMSNorm custom backward", custom_bwd)


def profile_layernorm():
    name = "LayerNorm"
    shape = (2, 4, 1024, 128)
    B, H, N, D = shape
    print_kernel_header(name, shape)

    x = rand_half(shape)
    gamma = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    beta = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    y = first(layernorm_cuda.forward(x, gamma, beta, EPS))

    x_ref = x.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    y_ref_fp32 = F.layer_norm(x_ref, (D,), gamma_ref, beta_ref, EPS)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, name, shape, 5e-2, 5e-2)

    out = layernorm_cuda.backward(dy, x, gamma, beta, EPS)
    dx, dgamma, dbeta = out[0], out[1], out[2]
    y_ref_fp32.backward(dy.float())

    dx_err = check_close("dx", dx, x_ref.grad, name, shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, name, shape, 5e-1, 8e-2)
    db_err = check_close("dbeta", dbeta, beta_ref.grad, name, shape, 5e-1, 8e-2)

    print(f"Correctness: PASS | fwd={fwd_err:.4g}, dx={dx_err:.4g}, dgamma={dg_err:.4g}, dbeta={db_err:.4g}")

    x_ref_b = x.detach().float().requires_grad_(True)
    gamma_ref_b = gamma.detach().clone().requires_grad_(True)
    beta_ref_b = beta.detach().clone().requires_grad_(True)
    y_ref_b = F.layer_norm(x_ref_b, (D,), gamma_ref_b, beta_ref_b, EPS)

    custom_fwd = lambda: first(layernorm_cuda.forward(x, gamma, beta, EPS))
    custom_bwd = lambda: layernorm_cuda.backward(dy, x, gamma, beta, EPS)
    ref_fwd = lambda: F.layer_norm(x.float(), (D,), gamma, beta, EPS).to(torch.float16)
    ref_bwd = lambda: torch.autograd.grad(y_ref_b, (x_ref_b, gamma_ref_b, beta_ref_b), grad_outputs=dy.float(), retain_graph=True)

    elems = B * H * N * D
    cf = time_cuda(custom_fwd)
    rf = time_cuda(ref_fwd)
    cb = time_cuda(custom_bwd)
    rb = time_cuda(ref_bwd)

    print_result(name, cf, rf, cb, rb, gbps(elems * 4 + D * 8, cf), gbps(elems * 6 + D * 8, cb))

    profile_table("LayerNorm custom forward", custom_fwd)
    profile_table("LayerNorm custom backward", custom_bwd)


def profile_softmax():
    name = "Softmax"
    shape = (1, 4, 512, 512)
    B, H, N, D = shape
    print_kernel_header(name, shape)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(softmax_cuda.forward(x))
    row_sum_err = (y.float().sum(dim=-1) - 1.0).abs().max().item()
    if row_sum_err > 5e-2:
        raise AssertionError(f"Softmax row sum failed. err={row_sum_err}")

    x_ref = x.detach().float().requires_grad_(True)
    y_ref_fp32 = F.softmax(x_ref, dim=-1)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, name, shape, 5e-2, 5e-2)

    dx = first(softmax_cuda.backward(dy, x))
    y_ref_fp32.backward(dy.float())

    dx_err = check_close("dx", dx, x_ref.grad, name, shape, 8e-2, 8e-2)

    print(f"Correctness: PASS | fwd={fwd_err:.4g}, dx={dx_err:.4g}, row_sum={row_sum_err:.4g}")

    x_ref_b = x.detach().float().requires_grad_(True)
    y_ref_b = F.softmax(x_ref_b, dim=-1)

    custom_fwd = lambda: first(softmax_cuda.forward(x))
    custom_bwd = lambda: softmax_cuda.backward(dy, x)
    ref_fwd = lambda: F.softmax(x.float(), dim=-1).to(torch.float16)
    ref_bwd = lambda: torch.autograd.grad(y_ref_b, x_ref_b, grad_outputs=dy.float(), retain_graph=True)

    elems = B * H * N * D
    cf = time_cuda(custom_fwd)
    rf = time_cuda(ref_fwd)
    cb = time_cuda(custom_bwd)
    rb = time_cuda(ref_bwd)

    print_result(name, cf, rf, cb, rb, gbps(elems * 4, cf), gbps(elems * 6, cb))

    profile_table("Softmax custom forward", custom_fwd)
    profile_table("Softmax custom backward", custom_bwd)


def profile_gelu():
    name = "GELU exact"
    shape = (2, 4, 1024, 128)
    B, H, N, D = shape
    print_kernel_header(name, shape)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(gelu_cuda.forward(x))

    x_ref = x.detach().float().requires_grad_(True)
    y_ref_fp32 = F.gelu(x_ref, approximate="none")
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, name, shape, 5e-2, 5e-2)

    dx = first(gelu_cuda.backward(dy, x))
    y_ref_fp32.backward(dy.float())

    dx_err = check_close("dx", dx, x_ref.grad, name, shape, 6e-2, 6e-2)

    print(f"Correctness: PASS | fwd={fwd_err:.4g}, dx={dx_err:.4g}")

    x_ref_b = x.detach().float().requires_grad_(True)
    y_ref_b = F.gelu(x_ref_b, approximate="none")

    custom_fwd = lambda: first(gelu_cuda.forward(x))
    custom_bwd = lambda: gelu_cuda.backward(dy, x)
    ref_fwd = lambda: F.gelu(x.float(), approximate="none").to(torch.float16)
    ref_bwd = lambda: torch.autograd.grad(y_ref_b, x_ref_b, grad_outputs=dy.float(), retain_graph=True)

    elems = B * H * N * D
    cf = time_cuda(custom_fwd)
    rf = time_cuda(ref_fwd)
    cb = time_cuda(custom_bwd)
    rb = time_cuda(ref_bwd)

    print_result(name, cf, rf, cb, rb, gbps(elems * 4, cf), gbps(elems * 6, cb))

    profile_table("GELU exact custom forward", custom_fwd)
    profile_table("GELU exact custom backward", custom_bwd)


def profile_silu():
    name = "SiLU"
    shape = (2, 4, 1024, 128)
    B, H, N, D = shape
    print_kernel_header(name, shape)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(silu_cuda.forward(x))

    x_ref = x.detach().float().requires_grad_(True)
    y_ref_fp32 = F.silu(x_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, name, shape, 5e-2, 5e-2)

    dx = first(silu_cuda.backward(dy, x))
    y_ref_fp32.backward(dy.float())

    dx_err = check_close("dx", dx, x_ref.grad, name, shape, 6e-2, 6e-2)

    print(f"Correctness: PASS | fwd={fwd_err:.4g}, dx={dx_err:.4g}")

    x_ref_b = x.detach().float().requires_grad_(True)
    y_ref_b = F.silu(x_ref_b)

    custom_fwd = lambda: first(silu_cuda.forward(x))
    custom_bwd = lambda: silu_cuda.backward(dy, x)
    ref_fwd = lambda: F.silu(x.float()).to(torch.float16)
    ref_bwd = lambda: torch.autograd.grad(y_ref_b, x_ref_b, grad_outputs=dy.float(), retain_graph=True)

    elems = B * H * N * D
    cf = time_cuda(custom_fwd)
    rf = time_cuda(ref_fwd)
    cb = time_cuda(custom_bwd)
    rb = time_cuda(ref_bwd)

    print_result(name, cf, rf, cb, rb, gbps(elems * 4, cf), gbps(elems * 6, cb))

    profile_table("SiLU custom forward", custom_fwd)
    profile_table("SiLU custom backward", custom_bwd)


def profile_fused_bias_gelu():
    name = "FusedBiasGELU exact"
    shape = (2, 4, 1024, 128)
    B, H, N, D = shape
    print_kernel_header(name, shape)

    x = rand_half(shape, scale=2.0)
    bias = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    y = first(fused_bias_gelu_cuda.forward(x, bias))

    x_ref = x.detach().float().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    y_ref_fp32 = F.gelu(x_ref + bias_ref, approximate="none")
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, name, shape, 5e-2, 5e-2)

    out = fused_bias_gelu_cuda.backward(dy, x, bias)
    dx, dbias = out[0], out[1]
    y_ref_fp32.backward(dy.float())

    dx_err = check_close("dx", dx, x_ref.grad, name, shape, 6e-2, 6e-2)
    db_err = check_close("dbias", dbias, bias_ref.grad, name, shape, 5e-1, 8e-2)

    print(f"Correctness: PASS | fwd={fwd_err:.4g}, dx={dx_err:.4g}, dbias={db_err:.4g}")

    x_ref_b = x.detach().float().requires_grad_(True)
    bias_ref_b = bias.detach().clone().requires_grad_(True)
    y_ref_b = F.gelu(x_ref_b + bias_ref_b, approximate="none")

    custom_fwd = lambda: first(fused_bias_gelu_cuda.forward(x, bias))
    custom_bwd = lambda: fused_bias_gelu_cuda.backward(dy, x, bias)
    ref_fwd = lambda: F.gelu(x.float() + bias, approximate="none").to(torch.float16)
    ref_bwd = lambda: torch.autograd.grad(y_ref_b, (x_ref_b, bias_ref_b), grad_outputs=dy.float(), retain_graph=True)

    elems = B * H * N * D
    cf = time_cuda(custom_fwd)
    rf = time_cuda(ref_fwd)
    cb = time_cuda(custom_bwd)
    rb = time_cuda(ref_bwd)

    print_result(name, cf, rf, cb, rb, gbps(elems * 6 + D * 4, cf), gbps(elems * 8 + D * 4, cb))

    profile_table("FusedBiasGELU custom forward", custom_fwd)
    profile_table("FusedBiasGELU custom backward", custom_bwd)


def profile_fused_residual_rmsnorm():
    name = "FusedResidualRMSNorm"
    shape = (2, 4, 1024, 128)
    B, H, N, D = shape
    print_kernel_header(name, shape)

    x = rand_half(shape)
    residual = rand_half(shape)
    gamma = rand_half((D,))
    dy = rand_half(shape)

    y = first(fused_residual_rmsnorm_cuda.forward(x, residual, gamma, EPS))

    x_ref = x.detach().float().requires_grad_(True)
    residual_ref = residual.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)
    y_ref_fp32 = ref_fused_residual_rmsnorm(x_ref, residual_ref, gamma_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, name, shape, 5e-2, 5e-2)

    out = fused_residual_rmsnorm_cuda.backward(dy, x, residual, gamma, EPS)
    dx, dresidual, dgamma = out[0], out[1], out[2]
    y_ref_fp32.backward(dy.float())

    dx_err = check_close("dx", dx, x_ref.grad, name, shape, 8e-2, 8e-2)
    dr_err = check_close("dresidual", dresidual, residual_ref.grad, name, shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, name, shape, 5e-1, 8e-2)

    dx_dr_err = (dx.float() - dresidual.float()).abs().max().item()
    if dx_dr_err != 0.0:
        raise AssertionError(f"dx != dresidual. max_err={dx_dr_err}")

    print(f"Correctness: PASS | fwd={fwd_err:.4g}, dx={dx_err:.4g}, dres={dr_err:.4g}, dgamma={dg_err:.4g}")

    x_ref_b = x.detach().float().requires_grad_(True)
    residual_ref_b = residual.detach().float().requires_grad_(True)
    gamma_ref_b = gamma.detach().float().requires_grad_(True)
    y_ref_b = ref_fused_residual_rmsnorm(x_ref_b, residual_ref_b, gamma_ref_b)

    custom_fwd = lambda: first(fused_residual_rmsnorm_cuda.forward(x, residual, gamma, EPS))
    custom_bwd = lambda: fused_residual_rmsnorm_cuda.backward(dy, x, residual, gamma, EPS)
    ref_fwd = lambda: ref_fused_residual_rmsnorm(x.float(), residual.float(), gamma.float()).to(torch.float16)
    ref_bwd = lambda: torch.autograd.grad(y_ref_b, (x_ref_b, residual_ref_b, gamma_ref_b), grad_outputs=dy.float(), retain_graph=True)

    elems = B * H * N * D
    cf = time_cuda(custom_fwd)
    rf = time_cuda(ref_fwd)
    cb = time_cuda(custom_bwd)
    rb = time_cuda(ref_bwd)

    print_result(name, cf, rf, cb, rb, gbps(elems * 6 + D * 2, cf), gbps(elems * 10 + D * 4, cb))

    profile_table("FusedResidualRMSNorm custom forward", custom_fwd)
    profile_table("FusedResidualRMSNorm custom backward", custom_bwd)


def profile_rope():
    name = "RoPE"
    shape = (2, 4, 1024, 128, 128, "position_ids")
    B, H, N, D, rotary_dim, mode = shape
    print_kernel_header(name, shape)

    x = rand_half((B, H, N, D))
    dy = rand_half((B, H, N, D))

    max_seq_len = N + 16
    cos, sin = rope_cuda.build_cache(x, max_seq_len, rotary_dim, 10000.0)
    cos = cos.float().contiguous()
    sin = sin.float().contiguous()

    base = torch.arange(N, device="cuda", dtype=torch.int32).unsqueeze(0).repeat(B, 1)
    batch_offsets = torch.arange(B, device="cuda", dtype=torch.int32).unsqueeze(1)
    position_ids = (base + batch_offsets).contiguous()
    position_offset = 0

    cos_ref, sin_ref = manual_rope_cache(max_seq_len, rotary_dim, x.device)
    cos_err = check_close("cos_cache", cos, cos_ref, "RoPE build_cache", shape, 1e-4, 1e-4)
    sin_err = check_close("sin_cache", sin, sin_ref, "RoPE build_cache", shape, 1e-4, 1e-4)

    y = first(rope_cuda.forward(x, position_ids, cos, sin, rotary_dim, position_offset))

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

    fwd_err = check_close("forward", y, y_ref, name, shape, 5e-2, 5e-2)

    dx = first(rope_cuda.backward(dy, position_ids, cos, sin, rotary_dim, position_offset))
    y_ref_fp32.backward(dy.float())

    dx_err = check_close("dx", dx, x_ref.grad, name, shape, 6e-2, 6e-2)

    print(f"Correctness: PASS | cache_cos={cos_err:.4g}, cache_sin={sin_err:.4g}, fwd={fwd_err:.4g}, dx={dx_err:.4g}")

    x_ref_b = x.detach().float().requires_grad_(True)
    y_ref_b = ref_rope_forward(
        x_ref_b,
        cos.detach(),
        sin.detach(),
        rotary_dim,
        position_ids=position_ids,
        position_offset=position_offset,
    )

    custom_cache = lambda: rope_cuda.build_cache(x, max_seq_len, rotary_dim, 10000.0)
    ref_cache = lambda: manual_rope_cache(max_seq_len, rotary_dim, x.device)
    custom_fwd = lambda: first(rope_cuda.forward(x, position_ids, cos, sin, rotary_dim, position_offset))
    custom_bwd = lambda: first(rope_cuda.backward(dy, position_ids, cos, sin, rotary_dim, position_offset))
    ref_fwd = lambda: ref_rope_forward(
        x.float(),
        cos,
        sin,
        rotary_dim,
        position_ids=position_ids,
        position_offset=position_offset,
    ).to(torch.float16)
    ref_bwd = lambda: torch.autograd.grad(y_ref_b, x_ref_b, grad_outputs=dy.float(), retain_graph=True)

    elems = B * H * N * D

    cc = time_cuda(custom_cache, iters=max(30, BENCH_ITERS // 3))
    rc = time_cuda(ref_cache, iters=max(30, BENCH_ITERS // 3))
    cf = time_cuda(custom_fwd)
    rf = time_cuda(ref_fwd)
    cb = time_cuda(custom_bwd)
    rb = time_cuda(ref_bwd)

    print("\nSpeed:")
    print(f"  cache custom   : {cc:.5f} ms")
    print(f"  cache ref      : {rc:.5f} ms")
    print(f"  cache speedup  : {rc / cc:.2f}x")
    print(f"  forward custom : {cf:.5f} ms")
    print(f"  forward ref    : {rf:.5f} ms")
    print(f"  forward speedup: {rf / cf:.2f}x")
    print(f"  forward approx : {gbps(elems * 4 + N * rotary_dim * 4, cf):.1f} GB/s")
    print(f"  backward custom : {cb:.5f} ms")
    print(f"  backward ref    : {rb:.5f} ms")
    print(f"  backward speedup: {rb / cb:.2f}x")
    print(f"  backward approx : {gbps(elems * 4 + N * rotary_dim * 4, cb):.1f} GB/s")

    profile_table("RoPE custom cache build", custom_cache)
    profile_table("RoPE custom forward", custom_fwd)
    profile_table("RoPE custom backward", custom_bwd)


def run_one(name, fn):
    try:
        fn()
    except Exception as e:
        print("\nFAILED:", name)
        print(type(e).__name__, e)
        FAILURES.append((name, str(e)))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    torch.manual_seed(0)

    print("\nCUDA REAL PROFILING SCRIPT")
    print("=" * 90)
    print("Device:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("BENCH_WARMUP:", BENCH_WARMUP)
    print("BENCH_ITERS:", BENCH_ITERS)
    print("PROFILE_TABLES:", RUN_PROFILER_TABLES)
    print("PROFILE_ITERS:", PROFILE_ITERS)
    print("=" * 90)

    run_one("RMSNorm", profile_rmsnorm)
    run_one("LayerNorm", profile_layernorm)
    run_one("Softmax", profile_softmax)
    run_one("GELU exact", profile_gelu)
    run_one("SiLU", profile_silu)
    run_one("FusedBiasGELU exact", profile_fused_bias_gelu)
    run_one("FusedResidualRMSNorm", profile_fused_residual_rmsnorm)
    run_one("RoPE", profile_rope)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    if not FAILURES:
        print("ALL PROFILE CHECKS PASSED")
        print("Paste the terminal output here and I will tell you which kernels are memory-bound, math-bound, reduction-bound, or atomic-bound.")
        return

    print(f"{len(FAILURES)} failures:")
    for i, (name, reason) in enumerate(FAILURES, 1):
        print(f"{i}. {name}: {reason}")

    raise SystemExit(1)


if __name__ == "__main__":
    main()

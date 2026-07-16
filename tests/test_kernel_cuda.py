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


def run_case(kernel, shape, fn):
    try:
        msg = fn()
        print(f"  PASS {shape} | {msg}")
    except Exception as e:
        print(f"  FAIL {shape} | {e}")
        FAILURES.append((kernel, shape, str(e)))



def ref_rmsnorm(x, gamma):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + EPS)
    return x / rms * gamma


def test_rmsnorm_shape(B, H, N, D):
    kernel = "RMSNorm"
    shape = (B, H, N, D)

    x = rand_half(shape)
    gamma = rand_half((D,))
    dy = rand_half(shape)

    y = first(rmsnorm_cuda.forward(x, gamma, EPS))
    torch.cuda.synchronize()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)

    y_ref_fp32 = ref_rmsnorm(x_ref, gamma_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = rmsnorm_cuda.backward(dy, x, gamma, EPS)
    dx = out[0]
    dgamma = out[1]
    torch.cuda.synchronize()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)
    check_finite("dgamma", dgamma, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, kernel, shape, 5e-1, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dgamma={dg_err:.4g}"



def test_layernorm_shape(B, H, N, D):
    kernel = "LayerNorm"
    shape = (B, H, N, D)

    x = rand_half(shape)
    gamma = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    beta = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    y = first(layernorm_cuda.forward(x, gamma, beta, EPS))
    torch.cuda.synchronize()

    check_finite("forward", y, kernel, shape)

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

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = layernorm_cuda.backward(dy, x, gamma, beta, EPS)
    dx = out[0]
    dgamma = out[1]
    dbeta = out[2]
    torch.cuda.synchronize()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)
    check_finite("dgamma", dgamma, kernel, shape)
    check_finite("dbeta", dbeta, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 8e-2, 8e-2)
    dg_err = check_close("dgamma", dgamma, gamma_ref.grad, kernel, shape, 5e-1, 8e-2)
    db_err = check_close("dbeta", dbeta, beta_ref.grad, kernel, shape, 5e-1, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dgamma={dg_err:.4g}, dbeta={db_err:.4g}"


def test_softmax_shape(B, H, N, D):
    kernel = "Softmax"
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(softmax_cuda.forward(x))
    torch.cuda.synchronize()

    check_finite("forward", y, kernel, shape)

    row_sum_err = (y.float().sum(dim=-1) - 1.0).abs().max().item()
    if row_sum_err > 5e-2:
        raise AssertionError(
            f"{kernel} {shape} row sum failed. max_row_sum_err={row_sum_err}"
        )

    x_ref = x.detach().float().requires_grad_(True)

    y_ref_fp32 = F.softmax(x_ref, dim=-1)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = softmax_cuda.backward(dy, x)
    dx = out[0]
    torch.cuda.synchronize()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 8e-2, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, row_sum={row_sum_err:.4g}"


def test_gelu_shape(B, H, N, D):
    kernel = "GELU exact"
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(gelu_cuda.forward(x))
    torch.cuda.synchronize()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)

    y_ref_fp32 = F.gelu(x_ref, approximate="none")
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = gelu_cuda.backward(dy, x)
    dx = out[0]
    torch.cuda.synchronize()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 6e-2, 6e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}"


def test_silu_shape(B, H, N, D):
    kernel = "SiLU"
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    dy = rand_half(shape)

    y = first(silu_cuda.forward(x))
    torch.cuda.synchronize()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)

    y_ref_fp32 = F.silu(x_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = silu_cuda.backward(dy, x)
    dx = out[0]
    torch.cuda.synchronize()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 6e-2, 6e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}"


def test_fused_bias_gelu_shape(B, H, N, D):
    kernel = "FusedBiasGELU exact"
    shape = (B, H, N, D)

    x = rand_half(shape, scale=2.0)
    bias = torch.randn(D, device="cuda", dtype=torch.float32).contiguous()
    dy = rand_half(shape)

    y = first(fused_bias_gelu_cuda.forward(x, bias))
    torch.cuda.synchronize()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)

    y_ref_fp32 = F.gelu(x_ref + bias_ref, approximate="none")
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = fused_bias_gelu_cuda.backward(dy, x, bias)
    dx = out[0]
    dbias = out[1]
    torch.cuda.synchronize()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)
    check_finite("dbias", dbias, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 6e-2, 6e-2)
    db_err = check_close("dbias", dbias, bias_ref.grad, kernel, shape, 5e-1, 8e-2)

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dbias={db_err:.4g}"


def ref_fused_residual_rmsnorm(x, residual, gamma):
    z = x + residual
    rms = torch.sqrt(torch.mean(z * z, dim=-1, keepdim=True) + EPS)
    return z / rms * gamma


def test_fused_residual_rmsnorm_shape(B, H, N, D):
    kernel = "FusedResidualRMSNorm"
    shape = (B, H, N, D)

    x = rand_half(shape)
    residual = rand_half(shape)
    gamma = rand_half((D,))
    dy = rand_half(shape)

    y = first(fused_residual_rmsnorm_cuda.forward(x, residual, gamma, EPS))
    torch.cuda.synchronize()

    check_finite("forward", y, kernel, shape)

    x_ref = x.detach().float().requires_grad_(True)
    residual_ref = residual.detach().float().requires_grad_(True)
    gamma_ref = gamma.detach().float().requires_grad_(True)

    y_ref_fp32 = ref_fused_residual_rmsnorm(x_ref, residual_ref, gamma_ref)
    y_ref = y_ref_fp32.to(torch.float16)

    fwd_err = check_close("forward", y, y_ref, kernel, shape, 5e-2, 5e-2)

    y_before = y.detach().clone()

    out = fused_residual_rmsnorm_cuda.backward(dy, x, residual, gamma, EPS)
    dx = out[0]
    dresidual = out[1]
    dgamma = out[2]
    torch.cuda.synchronize()

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
        raise AssertionError(
            f"{kernel} {shape} failed dx == dresidual. max_err={dx_dr_err}"
        )

    return f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, dres={dr_err:.4g}, dgamma={dg_err:.4g}"


def manual_rope_cache(max_seq_len, rotary_dim, device):
    half = rotary_dim // 2

    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    dim = torch.arange(half, device=device, dtype=torch.float32)

    inv_freq = 1.0 / (10000.0 ** (2.0 * dim / rotary_dim))
    freqs = torch.outer(pos, inv_freq)

    cos = torch.cos(freqs).float().contiguous()
    sin = torch.sin(freqs).float().contiguous()

    return cos, sin


def extension_rope_cache(reference, max_seq_len, rotary_dim):
    out = rope_cuda.build_cache(reference, max_seq_len, rotary_dim, 10000.0)

    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise AssertionError(
            "RoPE build_cache should return two tensors: cos_cache, sin_cache"
        )

    cos = out[0].float().contiguous()
    sin = out[1].float().contiguous()

    return cos, sin


def ref_rope_forward(x, cos_cache, sin_cache, rotary_dim, position_ids=None, position_offset=0):
    # x: [B,H,N,D]
    B, H, N, D = x.shape
    half = rotary_dim // 2

    x = x.float()

    x_rot = x[..., :rotary_dim]
    x_tail = x[..., rotary_dim:]

    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:rotary_dim]

    if position_ids is None:
        pos = torch.arange(N, device=x.device, dtype=torch.long) + int(position_offset)
        cos = cos_cache[pos]  # [N, half]
        sin = sin_cache[pos]  # [N, half]

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        pos = position_ids.long()
        cos = cos_cache[pos]  # [B, N, half]
        sin = sin_cache[pos]  # [B, N, half]

        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

    cos = cos.float()
    sin = sin.float()

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    y_rot = torch.cat([y1, y2], dim=-1)

    if rotary_dim < D:
        return torch.cat([y_rot, x_tail], dim=-1)

    return y_rot


def rope_forward_cuda_call(x, position_ids, cos, sin, rotary_dim, position_offset):
    out = rope_cuda.forward(
        x,
        position_ids,
        cos.float().contiguous(),
        sin.float().contiguous(),
        int(rotary_dim),
        int(position_offset),
    )
    return first(out)


def rope_backward_cuda_call(dy, position_ids, cos, sin, rotary_dim, position_offset):
    out = rope_cuda.backward(
        dy,
        position_ids,
        cos.float().contiguous(),
        sin.float().contiguous(),
        int(rotary_dim),
        int(position_offset),
    )
    return first(out)


def check_rope_cache(reference, max_seq_len, rotary_dim):
    kernel = "RoPE build_cache"
    shape = ("cache", max_seq_len, rotary_dim)

    cos_ext, sin_ext = extension_rope_cache(reference, max_seq_len, rotary_dim)
    cos_ref, sin_ref = manual_rope_cache(max_seq_len, rotary_dim, reference.device)

    check_finite("cos_cache", cos_ext, kernel, shape)
    check_finite("sin_cache", sin_ext, kernel, shape)

    cos_err = check_close("cos_cache", cos_ext, cos_ref, kernel, shape, 5e-5, 5e-5)
    sin_err = check_close("sin_cache", sin_ext, sin_ref, kernel, shape, 5e-5, 5e-5)

    return cos_ext, sin_ext, cos_err, sin_err


def test_rope_shape(B, H, N, D, rotary_dim, mode):
    kernel = "RoPE"
    shape = (B, H, N, D, rotary_dim, mode)

    if rotary_dim > D:
        raise AssertionError(f"{kernel} invalid test: rotary_dim={rotary_dim} > D={D}")

    if rotary_dim % 2 != 0:
        raise AssertionError(f"{kernel} invalid test: rotary_dim must be even, got {rotary_dim}")

    x = rand_half((B, H, N, D))
    dy = rand_half((B, H, N, D))

    # cache longer than N so position_offset and position_ids can be tested
    max_seq_len = max(N + 16, 32)

    cos, sin, cos_cache_err, sin_cache_err = check_rope_cache(x, max_seq_len, rotary_dim)

    if mode == "none":
        position_ids = None
        position_offset = 0

    elif mode == "offset":
        position_ids = None
        position_offset = 3
        if position_offset + N > max_seq_len:
            raise AssertionError("bad RoPE offset test: cache too short")

    elif mode == "position_ids":
        # Different batch rows can use different positions.
        base = torch.arange(N, device="cuda", dtype=torch.int32).unsqueeze(0).repeat(B, 1)
        batch_offsets = torch.arange(B, device="cuda", dtype=torch.int32).unsqueeze(1)
        position_ids = (base + batch_offsets).contiguous()
        position_offset = 0

        if int(position_ids.max().item()) >= max_seq_len:
            raise AssertionError("bad RoPE position_ids test: cache too short")

    else:
        raise AssertionError(f"unknown RoPE mode={mode}")

    y = rope_forward_cuda_call(x, position_ids, cos, sin, rotary_dim, position_offset)
    torch.cuda.synchronize()

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

    dx = rope_backward_cuda_call(dy, position_ids, cos, sin, rotary_dim, position_offset)
    torch.cuda.synchronize()

    check_not_mutated("forward output during backward", y_before, y, kernel, shape)

    y_ref_fp32.backward(dy.float())

    check_finite("dx", dx, kernel, shape)

    dx_err = check_close("dx", dx, x_ref.grad, kernel, shape, 6e-2, 6e-2)

    if rotary_dim < D:
        tail_fwd_err = (y[..., rotary_dim:].float() - x[..., rotary_dim:].float()).abs().max().item()
        if tail_fwd_err != 0.0:
            raise AssertionError(
                f"{kernel} {shape} failed forward tail passthrough. tail_err={tail_fwd_err}"
            )

        tail_bwd_err = (dx[..., rotary_dim:].float() - dy[..., rotary_dim:].float()).abs().max().item()
        if tail_bwd_err != 0.0:
            raise AssertionError(
                f"{kernel} {shape} failed backward tail passthrough. tail_dx_err={tail_bwd_err}"
            )
    else:
        tail_fwd_err = 0.0
        tail_bwd_err = 0.0

    return (
        f"cache_cos={cos_cache_err:.2g}, cache_sin={sin_cache_err:.2g}, "
        f"fwd={fwd_err:.4g}, dx={dx_err:.4g}, "
        f"tail_fwd={tail_fwd_err:.4g}, tail_bwd={tail_bwd_err:.4g}"
    )


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
        run_case("GELU exact", s, lambda s=s: test_gelu_shape(*s))

    print("\n========== SiLU ==========")
    for s in shapes_any:
        run_case("SiLU", s, lambda s=s: test_silu_shape(*s))

    print("\n========== Fused bias + GELU exact ==========")
    for s in shapes_any:
        run_case("FusedBiasGELU exact", s, lambda s=s: test_fused_bias_gelu_shape(*s))

    print("\n========== Fused residual + RMSNorm ==========")
    for s in shapes_d8:
        run_case("FusedResidualRMSNorm", s, lambda s=s: test_fused_residual_rmsnorm_shape(*s))

    print("\n========== RoPE ==========")
    rope_cases = [
        (1, 1, 1, 32, 32, "none"),
        (1, 1, 17, 32, 32, "none"),
        (1, 4, 31, 64, 64, "none"),
        (2, 4, 63, 64, 32, "none"),
        (1, 2, 128, 128, 64, "none"),
        (2, 1, 256, 128, 128, "none"),

        # position_offset path
        (1, 1, 17, 32, 32, "offset"),
        (2, 4, 63, 64, 32, "offset"),

        # explicit position_ids path
        (1, 1, 17, 32, 32, "position_ids"),
        (2, 4, 63, 64, 32, "position_ids"),
    ]

    for s in rope_cases:
        run_case("RoPE", s, lambda s=s: test_rope_shape(*s))

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
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def bias_gelu_fwd_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    M,
    H,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    x_ptrs = x_ptr + row * H + cols
    out_ptrs = out_ptr + row * H + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    z = x + bias

    inv_sqrt2 = 0.7071067811865475

    y = 0.5 * z * (
        1.0 + tl.math.erf(z * inv_sqrt2)
    )

    tl.store(
        out_ptrs,
        y.to(x.dtype),
        mask=mask,
    )


@triton.jit
def bias_gelu_bwd_kernel(
    x_ptr,
    bias_ptr,
    grad_out_ptr,
    grad_x_ptr,
    grad_bias_ptr,
    M,
    H,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    x_ptrs = x_ptr + row * H + cols
    go_ptrs = grad_out_ptr + row * H + cols
    gx_ptrs = grad_x_ptr + row * H + cols

    x = tl.load(
        x_ptrs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    grad_out = tl.load(
        go_ptrs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    bias = tl.load(
        bias_ptr + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # Fused bias + GELU input.
    z = x + bias

    inv_sqrt2 = 0.7071067811865475
    inv_sqrt_2pi = 0.3989422804014327

    erf_term = tl.math.erf(
        z * inv_sqrt2
    )

    exp_term = tl.math.exp(
        -0.5 * z * z
    )

    gelu_grad = (
        0.5 * (1.0 + erf_term)
        + z * exp_term * inv_sqrt_2pi
    )

    grad = grad_out * gelu_grad

    tl.store(
        gx_ptrs,
        grad.to(x.dtype),
        mask=mask,
    )

    tl.atomic_add(
        grad_bias_ptr + cols,
        grad,
        mask=mask,
    )


class FusedBiasGeluFunction(Function):

    @staticmethod
    def forward(ctx, x, bias):
        assert x.is_cuda
        assert bias.is_cuda

        assert bias.ndim == 1
        assert x.shape[-1] == bias.shape[0]

        x = x.contiguous()
        bias = bias.contiguous()

        # Flatten all leading dimensions.
        M = x.numel() // x.shape[-1]
        H = x.shape[-1]

        out = torch.empty_like(x)

        grid = (M,)

        BLOCK_H = triton.next_power_of_2(H)

        # Avoid absurdly large blocks.
        if BLOCK_H > 4096:
            raise ValueError(
                f"Hidden size {H} is too large for this kernel."
            )

        bias_gelu_fwd_kernel[grid](
            x,
            bias,
            out,
            M,
            H,
            BLOCK_H=BLOCK_H,
        )

        ctx.save_for_backward(x, bias)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, bias = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        M = x.numel() // x.shape[-1]
        H = x.shape[-1]

        grad_x = torch.empty_like(x)

        grad_bias_fp32 = torch.zeros(
            H,
            device=bias.device,
            dtype=torch.float32,
        )

        BLOCK_H = triton.next_power_of_2(H)

        grid = (M,)

        bias_gelu_bwd_kernel[grid](
            x,
            bias,
            grad_output,
            grad_x,
            grad_bias_fp32,
            M,
            H,
            BLOCK_H=BLOCK_H,
        )

        grad_bias = grad_bias_fp32.to(bias.dtype)

        return grad_x, grad_bias


class FusedBiasGelu(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.bias = nn.Parameter(
            torch.zeros(
                hidden_size,
                dtype=torch.float32,
            )
        )

    def forward(self, x):
        return FusedBiasGeluFunction.apply(
            x,
            self.bias,
        )

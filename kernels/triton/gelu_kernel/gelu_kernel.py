import triton
import triton.language as tl
import torch
from torch.autograd import Function
import torch.nn as nn


@triton.jit
def gelu_fwd_kernel(
    aptr,
    bptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    a = tl.load(aptr + offset, mask=mask, other=0.0)
    x = a.to(tl.float32)

    inv_sqrt2 = 0.7071067811865475
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))

    tl.store(bptr + offset, y.to(a.dtype), mask=mask)


@triton.jit
def gelu_bwd_kernel(
    xptr,   # saved input
    goptr,  # grad_output
    giptr,  # grad_input (written)
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    x_orig = tl.load(xptr + offset, mask=mask, other=0.0)
    go_orig = tl.load(goptr + offset, mask=mask, other=0.0)

    x  = x_orig.to(tl.float32)
    go = go_orig.to(tl.float32)

    inv_sqrt2   = 0.7071067811865475        # 1 / sqrt(2)
    inv_sqrt_2pi = 0.3989422804014327       # 1 / sqrt(2*pi)

    erf_term = tl.math.erf(x * inv_sqrt2)
    exp_term = tl.math.exp(-0.5 * x * x)

    dydx = 0.5 * (1.0 + erf_term) + x * exp_term * inv_sqrt_2pi
    gi = go * dydx

    tl.store(giptr + offset, gi.to(x_orig.dtype), mask=mask)


class GeluFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        N = x.numel()
        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)
        gelu_fwd_kernel[grid](x, out, N, BLOCK=BLOCK)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(x)
        N = x.numel()
        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)
        gelu_bwd_kernel[grid](x, grad_output, grad_input, N, BLOCK=BLOCK)
        return grad_input


class Gelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GeluFunction.apply(x)

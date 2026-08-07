import triton
import triton.language as tl
import torch
from torch.autograd import Function
import torch.nn as nn


@triton.jit
def silu_fwd_kernel(
    xptr,
    yptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x_orig = tl.load(xptr + offsets, mask=mask, other=0.0)
    x = x_orig.to(tl.float32)

    sigmoid = 1.0 / (1.0 + tl.math.exp(-x))
    y = x * sigmoid

    tl.store(yptr + offsets, y.to(x_orig.dtype), mask=mask)


@triton.jit
def silu_bwd_kernel(
    xptr,      # saved input
    goptr,     # grad_output
    giptr,     # grad_input
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x_orig = tl.load(xptr + offsets, mask=mask, other=0.0)
    go_orig = tl.load(goptr + offsets, mask=mask, other=0.0)

    x = x_orig.to(tl.float32)
    go = go_orig.to(tl.float32)

    sigmoid = 1.0 / (1.0 + tl.math.exp(-x))

    # d/dx [x * sigmoid(x)]
    dydx = sigmoid * (1.0 + x * (1.0 - sigmoid))

    gi = go * dydx

    tl.store(giptr + offsets, gi.to(x_orig.dtype), mask=mask)


class SiLUFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        x = x.contiguous()
        out = torch.empty_like(x)

        N = x.numel()
        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)

        silu_fwd_kernel[grid](
            x,
            out,
            N,
            BLOCK=BLOCK,
        )

        ctx.save_for_backward(x)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(x)

        N = x.numel()
        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)

        silu_bwd_kernel[grid](
            x,
            grad_output,
            grad_input,
            N,
            BLOCK=BLOCK,
        )

        return grad_input


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return SiLUFunction.apply(x)

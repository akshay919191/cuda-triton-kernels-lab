import triton
import triton.language as tl
import torch
from torch.autograd import Function
import torch.nn as nn


@triton.jit
def relu_fwd_kernel(
    xptr,
    yptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(xptr + offsets, mask=mask, other=0.0)
    y = tl.where(x > 0, x, 0.0)

    tl.store(yptr + offsets, y, mask=mask)


@triton.jit
def relu_bwd_kernel(
    xptr,      # saved input
    goptr,     # grad_output
    giptr,     # grad_input
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(xptr + offsets, mask=mask, other=0.0)
    go = tl.load(goptr + offsets, mask=mask, other=0.0)

    gi = tl.where(x > 0, go, 0.0)

    tl.store(giptr + offsets, gi, mask=mask)


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        x = x.contiguous()
        out = torch.empty_like(x)

        N = x.numel()
        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)

        relu_fwd_kernel[grid](
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

        relu_bwd_kernel[grid](
            x,
            grad_output,
            grad_input,
            N,
            BLOCK=BLOCK,
        )

        return grad_input


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return ReLUFunction.apply(x)
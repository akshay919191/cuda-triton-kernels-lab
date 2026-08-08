import torch
import triton
import triton.language as tl
import torch.nn as nn
from torch.autograd import Function



@triton.jit
def softmaxFWD(
    x_ptr,
    y_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    x_orig = tl.load(
        x_ptr + row * D + offsets,
        mask=mask,
        other=-float("inf"),
    )

    x = x_orig.to(tl.float32)

    x_max = tl.max(x, axis=0)

    x = x - x_max

    exp_x = tl.exp(x)

    denominator = tl.sum(
        exp_x,
        axis=0,
    )

    y = exp_x / denominator

    tl.store(
        y_ptr + row * D + offsets,
        y.to(x_orig.dtype),
        mask=mask,
    )


@triton.jit
def softmaxBWD(
    y_ptr,
    dy_ptr,
    dx_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    y_orig = tl.load(
        y_ptr + row * D + offsets,
        mask=mask,
        other=0.0,
    )

    dy_orig = tl.load(
        dy_ptr + row * D + offsets,
        mask=mask,
        other=0.0,
    )

    y = y_orig.to(tl.float32)
    dy = dy_orig.to(tl.float32)

    dot = tl.sum(
        dy * y,
        axis=0,
    )

    dx = y * (dy - dot)

    tl.store(
        dx_ptr + row * D + offsets,
        dx.to(y_orig.dtype),
        mask=mask,
    )


class SoftmaxFunction(Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
    ):

        x = x.contiguous()

        D = x.shape[-1]
        N = x.numel() // D

        # Flatten logically to [N, D]
        x_2d = x.reshape(N, D)

        out = torch.empty_like(x_2d)

        BLOCK_SIZE = triton.next_power_of_2(D)

        grid = (N,)

        softmaxFWD[grid](
            x_2d,
            out,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        #
        ctx.save_for_backward(out)

        ctx.D = D
        ctx.N = N
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.input_shape = x.shape

        return out.reshape(x.shape)

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        (y,) = ctx.saved_tensors

        D = ctx.D
        N = ctx.N
        BLOCK_SIZE = ctx.BLOCK_SIZE

        grad_output = grad_output.contiguous()

        dy = grad_output.reshape(N, D)

        dx = torch.empty_like(y)

        grid = (N,)

        softmaxBWD[grid](
            y,
            dy,
            dx,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return dx.reshape(ctx.input_shape)

class Softmax(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ):
        return SoftmaxFunction.apply(x)



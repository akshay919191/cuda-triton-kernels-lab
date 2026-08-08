import torch
import triton
import triton.language as tl
import torch.nn as nn
from torch.autograd import Function


@triton.jit
def rmsnormFWD(
    x_ptr,
    y_ptr,
    w_ptr,
    rms_ptr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    x_orig = tl.load(
        x_ptr + row * D + offsets,
        mask=mask,
        other=0.0,
    )

    x = x_orig.to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)

    rms = tl.sqrt(sum_sq / D + eps)

    w_orig = tl.load(
        w_ptr + offsets,
        mask=mask,
        other=0.0,
    )

    w = w_orig.to(tl.float32)

    y = (x / rms) * w

    tl.store(
        y_ptr + row * D + offsets,
        y.to(x_orig.dtype),
        mask=mask,
    )

    tl.store(
        rms_ptr + row,
        rms,
    )


@triton.jit
def rmsnorm_bwd_dx(
    x_ptr,
    dy_ptr,
    gamma_ptr,
    rms_ptr,
    dx_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    x_orig = tl.load(
        x_ptr + pid * D + offsets,
        mask=mask,
        other=0.0,
    )

    dy_orig = tl.load(
        dy_ptr + pid * D + offsets,
        mask=mask,
        other=0.0,
    )

    gamma_orig = tl.load(
        gamma_ptr + offsets,
        mask=mask,
        other=0.0,
    )

    x = x_orig.to(tl.float32)
    dy = dy_orig.to(tl.float32)
    gamma = gamma_orig.to(tl.float32)

    rms = tl.load(rms_ptr + pid)

    g = dy * gamma

    c = tl.sum(g * x, axis=0)

    dx = (
        g / rms
        - x * c / (D * rms * rms * rms)
    )

    tl.store(
        dx_ptr + pid * D + offsets,
        dx.to(x_orig.dtype),
        mask=mask,
    )


@triton.jit
def rmsnorm_bwd_dgamma(
    x_ptr,
    dy_ptr,
    rms_ptr,
    dgamma_ptr,
    D: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    acc = tl.zeros(
        (BLOCK_SIZE,),
        dtype=tl.float32,
    )

    for row in range(0, N):
        x_orig = tl.load(
            x_ptr + row * D + offsets,
            mask=mask,
            other=0.0,
        )

        dy_orig = tl.load(
            dy_ptr + row * D + offsets,
            mask=mask,
            other=0.0,
        )

        rms = tl.load(rms_ptr + row)

        x = x_orig.to(tl.float32)
        dy = dy_orig.to(tl.float32)

        # dgamma = dy * x / rms
        acc += dy * x / rms

    tl.store(
        dgamma_ptr + offsets,
        acc,
        mask=mask,
    )


class RMSNormFunction(Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float = 1e-5,
    ):
        # x -> [B, T, D] or any shape ending in D
        x = x.contiguous()
        gamma = gamma.contiguous()

        D = x.shape[-1]
        N = x.numel() // D

        # Flatten logically to [N, D]
        x_2d = x.reshape(N, D)

        out = torch.empty_like(x_2d)

        # One RMS per row
        rms = torch.empty(
            (N,),
            device=x.device,
            dtype=torch.float32,
        )

        BLOCK_SIZE = triton.next_power_of_2(D)

        grid = (N,)

        rmsnormFWD[grid](
            x_2d,
            out,
            gamma,
            rms,
            D=D,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Save everything needed by backward
        ctx.save_for_backward(
            x_2d,
            gamma,
            rms,
        )

        ctx.D = D
        ctx.N = N
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.input_shape = x.shape

        return out.reshape(x.shape)

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        x, gamma, rms = ctx.saved_tensors

        D = ctx.D
        N = ctx.N
        BLOCK_SIZE = ctx.BLOCK_SIZE

        grad_output = grad_output.contiguous()
        dy = grad_output.reshape(N, D)

        dx = torch.empty_like(x)

        grid_dx = (N,)

        rmsnorm_bwd_dx[grid_dx](
            x,
            dy,
            gamma,
            rms,
            dx,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        dgamma = torch.empty_like(gamma)

        grid_dgamma = (
            triton.cdiv(D, BLOCK_SIZE),
        )

        rmsnorm_bwd_dgamma[grid_dgamma](
            x,
            dy,
            rms,
            dgamma,
            D=D,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return (
            dx.reshape(ctx.input_shape),
            dgamma,
            None,  # eps has no gradient
        )


class RMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ):
        return RMSNormFunction.apply(
            x,
            self.weight,
            self.eps,
        )



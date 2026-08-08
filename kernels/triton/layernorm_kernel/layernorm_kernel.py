import torch
import triton
import triton.language as tl
import torch.nn as nn
from torch.autograd import Function


@triton.jit
def layernormFWD(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    layer_ptr,
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

    mu = tl.sum(x, axis=0) / D

    x_centered = x - mu

    sum_sq = tl.sum(
        x_centered * x_centered,
        axis=0,
    )

    layer = tl.sqrt(
        sum_sq / D + eps
    )


    w_orig = tl.load(
        w_ptr + offsets,
        mask=mask,
        other=0.0,
    )

    b_orig = tl.load(
        b_ptr + offsets,
        mask=mask,
        other=0.0,
    )

    w = w_orig.to(tl.float32)
    b = b_orig.to(tl.float32)
    y = (x_centered / layer) * w + b

    tl.store(
        y_ptr + row * D + offsets,
        y.to(x_orig.dtype),
        mask=mask,
    )
    tl.store(
        layer_ptr + row,
        layer,
    )


@triton.jit
def layernorm_bwd_dx(
    x_ptr,
    dy_ptr,
    gamma_ptr,
    layer_ptr,
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

    mu = tl.sum(x, axis=0) / D

    x_centered = x - mu

    layer = tl.load(layer_ptr + pid)

    g = dy * gamma

    sum_g = tl.sum(g, axis=0)

    c = tl.sum(
        g * x_centered,
        axis=0,
    )

    dx = (
        g / layer
        - sum_g / (D * layer)
        - x_centered * c
        / (D * layer * layer * layer)
    )

    tl.store(
        dx_ptr + pid * D + offsets,
        dx.to(x_orig.dtype),
        mask=mask,
    )

@triton.jit
def layernorm_bwd_dgamma(
    x_ptr,
    dy_ptr,
    layer_ptr,
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

        x = tl.load(
            x_ptr + row * D + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        dy = tl.load(
            dy_ptr + row * D + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        layer = tl.load(
            layer_ptr + row
        )

        # mean
        mu = tl.sum(x, axis=0) / D

        # normalized x
        x_hat = (x - mu) / layer

        # dgamma
        acc += dy * x_hat

    tl.store(
        dgamma_ptr + offsets,
        acc,
        mask=mask,
    )

@triton.jit
def layernorm_bwd_dbeta(
    dy_ptr,
    dbeta_ptr,
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

        dy = tl.load(
            dy_ptr + row * D + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        acc += dy

    tl.store(
        dbeta_ptr + offsets,
        acc,
        mask=mask,
    )

class LayerNormFunction(Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        eps: float = 1e-5,
    ):
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        D = x.shape[-1]
        N = x.numel() // D

        # [B, T, D] -> [N, D]
        x_2d = x.reshape(N, D)

        # Output
        out = torch.empty_like(x_2d)

        # One saved layer value per row
        layer = torch.empty(
            (N,),
            device=x.device,
            dtype=torch.float32,
        )

        BLOCK_SIZE = triton.next_power_of_2(D)

        grid = (N,)

        layernormFWD[grid](
            x_2d,
            out,
            gamma,
            beta,
            layer,
            D=D,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Save for backward
        ctx.save_for_backward(
            x_2d,
            gamma,
            beta,
            layer,
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
        x, gamma, beta, layer = ctx.saved_tensors

        D = ctx.D
        N = ctx.N
        BLOCK_SIZE = ctx.BLOCK_SIZE

        grad_output = grad_output.contiguous()

        dy = grad_output.reshape(N, D)

        dx = torch.empty_like(x)

        grid_dx = (N,)

        layernorm_bwd_dx[grid_dx](
            x,
            dy,
            gamma,
            layer,
            dx,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        dgamma = torch.empty_like(gamma)

        grid_dgamma = (
            triton.cdiv(D, BLOCK_SIZE),
        )

        layernorm_bwd_dgamma[grid_dgamma](
            x,
            dy,
            layer,
            dgamma,
            D=D,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        dbeta = torch.empty_like(beta)

        grid_dbeta = (
            triton.cdiv(D, BLOCK_SIZE),
        )

        layernorm_bwd_dbeta[grid_dbeta](
            dy,
            dbeta,
            D=D,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return (
            dx.reshape(ctx.input_shape),
            dgamma,
            dbeta,
            None,       # eps
        )


class LayerNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.bias = nn.Parameter(
            torch.zeros(dim)
        )

        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ):
        return LayerNormFunction.apply(
            x,
            self.weight,
            self.bias,
            self.eps,
        )


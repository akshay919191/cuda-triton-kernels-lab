import torch
import triton
import triton.language as tl


MBLOCK = 128
NBLOCK = 128
KBLOCK = 32

NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit
def matmul(
    xptr,
    yptr,
    out,
    B, H, M, N, K,
    NBLOCK: tl.constexpr,
    MBLOCK: tl.constexpr,
    KBLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    batch_id = pid_bh // H
    head_id = pid_bh % H

    x_base = (
        xptr
        + batch_id * H * M * K
        + head_id * M * K
    )

    y_base = (
        yptr
        + batch_id * H * K * N
        + head_id * K * N
    )

    out_base = (
        out
        + batch_id * H * M * N
        + head_id * M * N
    )

    offs_m = pid_m * MBLOCK + tl.arange(0, MBLOCK)
    offs_n = pid_n * NBLOCK + tl.arange(0, NBLOCK)

    offs_m = tl.max_contiguous(offs_m, MBLOCK)
    offs_n = tl.max_contiguous(offs_n, NBLOCK)

    acc = tl.zeros(
        (MBLOCK, NBLOCK),
        dtype=tl.float32,
    )

    for k in range(0, K, KBLOCK):

        offs_k = k + tl.arange(0, KBLOCK)

        x = tl.load(
            x_base
            + offs_m[:, None] * K
            + offs_k[None, :],

            mask=(
                (offs_m[:, None] < M)
                & (offs_k[None, :] < K)
            ),

            other=0.0,
        )

        y = tl.load(
            y_base
            + offs_k[:, None] * N
            + offs_n[None, :],

            mask=(
                (offs_k[:, None] < K)
                & (offs_n[None, :] < N)
            ),

            other=0.0,
        )

        acc += tl.dot(
            x,
            y,
            input_precision="tf32",
        )


    tl.store(
        out_base
        + offs_m[:, None] * N
        + offs_n[None, :],

        acc,

        mask=(
            (offs_m[:, None] < M)
            & (offs_n[None, :] < N)
        ),
    )


def triton_matmul(
    x,
    y,
    num_warps=NUM_WARPS,
    num_stages=NUM_STAGES,
):
    B, H, M, K = x.shape
    B2, H2, K2, N = y.shape

    assert B == B2
    assert H == H2
    assert K == K2

    out = torch.empty(
        (B, H, M, N),
        device=x.device,
        dtype=x.dtype,
    )


    grid = (
        triton.cdiv(M, MBLOCK),
        triton.cdiv(N, NBLOCK),
        B * H,
    )

    matmul[grid](
        x,
        y,
        out,
        B, H, M, N, K,

        NBLOCK=NBLOCK,
        MBLOCK=MBLOCK,
        KBLOCK=KBLOCK,

        num_warps=num_warps,

        num_stages=num_stages,
    )

    return out


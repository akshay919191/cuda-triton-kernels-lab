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
    out_ptr,
    B, H, M, N, K,
    NBLOCK: tl.constexpr,
    MBLOCK: tl.constexpr,
    KBLOCK: tl.constexpr,
    GROUP: tl.constexpr
):
    pid = tl.program_id(1)
    BH  = tl.program_id(0)

    num_pid_m = tl.cdiv(M, MBLOCK) # number of rows
    num_pid_n = tl.cdiv(N, NBLOCK) # number of cols

    group_size = GROUP * num_pid_n     # total tiles in GROUP

    group_id = pid // group_size       # total pid by this gives group id
    pid_in_group = pid % group_size    # pid number in group

    first_pid_m = group_id * GROUP     # where the first row starts

    group_size_m = min(num_pid_m - first_pid_m, GROUP)

    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_am = pid_m * MBLOCK + tl.arange(0, MBLOCK)
    offs_bn = pid_n * NBLOCK + tl.arange(0, NBLOCK)
    offs_k = tl.arange(0, KBLOCK)

    offs_am = tl.max_contiguous(offs_am, MBLOCK)
    offs_am = tl.multiple_of(offs_am, MBLOCK)

    offs_bn = tl.max_contiguous(offs_bn, NBLOCK)
    offs_bn = tl.multiple_of(offs_bn, NBLOCK)

    offs_k = tl.max_contiguous(offs_k, KBLOCK)
    offs_k = tl.multiple_of(offs_k, KBLOCK)

    a_ptrs = (
        xptr
        + BH * M * K
        + offs_am[:, None] * K
        + offs_k[None, :]
    )

    b_ptrs = (
        yptr
        + BH * K * N
        + offs_k[:, None] * N
        + offs_bn[None, :]
    )

    acc = tl.zeros((MBLOCK, NBLOCK), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, KBLOCK)):

        k_mask = offs_k < (K - k * KBLOCK)

        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & k_mask[None, :],
            other=0.0
        )

        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_bn[None, :] < N),
            other=0.0
        )

        acc = tl.dot(a, b, acc)

        a_ptrs += KBLOCK
        b_ptrs += KBLOCK * N

    c_ptrs = (
        out_ptr
        + BH * M * N
        + offs_am[:, None] * N
        + offs_bn[None, :]
    )

    tl.store(
        c_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N)
    )


def triton_matmul(x, y, GROUP=2):

    x = x.contiguous()
    y = y.contiguous()

    B, H, M, K = x.shape
    _, _, K2, N = y.shape

    assert K == K2

    out = torch.empty(
        (B, H, M, N),
        device=x.device,
        dtype=x.dtype
    )

    num_pid_m = triton.cdiv(M, MBLOCK)
    num_pid_n = triton.cdiv(N, NBLOCK)

    grid = (
        B * H,
        num_pid_m * num_pid_n,
    )

    matmul[grid](
        x,
        y,
        out,
        B, H, M, N, K,
        NBLOCK=NBLOCK,
        MBLOCK=MBLOCK,
        KBLOCK=KBLOCK,
        GROUP=GROUP,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return out
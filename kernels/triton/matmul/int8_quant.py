import torch
import triton
import triton.language as tl


GROUP = 2


AUTOTUNE_CONFIGS = [
    triton.Config(
        {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 32},
        num_warps=4,
        num_stages=2,
    ),

    triton.Config(
        {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 32},
        num_warps=4,
        num_stages=3,
    ),

    triton.Config(
        {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 32},
        num_warps=8,
        num_stages=2,
    ),

    triton.Config(
        {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 32},
        num_warps=8,
        num_stages=3,
    ),

    triton.Config(
        {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 64},
        num_warps=4,
        num_stages=2,
    ),

    triton.Config(
        {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 64},
        num_warps=4,
        num_stages=3,
    ),

    triton.Config(
        {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 64},
        num_warps=8,
        num_stages=2,
    ),

    triton.Config(
        {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 64},
        num_warps=8,
        num_stages=3,
    ),
]



@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def int8_matmul(
    xptr,
    yptr,
    out_ptr,

    B,
    H,
    M,
    N,
    K,

    MBLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
    KBLOCK: tl.constexpr,

    GROUP: tl.constexpr,
):

    pid = tl.program_id(1)
    BH = tl.program_id(0)

    num_pid_m = tl.cdiv(M, MBLOCK)
    num_pid_n = tl.cdiv(N, NBLOCK)

    group_size = GROUP * num_pid_n

    group_id = pid // group_size
    pid_in_group = pid % group_size

    first_pid_m = group_id * GROUP

    group_size_m = min(
        num_pid_m - first_pid_m,
        GROUP,
    )

    pid_m = first_pid_m + (
        pid_in_group % group_size_m
    )

    pid_n = pid_in_group // group_size_m

    offs_am = (
        pid_m * MBLOCK
        + tl.arange(0, MBLOCK)
    )

    offs_bn = (
        pid_n * NBLOCK
        + tl.arange(0, NBLOCK)
    )

    offs_k = tl.arange(0, KBLOCK)

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
    acc = tl.zeros(
        (MBLOCK, NBLOCK),
        dtype=tl.int32,
    )

    for k in range(0, tl.cdiv(K, KBLOCK)):

        k_mask = offs_k < (
            K - k * KBLOCK
        )

        a = tl.load(
            a_ptrs,
            mask=(
                (offs_am[:, None] < M)
                & k_mask[None, :]
            ),
            other=0,
        )

        b = tl.load(
            b_ptrs,
            mask=(
                k_mask[:, None]
                & (offs_bn[None, :] < N)
            ),
            other=0,
        )

        acc = tl.dot(
            a,
            b,
            acc,
        )

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
        acc,
        mask=(
            (offs_am[:, None] < M)
            & (offs_bn[None, :] < N)
        ),
    )



def quantize_int8(x):

    scale = x.abs().amax() / 127.0

    x_int8 = torch.clamp(
        torch.round(x / scale),
        -127,
        127,
    ).to(torch.int8)

    return x_int8, scale

def dequantize_int8(x , scale):

    return x * scale

def testss():
    x = torch.Tensor([1.0 , 2.2 , 3.45])
    y , scale = quantize_int8(x)
    z = dequantize_int8(y , scale)

    print(torch.allclose(x , z))
    print(z - x)
    error = (x - z).abs()

    print("max error :", error.max().item())
    print("mean error:", error.mean().item())

def triton_int8_matmul(x, y):

    x_int8, scale_x = quantize_int8(x)
    y_int8, scale_y = quantize_int8(y)

    B, H, M, K = x.shape
    _, _, K2, N = y.shape

    assert K == K2

    out_int32 = torch.empty(
        (B, H, M, N),
        device=x.device,
        dtype=torch.int32,
    )

    grid = lambda META: (
        B * H,
        triton.cdiv(M, META["MBLOCK"])
        * triton.cdiv(N, META["NBLOCK"]),
    )

    int8_matmul[grid](
        x_int8,
        y_int8,
        out_int32,

        B,
        H,
        M,
        N,
        K,

        GROUP=GROUP,
    )


    out = (
        out_int32.float()
        * scale_x
        * scale_y
    )

    return out



def main():

    B = 1
    H = 1
    M = 1024
    N = 1024
    K = 1024

    print(
        f"Shape: "
        f"B={B}, H={H}, "
        f"M={M}, N={N}, K={K}"
    )


    x = torch.randn(
        (B, H, M, K),
        device="cuda",
        dtype=torch.float16,
    )

    y = torch.randn(
        (B, H, K, N),
        device="cuda",
        dtype=torch.float16,
    )

    ref = torch.matmul(x, y)


    out = triton_int8_matmul(
        x,
        y,
    )


    error = (
        out.float()
        - ref.float()
    ).abs()

    max_error = error.max().item()
    mean_error = error.mean().item()

    relative_error = (
        error
        / ref.float().abs().clamp_min(1e-5)
    ).mean().item()

    print()
    print("INT8 QUANTIZATION")
    print("-" * 50)

    print(
        f"X dtype       : {x.dtype}"
    )

    print(
        f"Quantized X   : int8"
    )

    print(
        f"Quantized Y   : int8"
    )

    print(
        f"Accumulator   : int32"
    )

    print(
        f"Output dtype  : {out.dtype}"
    )

    print()

    print(
        f"Max error     : {max_error:.6f}"
    )

    print(
        f"Mean error    : {mean_error:.6f}"
    )

    print(
        f"Mean rel error: {relative_error:.6f}"
    )


if __name__ == "__main__":
    testss()
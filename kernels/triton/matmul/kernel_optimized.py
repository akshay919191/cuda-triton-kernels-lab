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
def matmul(
    xptr,
    yptr,
    out_ptr,

    B, H, M, N, K,

    MBLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
    KBLOCK: tl.constexpr,

    GROUP: tl.constexpr,
    HINT_MODE: tl.constexpr,
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


    if HINT_MODE >= 1:

        offs_am = tl.multiple_of(
            offs_am,
            MBLOCK,
        )

        offs_bn = tl.multiple_of(
            offs_bn,
            NBLOCK,
        )

        offs_k = tl.multiple_of(
            offs_k,
            KBLOCK,
        )

    if HINT_MODE >= 2:

        offs_am = tl.max_contiguous(
            offs_am,
            MBLOCK,
        )

        offs_bn = tl.max_contiguous(
            offs_bn,
            NBLOCK,
        )

        offs_k = tl.max_contiguous(
            offs_k,
            KBLOCK,
        )

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
        dtype=tl.float32,
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
            other=0.0,
        )


        b = tl.load(
            b_ptrs,
            mask=(
                k_mask[:, None]
                & (offs_bn[None, :] < N)
            ),
            other=0.0,
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
        acc.to(out_ptr.dtype.element_ty),
        mask=(
            (offs_am[:, None] < M)
            & (offs_bn[None, :] < N)
        ),
    )



def triton_matmul(
    x,
    y,
    HINT_MODE=0,
):

    x = x.contiguous()
    y = y.contiguous()

    B, H, M, K = x.shape
    _, _, K2, N = y.shape

    assert K == K2

    out = torch.empty(
        (B, H, M, N),
        device=x.device,
        dtype=x.dtype,
    )


    grid = lambda META: (
        B * H,
        triton.cdiv(M, META["MBLOCK"])
        * triton.cdiv(N, META["NBLOCK"]),
    )

    matmul[grid](
        x,
        y,
        out,

        B, H, M, N, K,

        GROUP=GROUP,

        HINT_MODE=HINT_MODE,
    )

    return out


def benchmark(
    fn,
    warmup=30,
    rep=200,
):

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    for _ in range(rep):
        fn()

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / rep


TEST_SHAPES = [

    (2, 4, 512, 512, 512),

    (1, 1, 1024, 1024, 1024),

    (1, 1, 1024, 4096, 1024),

    (3, 1, 4096, 4096, 4096),

    (3, 1, 4096, 1024, 1024),


]


def main():

    modes = [
        (0, "NONE"),
        (1, "MULTIPLE_OF"),
        (2, "MULTIPLE_OF + MAX_CONTIGUOUS"),
    ]


    for B, H, M, N, K in TEST_SHAPES:

        print()
        print("#" * 110)

        print(
            f"Shape: "
            f"B={B}, H={H}, "
            f"M={M}, N={N}, K={K}"
        )

        print("#" * 110)

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

        flops = (
            2
            * B
            * H
            * M
            * N
            * K
        )

        torch_ms = benchmark(
            lambda: torch.matmul(x, y),
            warmup=30,
            rep=200,
        )

        torch_tflops = (
            flops
            / (torch_ms * 1e9)
        )


        print(
            f"{'Mode':<35}"
            f"{'Error':<12}"
            f"{'Triton ms':<15}"
            f"{'Triton TFLOPS':<16}"
            f"{'Torch ms':<15}"
            f"{'Torch TFLOPS':<16}"
            f"{'Speedup':<10}"
        )

        print("-" * 120)

        for mode, name in modes:

            out = triton_matmul(
                x,
                y,
                HINT_MODE=mode,
            )

            max_error = (
                out.float()
                - ref.float()
            ).abs().max().item()

            torch.testing.assert_close(
                out,
                ref,
                atol=1e-2,
                rtol=1e-2,
            )

            triton_ms = benchmark(
                lambda: triton_matmul(
                    x,
                    y,
                    HINT_MODE=mode,
                ),
                warmup=30,
                rep=200,
            )

            triton_tflops = (
                flops
                / (triton_ms * 1e9)
            )

            speedup = (
                torch_ms
                / triton_ms
            )

            print(
                f"{name:<35}"
                f"{max_error:<12.4g}"
                f"{triton_ms:<15.3f}"
                f"{triton_tflops:<16.2f}"
                f"{torch_ms:<15.3f}"
                f"{torch_tflops:<16.2f}"
                f"{speedup:<10.2f}"
            )



if __name__ == "__main__":
    main()
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# Add repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Import Triton GELU
from kernels.triton.gelu_kernel.gelu_kernel import gelu


def sync():
    torch.cuda.synchronize()


def rand_half(shape):
    return torch.randn(
        *shape,
        device="cuda",
        dtype=torch.float16
    ).contiguous()


def check_close(custom, ref, atol=5e-2, rtol=5e-2):
    custom = custom.float()
    ref = ref.float()

    diff = (custom - ref).abs()

    print(
        f"max_error={diff.max().item():.6f}, "
        f"mean_error={diff.mean().item():.6f}"
    )

    torch.testing.assert_close(
        custom,
        ref,
        atol=atol,
        rtol=rtol,
    )


def bench_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()

    sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        fn()

    end.record()

    sync()

    return start.elapsed_time(end) / iters


def test_gelu(shape):
    print(f"\nTesting GELU {shape}")

    x = rand_half(shape)

    # Triton
    y = gelu(x)
    sync()

    # PyTorch reference
    y_ref = F.gelu(
        x,
        approximate="tanh"
    )

    check_close(y, y_ref)

    print("CORRECT PASS")


def benchmark_gelu(shape):
    print(f"\nBenchmark GELU {shape}")

    x = rand_half(shape)

    triton_fn = lambda: gelu(x)

    torch_fn = lambda: F.gelu(
        x,
        approximate="none"
    )

    triton_ms = bench_ms(triton_fn)
    torch_ms = bench_ms(torch_fn)

    print(
        f"Triton: {triton_ms:.4f} ms\n"
        f"PyTorch: {torch_ms:.4f} ms\n"
        f"Speedup: {torch_ms/triton_ms:.2f}x"
    )


def main():
    assert torch.cuda.is_available()

    shapes = [
        (1, 1, 128, 64),
        (1, 4, 512, 128),
        (2, 4, 1024, 128),
        (1, 4, 512, 512),
    ]

    print("========== GELU correctness ==========")

    for s in shapes:
        test_gelu(s)


    print("\n========== GELU benchmark ==========")

    for s in shapes:
        benchmark_gelu(s)


if __name__ == "__main__":
    main()
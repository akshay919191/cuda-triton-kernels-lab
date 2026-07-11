import torch
import fused_residual_rmsnorm_cuda


def main():
    assert torch.cuda.is_available()

    x = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.float16)

    out = fused_residual_rmsnorm_cuda.forward(x)
    y = out[0] if isinstance(out, (tuple, list)) else out

    torch.cuda.synchronize()

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("PASS import/build smoke test")


if __name__ == "__main__":
    main()

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2, dtype=torch.float32) / dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_seq_len

    def forward(self, x, position_ids):
        """
        x:            [batch, heads, seq, head_dim]
        position_ids: [batch, seq]
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected x to have shape [B, H, S, D], got {x.shape}"
            )

        if position_ids.ndim != 2:
            raise ValueError(
                f"Expected position_ids to have shape [B, S], "
                f"got {position_ids.shape}"
            )

        if x.shape[0] != position_ids.shape[0]:
            raise ValueError("Batch dimensions do not match")

        if x.shape[2] != position_ids.shape[1]:
            raise ValueError("Sequence dimensions do not match")

        if x.shape[-1] != self.inv_freq.numel() * 2:
            raise ValueError(
                f"x head_dim is {x.shape[-1]}, but RoPE was initialized "
                f"with dimension {self.inv_freq.numel() * 2}"
            )

        # Compute angles entirely in FP32.
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)

        inv_freq_expanded = inv_freq[None, :, None].expand(
            position_ids.shape[0], -1, 1
        )

        position_ids_expanded = position_ids.to(
            device=x.device,
            dtype=torch.float32
        )[:, None, :]

        freqs = (
            inv_freq_expanded @ position_ids_expanded
        ).transpose(1, 2)

        emb = torch.cat((freqs, freqs), dim=-1)

        # [B, S, D] -> [B, 1, S, D]
        cos = emb.cos().to(dtype=x.dtype).unsqueeze(1)
        sin = emb.sin().to(dtype=x.dtype).unsqueeze(1)

        return x * cos + rotate_half(x) * sin


def apply_hf_rope(hf_rope, x, position_ids):
    hf_cos, hf_sin = hf_rope(x, position_ids)

    # HF returns [B, S, D].
    hf_cos = hf_cos.unsqueeze(1)
    hf_sin = hf_sin.unsqueeze(1)

    return x * hf_cos + rotate_half(x) * hf_sin


def run_hard_tests():
    torch.manual_seed(42)

    batch = 2
    heads = 4
    seq = 16
    head_dim = 64

    x_fp32 = torch.randn(batch, heads, seq, head_dim)

    config = LlamaConfig(
        # Hidden size is all attention heads combined.
        hidden_size=heads * head_dim,
        num_attention_heads=heads,
        num_key_value_heads=heads,

        # Explicitly remove ambiguity.
        head_dim=head_dim,

        max_position_embeddings=512,
        rope_theta=10000.0,
    )

    hf_rope = LlamaRotaryEmbedding(config=config).to(torch.float32)
    my_rope = RotaryEmbedding(
        dim=head_dim,
        max_seq_len=512,
        base=10000.0,
    )

    print("Running Hard RoPE Tests...\n")

    # --------------------------------------------------
    # Test 1: Standard positions
    # --------------------------------------------------
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)

    hf_out = apply_hf_rope(hf_rope, x_fp32, position_ids)
    my_out = my_rope(x_fp32, position_ids)

    torch.testing.assert_close(
        my_out,
        hf_out,
        rtol=1e-5,
        atol=1e-6,
    )

    print("✅ Test 1 Passed: FP32 standard sequence")

    # --------------------------------------------------
    # Test 2: BF16 path
    # --------------------------------------------------
    x_bf16 = x_fp32.to(torch.bfloat16)

    hf_out_bf16 = apply_hf_rope(
        hf_rope,
        x_bf16,
        position_ids,
    )

    my_out_bf16 = my_rope(
        x_bf16,
        position_ids,
    )

    torch.testing.assert_close(
        my_out_bf16,
        hf_out_bf16,
        rtol=0,
        atol=0,
    )

    print("✅ Test 2 Passed: BF16 matches Hugging Face BF16 path")

    # --------------------------------------------------
    # Test 3: Arbitrary KV-cache positions
    # --------------------------------------------------
    cache_position_ids = torch.tensor([
        [10, 11, 12],
        [50, 51, 52],
    ])

    x_cache = torch.randn(batch, heads, 3, head_dim)

    hf_out_cache = apply_hf_rope(
        hf_rope,
        x_cache,
        cache_position_ids,
    )

    my_out_cache = my_rope(
        x_cache,
        cache_position_ids,
    )

    torch.testing.assert_close(
        my_out_cache,
        hf_out_cache,
        rtol=1e-5,
        atol=1e-6,
    )

    print("✅ Test 3 Passed: Arbitrary KV-cache positions")

    # --------------------------------------------------
    # Test 4: Position beyond nominal maximum
    # --------------------------------------------------
    overflow_ids = torch.tensor([[600, 601, 602]])
    x_overflow = torch.randn(1, heads, 3, head_dim)

    my_out_overflow = my_rope(
        x_overflow,
        overflow_ids,
    )

    assert torch.isfinite(my_out_overflow).all()

    print("✅ Test 4 Passed: Positions beyond nominal cache length")

    # --------------------------------------------------
    # Test 5: Gradient correctness
    # --------------------------------------------------
    x_grad = torch.randn(
        batch,
        heads,
        seq,
        head_dim,
        requires_grad=True,
    )

    my_grad_out = my_rope(x_grad, position_ids)
    my_grad_out.square().mean().backward()

    assert x_grad.grad is not None
    assert torch.isfinite(x_grad.grad).all()

    print("✅ Test 5 Passed: Finite input gradients")

    print("\n🎉 All corrected RoPE tests passed.")


if __name__ == "__main__":
    run_hard_tests()
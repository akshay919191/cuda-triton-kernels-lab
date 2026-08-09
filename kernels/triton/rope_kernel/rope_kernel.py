import triton
import triton.language as tl
import torch
from torch.autograd import Function
import torch.nn as nn

@triton.jit
def sincos_cache_kernel(
    sin_cache,
    cos_cache,
    max_seq_len,
    rotary_dim,
    base: tl.constexpr,
    BLOCK_POS: tl.constexpr,    
    BLOCK_DIM: tl.constexpr,
):
    pid_pos = tl.program_id(0)
    pid_dim = tl.program_id(1)

    pos_offs = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)
    dim_offs = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    half_rotary = rotary_dim // 2

    pos_mask = pos_offs < max_seq_len
    dim_mask = dim_offs < half_rotary

    inv_freq = tl.exp(
        -2.0
        * dim_offs.to(tl.float32)
        / rotary_dim
        * tl.log(tl.full((), base, tl.float32))
    )

    theta = (
        pos_offs[:, None].to(tl.float32)
        * inv_freq[None, :]
    )

    mask = pos_mask[:, None] & dim_mask[None, :]

    sin_val = tl.sin(theta)
    cos_val = tl.cos(theta)

    offsets = (
        pos_offs[:, None] * half_rotary
        + dim_offs[None, :]
    )

    tl.store(
        sin_cache + offsets,
        sin_val,
        mask=mask,
    )

    tl.store(
        cos_cache + offsets,
        cos_val,
        mask=mask,
    )

@triton.jit
def rope_kernel(
    xptr,
    optr,
    sin_cache,
    cos_cache,
    position_ids,
    rotary_dim,
    max_seq_len,
    head_dim,
    seq_len,
    num_heads,
    position_offset,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    seq_block = tl.program_id(2)

    seq_offs = seq_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_offs = tl.arange(0, BLOCK_DIM)

    seq_mask = seq_offs < seq_len
    dim_mask = dim_offs < head_dim
    mask2d = seq_mask[:, None] & dim_mask[None, :]

    if HAS_POSITION_IDS:
        positions = tl.load(position_ids + batch * seq_len + seq_offs, mask=seq_mask, other=0)
    else:
        positions = position_offset + seq_offs

    base_idx = (batch * num_heads + head) * seq_len + seq_offs
    x_offs = base_idx[:, None] * head_dim + dim_offs[None, :]

    x = tl.load(xptr + x_offs, mask=mask2d, other=0.0)

    half_rotary = rotary_dim // 2

    is_first = dim_offs < half_rotary
    is_second = (dim_offs >= half_rotary) & (dim_offs < rotary_dim)
    is_rotary = is_first | is_second

    pair_offs = tl.where(is_first, dim_offs + half_rotary, dim_offs - half_rotary)
    x_pair_offs = base_idx[:, None] * head_dim + pair_offs[None, :]
    
    x_pair = tl.load(xptr + x_pair_offs, mask=mask2d & is_rotary, other=0.0)

    cache_dim_offs = tl.where(is_first, dim_offs, dim_offs - half_rotary)
    cache_offs = positions[:, None] * half_rotary + cache_dim_offs[None, :]

    cos_val = tl.load(cos_cache + cache_offs, mask=mask2d & is_rotary, other=1.0)
    sin_val = tl.load(sin_cache + cache_offs, mask=mask2d & is_rotary, other=0.0)

    if BACKWARD:
        sin_val = -sin_val

    sign = tl.where(is_first, -1.0, 1.0)
    y = x * cos_val + sign * x_pair * sin_val

    tl.store(optr + x_offs, y, mask=mask2d)


def build_rope_cache(max_seq_len, rotary_dim, base=10000.0, device="cuda", dtype=torch.float32):
    sin_cache = torch.empty((max_seq_len, rotary_dim // 2), device=device, dtype=dtype)
    cos_cache = torch.empty((max_seq_len, rotary_dim // 2), device=device, dtype=dtype)
    
    BLOCK_POS = 128
    BLOCK_DIM = triton.next_power_of_2(rotary_dim // 2)
    
    grid = (triton.cdiv(max_seq_len, BLOCK_POS), triton.cdiv(rotary_dim // 2, BLOCK_DIM))
    sincos_cache_kernel[grid](
        sin_cache, cos_cache, max_seq_len, rotary_dim, base, BLOCK_POS, BLOCK_DIM
    )
    return sin_cache, cos_cache


class RoPEFunction(Function):
    @staticmethod
    def forward(ctx, x, sin_cache, cos_cache, position_ids=None, rotary_dim=None, position_offset=0):
        batch, num_heads, seq_len, head_dim = x.shape
        if rotary_dim is None:
            rotary_dim = head_dim
            
        x = x.contiguous()
        out = torch.empty_like(x)
        
        pos_ids = position_ids if position_ids is not None else x
        
        BLOCK_SEQ = 16
        BLOCK_DIM = triton.next_power_of_2(head_dim)
        
        grid = (batch, num_heads, triton.cdiv(seq_len, BLOCK_SEQ))
        
        rope_kernel[grid](
            x, out, sin_cache, cos_cache, pos_ids,
            rotary_dim, sin_cache.shape[0], head_dim, seq_len, num_heads,
            position_offset,
            BLOCK_SEQ=BLOCK_SEQ,
            BLOCK_DIM=BLOCK_DIM,
            HAS_POSITION_IDS=position_ids is not None,
            BACKWARD=False,
        )
        
        ctx.save_for_backward(sin_cache, cos_cache, position_ids)
        ctx.rotary_dim = rotary_dim
        ctx.position_offset = position_offset
        ctx.head_dim = head_dim
        ctx.seq_len = seq_len
        ctx.num_heads = num_heads
        return out

    @staticmethod
    def backward(ctx, grad_output):
        sin_cache, cos_cache, position_ids = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        
        batch, num_heads, seq_len, head_dim = grad_output.shape
        
        pos_ids = position_ids if position_ids is not None else grad_output
        
        BLOCK_SEQ = 16
        BLOCK_DIM = triton.next_power_of_2(head_dim)
        
        grid = (batch, num_heads, triton.cdiv(seq_len, BLOCK_SEQ))
        
        rope_kernel[grid](
            grad_output, grad_input, sin_cache, cos_cache, pos_ids,
            ctx.rotary_dim, sin_cache.shape[0], ctx.head_dim, ctx.seq_len, ctx.num_heads,
            ctx.position_offset,
            BLOCK_SEQ=BLOCK_SEQ,
            BLOCK_DIM=BLOCK_DIM,
            HAS_POSITION_IDS=position_ids is not None,
            BACKWARD=True,
        )
        return grad_input, None, None, None, None, None


class RoPE(nn.Module):
    def __init__(self, head_dim, rotary_dim=None, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim if rotary_dim is not None else head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)

    def _build_cache(self, device, dtype):
        if self.sin_cache.numel() == 0 or self.sin_cache.device != device:
            sin_cache, cos_cache = build_rope_cache(
                self.max_seq_len, self.rotary_dim, self.base, device, dtype
            )
            self.sin_cache = sin_cache
            self.cos_cache = cos_cache

    def forward(self, x, position_ids=None, position_offset=0):
        self._build_cache(x.device, x.dtype)
        return RoPEFunction.apply(
            x, self.sin_cache, self.cos_cache, position_ids, self.rotary_dim, position_offset
        )


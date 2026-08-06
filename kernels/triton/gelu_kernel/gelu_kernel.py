import triton
import triton.language as tl
import torch

import triton
import triton.language as tl
import torch

@triton.jit
def gelu_kernel(
    aptr,
    bptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    a = tl.load(aptr + offset, mask=mask, other=0.0)
    x = a.to(tl.float32)

    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865475))

    tl.store(bptr + offset, y.to(tl.float16), mask=mask)


def gelu(x: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    
    BLOCK = 4096
    grid = (triton.cdiv(N, BLOCK),)
    
    gelu_kernel[grid](
        x, out, N,
        BLOCK=BLOCK
    )
    return out
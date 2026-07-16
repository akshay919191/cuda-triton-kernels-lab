Paste this **from repo root**. It will directly create/overwrite `README.md`. Bench numbers are taken from your passed benchmark output. 

````bash
cat > README.md <<'EOF'
# CUDA Kernels Lab

This repo is my CUDA kernel lab for transformer-style workloads.

I wanted to take operations that appear again and again in LLMs / diffusion / transformer blocks, write them manually in CUDA, expose them as PyTorch extensions, and then test them properly.

The focus is not just "kernel compiles."  
The focus is:

```text
math -> CUDA kernel -> PyTorch binding -> correctness test -> backward test -> benchmark
````

This repo currently includes normalization kernels, activation kernels, softmax, RoPE, and fused transformer-style ops.

---

## Implemented kernels

```text
RMSNorm forward/backward
LayerNorm forward/backward
Softmax forward/backward
GELU exact forward/backward
SiLU forward/backward
Fused bias + GELU exact forward/backward
Fused residual + RMSNorm forward/backward
RoPE cache builder + forward/backward
```

Most kernels use transformer-style tensor layout:

```text
[B, H, N, D]
```

where:

```text
B = batch size
H = number of heads
N = sequence length
D = hidden/head dimension
```

---

## Why these kernels?

These ops are common in real model code:

```text
RMSNorm / LayerNorm        -> transformer normalization
Softmax                    -> attention-style reductions
GELU / SiLU                -> MLP activations
Bias + GELU                -> fused MLP block operation
Residual + RMSNorm         -> transformer residual path
RoPE                       -> rotary position embeddings
```

The goal is to show both sides:

```text
model math understanding
+
low-level CUDA implementation
```

---

## Current results

Benchmarks were run on:

```text
GPU: NVIDIA RTX 3050 Laptop GPU
Input dtype: fp16
Reference: PyTorch eager reference implementations used inside benchmark file
```

These speedups are against the PyTorch reference code in this repo's benchmark file.
They are not claims against Apex, xFormers, FlashAttention, TorchInductor, cuDNN, or vendor-tuned production kernels.

| Kernel                   | Forward speedup | Backward speedup |
| ------------------------ | --------------: | ---------------: |
| RMSNorm                  |    ~8.2x - 8.7x |    ~9.1x - 10.0x |
| LayerNorm                |    ~2.9x - 4.6x |     ~1.3x - 2.1x |
| Softmax                  |    ~2.5x - 3.9x |     ~2.3x - 4.7x |
| GELU exact               |    ~3.0x - 4.1x |     ~3.0x - 5.0x |
| SiLU                     |    ~3.1x - 4.2x |     ~2.7x - 5.1x |
| Fused bias + GELU exact  |    ~3.9x - 5.5x |     ~3.2x - 4.1x |
| Fused residual + RMSNorm |   ~8.9x - 10.5x |     ~7.0x - 8.8x |
| RoPE cache build         |    ~7.9x - 9.3x |              N/A |
| RoPE                     |   ~9.8x - 13.2x |   ~14.4x - 20.6x |

Some strongest results:

```text
RoPE backward: up to ~20.6x
RoPE forward: up to ~13.2x
Fused residual RMSNorm forward: up to ~10.5x
RMSNorm backward: up to ~10.0x
Fused bias GELU forward: up to ~5.5x
```

---

## Correctness coverage

The test suite checks more than basic output.

It checks:

```text
forward correctness
backward correctness
gradient correctness
finite outputs
NaN / Inf detection
small edge cases
odd sequence lengths
non-power-of-two dimensions where supported
large-ish transformer shapes
backward should not mutate forward output
shape-specific behavior
```

For RoPE, the tests cover:

```text
build_cache correctness
position_ids = None
position_offset path
explicit position_ids path
forward rotation
backward rotation
tail passthrough when rotary_dim < D
```

For fused residual RMSNorm, the tests also verify:

```text
dx == dresidual
```

because the operation is:

```text
z = x + residual
y = RMSNorm(z)
```

so both branches receive the same gradient through the addition.

---

## Hard bounds and limitations

This project is honest about its current constraints.

### 1. Benchmarks are against PyTorch references

The speedups are against PyTorch eager reference implementations written in the benchmark file.

They are not claims against:

```text
Apex
xFormers
FlashAttention kernels
TorchInductor fused kernels
cuDNN
vendor-tuned production kernels
```

A production kernel may already fuse operations or use deeper architecture-specific tricks.

---

### 2. Expected layout is usually contiguous `[B, H, N, D]`

The kernels assume tensors are contiguous and laid out as:

```text
[B, H, N, D]
```

Python tests use contiguous inputs.

---

### 3. Dtype support is intentionally narrow

Current dtype assumptions:

```text
main input/output: fp16
LayerNorm gamma/beta: fp32
Fused bias GELU bias: fp32
RoPE cos/sin cache: fp32
internal math: fp32 where needed
```

Not currently covered:

```text
bf16
fp8
full fp32 dispatch
arbitrary dtype dispatch
```

---

### 4. Some optimized paths have shape constraints

Some kernels are hard-bound to vectorized paths.

Current important constraints:

```text
RMSNorm expects D divisible by 8
Fused residual RMSNorm expects D divisible by 8
RoPE requires rotary_dim to be even
RoPE requires rotary_dim <= D
RoPE cos_cache and sin_cache must be float32
```

These constraints exist because the kernels are written around specific vectorized memory access patterns.

---

### 5. Atomic reductions are not bit-exact deterministic

Some backward kernels use `atomicAdd` for reductions:

```text
dgamma
dbeta
dbias
```

Atomic ordering can change the exact bit pattern, so tests use numerical tolerances instead of strict bitwise equality.

---

### 6. GELU is exact GELU, not tanh approximation

The GELU kernels implement exact erf GELU:

```text
0.5 * x * (1 + erf(x / sqrt(2)))
```

The PyTorch reference uses:

```python
F.gelu(x, approximate="none")
```

This repo is not testing the tanh approximation path for GELU.

---

## Project structure

```text
kernels/
  cuda/
    rmsnorm_kernel/
    layernorm_kernel/
    softmax_kernel/
    gelu_kernel/
    silu_kernel/
    fused_bias_gelu_kernel/
    fused_residual_rmsnorm_kernel/
    rope_kernel/

tests/
  test_kernel_cuda.py

benchmarks/
  bench_cuda.py
```

---

## Build

Each kernel is a PyTorch CUDA extension.

Build from inside a kernel folder:

```bash
python setup.py build_ext --inplace
```

Example:

```bash
cd kernels/cuda/rmsnorm_kernel
python setup.py build_ext --inplace
```

---

## Run correctness tests

From repo root:

```bash
CUDA_LAUNCH_BLOCKING=1 python tests/test_kernel_cuda.py
```

Expected result:

```text
ALL CUDA KERNEL TESTS PASSED
```

---

## Run benchmark suite

From repo root:

```bash
CUDA_LAUNCH_BLOCKING=1 python benchmarks/bench_cuda.py
```

Expected result:

```text
ALL CORRECTNESS + BENCHMARK CHECKS PASSED
```

The benchmark file first verifies correctness, then runs timing.

---

## Benchmark method

Benchmarks use CUDA events.

Each benchmark includes:

```text
warmup iterations
multiple timed iterations
torch.cuda.synchronize()
forward timing
backward timing
speedup vs PyTorch reference
rough GB/s estimate
```

The GB/s estimate is approximate. Some kernels are not pure memory bandwidth kernels because they also do exp/erf/sqrt/reduction work.

---

## Example benchmark numbers

RMSNorm example:

```text
forward custom: 0.0387 ms
forward PyTorch: 0.3376 ms
speedup: 8.72x

backward custom: 0.1137 ms
backward PyTorch: 1.0688 ms
speedup: 9.40x
```

Fused residual RMSNorm example:

```text
forward custom: 0.0488 ms
forward PyTorch: 0.4734 ms
speedup: 9.70x

backward custom: 0.1366 ms
backward PyTorch: 1.2063 ms
speedup: 8.83x
```

RoPE example:

```text
cache build speedup: ~7.9x - 9.3x
forward speedup: ~9.8x - 13.2x
backward speedup: ~14.4x - 20.6x
```

---

## Kernel notes

### RMSNorm

RMSNorm computes:

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms * gamma
```

Backward returns:

```text
dx
dgamma
```

The current optimized path is hard-bound to dimensions divisible by 8.

---

### LayerNorm

LayerNorm computes mean and variance over the last dimension.

Backward returns:

```text
dx
dgamma
dbeta
```

This kernel supports more odd dimensions than the RMSNorm optimized path.

---

### Softmax

Softmax is applied over the last dimension.

The test checks:

```text
output correctness
row sums close to 1
backward correctness
```

---

### GELU exact

This uses exact erf GELU, not the tanh approximation.

Backward is manually implemented from the exact derivative.

---

### SiLU

SiLU computes:

```text
x * sigmoid(x)
```

Backward is manually implemented.

---

### Fused bias + GELU

This fuses:

```text
z = x + bias
y = GELU(z)
```

Backward returns:

```text
dx
dbias
```

This reduces extra intermediate work compared with doing bias add and GELU separately at Python level.

---

### Fused residual + RMSNorm

This fuses:

```text
z = x + residual
y = RMSNorm(z, gamma)
```

Backward returns:

```text
dx
dresidual
dgamma
```

Since `z = x + residual`, the test checks that:

```text
dx == dresidual
```

---

### RoPE

RoPE supports:

```text
cache building
position_ids = None
position_offset
explicit position_ids
partial rotary_dim
forward
backward
```

The cache tensors must be:

```text
float32 cos_cache
float32 sin_cache
```

---

## What this repo demonstrates

This repo is meant to show that I can go beyond high-level PyTorch code.

It demonstrates:

```text
manual CUDA indexing
grid/block launch setup
PyTorch extension binding
forward math implementation
manual backward math implementation
reduction kernels
fused kernels
edge-case testing
benchmarking with CUDA events
debugging numerical differences
```

---

## Current status

```text
[x] RMSNorm forward/backward
[x] LayerNorm forward/backward
[x] Softmax forward/backward
[x] GELU exact forward/backward
[x] SiLU forward/backward
[x] Fused bias + GELU exact forward/backward
[x] Fused residual + RMSNorm forward/backward
[x] RoPE cache/forward/backward
[x] All-kernel correctness test
[x] All-kernel benchmark suite
```

---

## Future work

Possible next improvements:

```text
add bf16 support
add half2 optimized paths
add scalar fallbacks for vectorized kernels
add TorchInductor comparison
add Nsight Compute reports
add occupancy/register/shared-memory notes
add CI test script
add benchmark plots
add more fused transformer blocks
```

---

## Final note

This repo is not trying to hide behind high-level APIs.

The goal is to show the full path:

```text
understand the math
write the CUDA
bind it to PyTorch
test it hard
benchmark it honestly
state the limitations clearly
```

That is the point of this project.
EOF

```
```

# CUDA Kernel Profiling Notes

This file records the profiling results for the CUDA kernels in this repo.

The goal of profiling is not only to show speedup numbers.  
The goal is to understand what each kernel is probably bounded by:

```text
memory bandwidth
reduction cost
special-function math
shared memory synchronization
atomic reductions
Python / PyTorch extension allocation overhead
```

All numbers below come from the real profiling script:

```bash
PROFILE_TABLES=0 BENCH_WARMUP=50 BENCH_ITERS=200 CUDA_LAUNCH_BLOCKING=1 python benchmarks/profile_real_cuda.py
```

---

## Hardware / software

```text
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
PyTorch: 2.7.1+cu118
CUDA: 11.8
BENCH_WARMUP: 50
BENCH_ITERS: 200
PROFILE_TABLES: False
```

The main profiling shape for most kernels was:

```text
[B, H, N, D] = [2, 4, 1024, 128]
```

Softmax was profiled on:

```text
[B, H, N, D] = [1, 4, 512, 512]
```

RoPE was profiled on:

```text
[B, H, N, D, rotary_dim] = [2, 4, 1024, 128, 128]
mode = position_ids
```

---

## Summary table

| Kernel | Forward custom | Forward ref | Forward speedup | Backward custom | Backward ref | Backward speedup | Likely bound |
|---|---:|---:|---:|---:|---:|---:|---|
| RMSNorm | 0.03622 ms | 0.33679 ms | 9.30x | 0.10947 ms | 1.26021 ms | 11.51x | reduction + memory + dgamma atomic |
| LayerNorm | 0.04215 ms | 0.19154 ms | 4.54x | 0.12675 ms | 0.22202 ms | 1.75x | mean/variance reductions + atomics |
| Softmax | 0.04043 ms | 0.15386 ms | 3.81x | 0.07510 ms | 0.24299 ms | 3.24x | reductions + exp + memory |
| GELU exact | 0.03864 ms | 0.15431 ms | 3.99x | 0.04788 ms | 0.16549 ms | 3.46x | erf/exp special-function math |
| SiLU | 0.03689 ms | 0.15426 ms | 4.18x | 0.04732 ms | 0.16370 ms | 3.46x | exp/sigmoid math |
| Fused bias + GELU exact | 0.03931 ms | 0.21652 ms | 5.51x | 0.06335 ms | 0.20371 ms | 3.22x | erf/exp + dbias atomic |
| Fused residual + RMSNorm | 0.04963 ms | 0.46703 ms | 9.41x | 0.13543 ms | 1.27198 ms | 9.39x | RMS reduction + sync + dgamma atomic |
| RoPE | 0.04834 ms | 0.46514 ms | 9.62x | 0.03916 ms | 0.67140 ms | 17.15x | memory + cos/sin cache reads |

RoPE cache build:

| Kernel | Custom | Reference | Speedup |
|---|---:|---:|---:|
| RoPE cache build | 0.01012 ms | 0.08440 ms | 8.34x |

---

## Correctness status

All profiling checks passed.

```text
ALL PROFILE CHECKS PASSED
```

The profiling script checks correctness before timing.

It checks:

```text
forward correctness
backward correctness
gradient correctness
finite outputs
RoPE cache correctness
RoPE position_ids path
RMSNorm / LayerNorm / fused gradients
```

---

## Kernel-by-kernel notes

## RMSNorm

Shape:

```text
[2, 4, 1024, 128]
```

Correctness:

```text
forward max error: 0.003906
dx max error: 0.008735
dgamma max error: 0.0006561
```

Speed:

```text
forward custom : 0.03622 ms
forward ref    : 0.33679 ms
forward speedup: 9.30x
forward approx : 115.8 GB/s

backward custom : 0.10947 ms
backward ref    : 1.26021 ms
backward speedup: 11.51x
backward approx : 57.5 GB/s
```

Likely bound:

```text
forward  -> reduction + memory bandwidth
backward -> reduction + memory bandwidth + dgamma atomic
```

RMSNorm forward is very strong. The backward is slower than forward because it has to compute `dx` and `dgamma`, and `dgamma` is a reduction.

Current hard bounds:

```text
input/output fp16
gamma fp16
internal accumulation fp32
optimized path expects D divisible by 8
```

---

## LayerNorm

Shape:

```text
[2, 4, 1024, 128]
```

Correctness:

```text
forward max error: 0.001953
dx max error: 0.007193
dgamma max error: 0.05476
dbeta max error: 7.629e-05
```

Speed:

```text
forward custom : 0.04215 ms
forward ref    : 0.19154 ms
forward speedup: 4.54x
forward approx : 99.5 GB/s

backward custom : 0.12675 ms
backward ref    : 0.22202 ms
backward speedup: 1.75x
backward approx : 49.6 GB/s
```

Likely bound:

```text
mean reduction
variance reduction
synchronization
dgamma/dbeta atomic reductions
```

LayerNorm backward is the weakest relative speedup in the current repo. That makes sense because LayerNorm backward has more reduction work than RMSNorm and returns both `dgamma` and `dbeta`.

Current hard bounds:

```text
input/output fp16
gamma/beta fp32
dgamma/dbeta fp32
atomicAdd used for reductions
```

Possible optimization direction:

```text
reduce atomic pressure
do block-level reductions first
use two-stage dgamma/dbeta reduction
reduce syncs
specialize for common D values
```

---

## Softmax

Shape:

```text
[1, 4, 512, 512]
```

Correctness:

```text
forward max error: 0.0002441
dx max error: 0.0003377
row_sum error: 0.0002301
```

Speed:

```text
forward custom : 0.04043 ms
forward ref    : 0.15386 ms
forward speedup: 3.81x
forward approx : 103.7 GB/s

backward custom : 0.07510 ms
backward ref    : 0.24299 ms
backward speedup: 3.24x
backward approx : 83.8 GB/s
```

Likely bound:

```text
max reduction
exp
sum reduction
normalization
memory traffic
```

Softmax has lower speedups than RMSNorm because it has more math per row and reduction stages.

Current behavior:

```text
softmax over last dimension
stable max-subtraction path
backward uses dx = y * (dy - sum(dy * y))
```

Possible optimization direction:

```text
warp-specialized path for small D
better block-level reduction
reduce shared memory traffic
specialize common D values like 64, 128, 512, 1024
```

---

## GELU exact

Shape:

```text
[2, 4, 1024, 128]
```

Correctness:

```text
forward max error: 0
dx max error: 0.001896
```

Speed:

```text
forward custom : 0.03864 ms
forward ref    : 0.15431 ms
forward speedup: 3.99x
forward approx : 108.5 GB/s

backward custom : 0.04788 ms
backward ref    : 0.16549 ms
backward speedup: 3.46x
backward approx : 131.4 GB/s
```

Likely bound:

```text
erf special-function math
exp in backward
memory bandwidth
```

This kernel uses exact GELU:

```text
0.5 * x * (1 + erf(x / sqrt(2)))
```

It does not use the tanh approximation.

Current hard bounds:

```text
exact GELU only
input/output fp16
internal math fp32
```

Possible optimization direction:

```text
half2 vectorization
optional tanh approximation kernel
fuse with bias or other MLP ops
```

---

## SiLU

Shape:

```text
[2, 4, 1024, 128]
```

Correctness:

```text
forward max error: 0.0001221
dx max error: 0.00195
```

Speed:

```text
forward custom : 0.03689 ms
forward ref    : 0.15426 ms
forward speedup: 4.18x
forward approx : 113.7 GB/s

backward custom : 0.04732 ms
backward ref    : 0.16370 ms
backward speedup: 3.46x
backward approx : 133.0 GB/s
```

Likely bound:

```text
exp/sigmoid special-function math
memory bandwidth
```

SiLU computes:

```text
y = x * sigmoid(x)
```

Backward recomputes sigmoid from `x`.

Possible optimization direction:

```text
half2 path
fuse with other elementwise ops
avoid unnecessary casts
```

---

## Fused bias + GELU exact

Shape:

```text
[2, 4, 1024, 128]
```

Correctness:

```text
forward max error: 0
dx max error: 0.001901
dbias max error: 0.000946
```

Speed:

```text
forward custom : 0.03931 ms
forward ref    : 0.21652 ms
forward speedup: 5.51x
forward approx : 160.1 GB/s

backward custom : 0.06335 ms
backward ref    : 0.20371 ms
backward speedup: 3.22x
backward approx : 132.4 GB/s
```

Likely bound:

```text
forward  -> erf math + memory bandwidth
backward -> erf/exp math + dbias atomic
```

This kernel fuses:

```text
z = x + bias
y = GELU(z)
```

The forward speedup is higher than plain GELU because it avoids doing bias-add and GELU as separate PyTorch-level operations.

Backward returns:

```text
dx
dbias
```

Current hard bounds:

```text
input/output fp16
bias fp32
dbias fp32
exact GELU only
atomicAdd for dbias
```

Possible optimization direction:

```text
reduce dbias atomic pressure
block-level dbias reduction
two-stage dbias reduction
half2 path for dx
```

---

## Fused residual + RMSNorm

Shape:

```text
[2, 4, 1024, 128]
```

Correctness:

```text
forward max error: 0.003906
dx max error: 0.006418
dresidual max error: 0.006418
dgamma max error: 0.0642
```

Speed:

```text
forward custom : 0.04963 ms
forward ref    : 0.46703 ms
forward speedup: 9.41x
forward approx : 126.8 GB/s

backward custom : 0.13543 ms
backward ref    : 1.27198 ms
backward speedup: 9.39x
backward approx : 77.4 GB/s
```

Likely bound:

```text
RMS reduction
shared memory synchronization
dgamma atomic
extra writes for dx and dresidual
```

This kernel fuses:

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

Since:

```text
z = x + residual
```

the test checks:

```text
dx == dresidual
```

This is one of the strongest kernels in the repo because fusion avoids materializing the intermediate residual output separately.

Current hard bounds:

```text
D divisible by 8
input/output fp16
gamma fp16
dgamma fp32
atomicAdd for dgamma
```

Possible optimization direction:

```text
reduce dgamma atomic pressure
reuse shared memory more tightly
profile barrier stalls with Nsight Compute
try half2/vectorized path
```

---

## RoPE

Shape:

```text
[2, 4, 1024, 128]
rotary_dim = 128
mode = position_ids
```

Correctness:

```text
cos cache max error: 6.104e-05
sin cache max error: 6.104e-05
forward max error: 0.0009766
dx max error: 0.001941
```

Speed:

```text
cache custom   : 0.01012 ms
cache ref      : 0.08440 ms
cache speedup  : 8.34x

forward custom : 0.04834 ms
forward ref    : 0.46514 ms
forward speedup: 9.62x
forward approx : 97.6 GB/s

backward custom : 0.03916 ms
backward ref    : 0.67140 ms
backward speedup: 17.15x
backward approx : 120.5 GB/s
```

Likely bound:

```text
memory bandwidth
cos/sin cache reads
position_ids indexing overhead
```

RoPE supports:

```text
position_ids = None
position_offset
explicit position_ids
partial rotary_dim
```

Current hard bounds:

```text
rotary_dim must be even
rotary_dim <= D
cos_cache fp32
sin_cache fp32
input/output fp16
```

The cache error around `6e-05` is acceptable for profiling because CUDA device math and PyTorch math can differ slightly for long sin/cos ranges. The actual forward/backward tests still pass with tight tolerances.

Possible optimization direction:

```text
half2 pair rotation
avoid repeated position index work
improve cache locality for cos/sin
specialize rotary_dim values like 32, 64, 128
```

---

## Overall interpretation

The strongest kernels right now are:

```text
RoPE backward
RMSNorm backward
Fused residual RMSNorm forward/backward
RMSNorm forward
Fused bias GELU forward
```

The weakest relative speedup is:

```text
LayerNorm backward
```

That is expected because LayerNorm backward has more reduction work and two parameter-gradient reductions.

---

## What the profile suggests

### Memory/reduction-bound kernels

```text
RMSNorm
LayerNorm
Softmax
Fused residual RMSNorm
```

These depend heavily on reductions and memory movement.

Main things to inspect later with Nsight Compute:

```text
barrier stalls
shared memory pressure
register pressure
global memory throughput
atomic overhead
```

---

### Math/special-function-bound kernels

```text
GELU exact
SiLU
Fused bias GELU
```

These use expensive math functions:

```text
erf
exp
sigmoid
```

Main things to inspect later:

```text
math pipe utilization
special function unit pressure
instruction throughput
```

---

### Memory/indexing-bound kernel

```text
RoPE
```

RoPE mostly rotates values using cached cos/sin.

Main things to inspect later:

```text
global load efficiency
cos/sin cache access
position_ids overhead
memory coalescing
```

---

## PyTorch profiler notes

The PyTorch profiler tables showed CPU-side overhead from:

```text
aten::empty_like
aten::empty_strided
aten::zeros
aten::fill_
cudaLaunchKernel
```

This is expected because the C++ extension allocates output tensors inside the wrapper.

For example:

```text
forward kernels allocate output tensors
backward kernels allocate dx / dgamma / dbias outputs
some backward paths call torch::zeros for reduction outputs
```

This does not mean the CUDA kernel itself is slow. It means the full extension call includes allocation overhead.

For deeper GPU analysis, Nsight Compute should be used.

---

## Nsight Compute next step

The current profiling script gives terminal-level timings.

For deeper kernel counters, use Nsight Compute:

```bash
ncu --target-processes all --set roofline python benchmarks/profile_real_cuda.py
```

For one kernel, create a smaller script that repeatedly calls only that kernel.

Useful Nsight metrics to inspect:

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
smsp__warps_active.avg.pct_of_peak_sustained_active
launch__registers_per_thread
launch__shared_mem_per_block
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_barrier_per_warp_active.pct
```

---

## Optimization priority

Based on the current numbers, the best next optimization targets are:

```text
1. LayerNorm backward
2. Fused bias GELU backward dbias reduction
3. RMSNorm / fused RMSNorm dgamma reduction
4. Softmax reduction path
5. RoPE half2/vectorized pair rotation
```

LayerNorm backward is the first real target because its speedup is only:

```text
1.75x
```

while most other kernels are much stronger.

---

## Final profiling status

```text
[x] RMSNorm profiled
[x] LayerNorm profiled
[x] Softmax profiled
[x] GELU exact profiled
[x] SiLU profiled
[x] Fused bias + GELU exact profiled
[x] Fused residual + RMSNorm profiled
[x] RoPE profiled
[x] Correctness passed before timing
[x] Terminal-level profiling script passed
[ ] Nsight Compute kernel-counter reports
```

Current conclusion:

```text
The kernels are correct, fast against the PyTorch reference path, and the main remaining work is deeper Nsight-level profiling plus targeted optimization of LayerNorm backward and atomic-heavy parameter-gradient paths.
```

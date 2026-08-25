# Triton MatMul — Tiled, Grouped & Autotuned

A from-scratch matrix multiplication kernel implemented in **Triton**, optimized through progressive experiments in tiling, program mapping, warp configuration, pipelining, cache reuse, compiler hints, and autotuning.

The goal of this project was not simply to beat `torch.matmul`, but to understand how Triton maps programs to GPU execution and how kernel design choices affect memory access, cache reuse, occupancy, and overall performance.

---

## Overview

The kernel computes batched matrix multiplication:

```text
X : [B, H, M, K]
Y : [B, H, K, N]

Output : [B, H, M, N]
```

For every batch/head pair:

```text
C = X @ Y
```

The implementation uses:

* Tiled matrix multiplication
* FP32 accumulation
* Configurable `M/N/K` tile sizes
* Grouped program ordering
* Explicit warp configuration
* Explicit pipeline stages
* Triton compiler hints
* Triton autotuning
* FP16 input/output with FP32 accumulation

---

# Kernel Structure

Each Triton program computes one output tile:

```text
C_tile = A_tile @ B_tile
```

with dimensions:

```text
MBLOCK × NBLOCK
```

The reduction over `K` is processed in chunks of:

```text
KBLOCK
```

The basic execution structure is:

```text
for k in K:

    load A[MBLOCK × KBLOCK]

    load B[KBLOCK × NBLOCK]

    accumulate:

        C += A @ B
```

The accumulator remains FP32 throughout the reduction:

```python
acc = tl.zeros(
    (MBLOCK, NBLOCK),
    dtype=tl.float32,
)
```

The result is converted back to the output dtype only when stored.

---

# Program Mapping

The kernel uses a 2D grid:

```python
grid = (
    B * H,
    num_pid_m * num_pid_n,
)
```

The first program dimension selects the batch/head pair:

```python
BH = tl.program_id(0)
```

The second dimension represents the output matrix tiles:

```python
pid = tl.program_id(1)
```

Instead of directly mapping:

```text
pid → (pid_m, pid_n)
```

the kernel uses **grouped ordering**.

---

# Grouped Program Mapping

The output matrix is divided into `MBLOCK × NBLOCK` tiles.

For example, with:

```text
M = 512
N = 512
MBLOCK = 128
NBLOCK = 128
```

there are:

```text
4 × 4 = 16 output tiles
```

A simple row-major mapping would execute:

```text
(0,0) → (0,1) → (0,2) → (0,3)
(1,0) → (1,1) → ...
```

The grouped mapping instead processes several `M` tiles together before moving across the `N` dimension.

For:

```text
GROUP = 2
```

the ordering is approximately:

```text
(0,0)
(1,0)

(0,1)
(1,1)

(0,2)
(1,2)
...
```

This changes the order in which CTAs access the matrices.

The motivation is to increase the probability that data loaded by neighboring programs remains useful in the cache hierarchy.

The important principle is:

> Program ordering affects memory reuse.

Grouping does not change the mathematical result. It changes **which CTAs execute next**, and therefore which cache lines may be reused.

---

# Memory Access Pattern

For a tile:

```text
A : MBLOCK × KBLOCK
B : KBLOCK × NBLOCK
C : MBLOCK × NBLOCK
```

the kernel repeatedly loads:

```text
A[m, k]
B[k, n]
```

for each `KBLOCK` chunk.

Because tensors are contiguous, the important question is not simply whether A or B is "good" or "bad".

The important question is:

> Which dimension is contiguous in memory, and how are lanes mapped onto the resulting addresses?

This determines whether accesses are coalesced and how effectively memory transactions are utilized.

Within a CTA, portions of the loaded tiles are reused across many multiply-accumulate operations.

Between neighboring CTAs, grouped ordering attempts to improve cache reuse.

---

# Tile Size Experiments

Several configurations were evaluated:

```text
64  × 64  × 32
64  × 128 × 32
128 × 64  × 32
128 × 128 × 32
128 × 128 × 64
128 × 256 × 32
256 × 128 × 32
```

The main configuration used during optimization was:

```text
MBLOCK = 128
NBLOCK = 128
KBLOCK = 32
```

The experiments showed that tile size is shape-dependent.

Larger tiles can improve arithmetic intensity and reduce program-launch overhead, but can also increase:

* register usage
* shared/local resource pressure
* wasted work on boundary tiles
* sensitivity to matrix shape

Therefore there is no universally optimal tile size.

---

# Warp Configuration

The kernel was tested with:

```text
num_warps = 4
num_warps = 8
```

The effect depended on matrix dimensions.

For example, one configuration produced approximately:

```text
1024 × 1024 × 1024

4 warps  → ~16.2 TFLOPS
8 warps  → ~14.5 TFLOPS
```

while other shapes favored different configurations.

This demonstrated that warp count should be treated as a tunable kernel parameter rather than assuming that more warps automatically means higher performance.

---

# Pipeline Stages

The kernel was tested with:

```text
num_stages = 2
num_stages = 3
```

Pipeline stages affect how aggressively Triton pipelines memory operations and computation.

More stages can hide memory latency, but they also increase resource requirements.

The optimal value therefore depends on:

* tile dimensions
* matrix shape
* warp count
* register/resource pressure
* GPU architecture

---

# Compiler Hints

The kernel also experimented with:

```python
tl.multiple_of(...)
```

and:

```python
tl.max_contiguous(...)
```

Three modes were compared:

```text
NONE

MULTIPLE_OF

MULTIPLE_OF + MAX_CONTIGUOUS
```

The hints were applied to the tile offsets when their alignment/contiguity properties were known.

Example:

```python
offs_am = tl.multiple_of(
    offs_am,
    MBLOCK,
)
```

and:

```python
offs_am = tl.max_contiguous(
    offs_am,
    MBLOCK,
)
```

## Observed behavior

The hints did **not** consistently improve performance.

For example:

```text
1024 × 1024 × 1024

NONE                       ~14.64 TFLOPS
MULTIPLE_OF                ~14.30 TFLOPS
MULTIPLE_OF + MAX_CONTIGUOUS
                           ~14.39 TFLOPS
```

while for other shapes the difference was negligible.

This is an important result.

Compiler hints are not optimization switches that automatically make a kernel faster.

They communicate information to the compiler. Whether that information changes generated code enough to matter depends on the surrounding kernel structure and hardware.

---

# Autotuning

The final implementation uses Triton's autotuning system.

Configurations vary:

```text
MBLOCK
NBLOCK
KBLOCK
num_warps
num_stages
```

Example configurations include:

```python
triton.Config(
    {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 32},
    num_warps=4,
    num_stages=2,
)
```

and:

```python
triton.Config(
    {"MBLOCK": 128, "NBLOCK": 128, "KBLOCK": 64},
    num_warps=8,
    num_stages=3,
)
```

Autotuning allows Triton to select a configuration based on the matrix shape.

The autotuning key is:

```python
key=["M", "N", "K"]
```

This means the selected configuration is associated with the problem dimensions.

---

# Performance

Performance was measured against:

```python
torch.matmul
```

using CUDA events after warmup.

Representative results on the development GPU:

### 1024 × 1024 × 1024

```text
Triton   ≈ 14.6 TFLOPS
PyTorch  ≈ 14.5 TFLOPS
```

### 1024 × 4096 × 1024

```text
Triton   ≈ 17.2 TFLOPS
PyTorch  ≈ 15.5 TFLOPS
```

### 4096 × 4096 × 4096

```text
Triton   ≈ 14.8 TFLOPS
PyTorch  ≈ 14.8 TFLOPS
```

### 4096 × 1024 × 1024 with B=3

```text
Triton   ≈ 16.1 TFLOPS
PyTorch  ≈ 15.6 TFLOPS
```

Performance varies substantially with matrix dimensions.

Small matrices are dominated more heavily by fixed launch and execution overhead, while larger matrices provide enough work for the tiled kernel to approach the GPU's sustained compute throughput.

---

# Important Observations

## 1. Alignment matters

Well-aligned dimensions often produced substantially better performance than irregular dimensions.

Boundary tiles require masking and can perform work on elements that are ultimately discarded.

---

## 2. Bigger tiles are not automatically better

Increasing tile size increases the amount of computation handled by a CTA, but also increases resource requirements.

The best configuration depends on:

```text
M × N × K
```

rather than on tile size alone.

---

## 3. More warps are not automatically faster

`num_warps=8` does not universally outperform `num_warps=4`.

The additional parallelism can be offset by increased resource usage and different scheduling behavior.

---

## 4. Grouping is about execution order

Grouping does not reduce the mathematical amount of work.

It changes:

```text
CTA ordering
    ↓
memory access ordering
    ↓
cache reuse opportunities
```

This is particularly relevant when multiple neighboring CTAs operate on related tiles.

---

## 5. Compiler hints are not magic

`tl.multiple_of` and `tl.max_contiguous` can communicate useful information to Triton's compiler, but they only matter when that information enables a meaningful code-generation improvement.

Sometimes the compiler already generates efficient code without them.

---

## 6. Shape matters enormously

A kernel can perform extremely well for one shape and poorly for another.

This is why serious GPU kernels generally rely on:

* multiple configurations
* autotuning
* shape-aware heuristics
* architecture-specific optimization

rather than a single hard-coded configuration.

---

# Correctness

The kernel is tested against:

```python
torch.matmul
```

using:

```python
torch.testing.assert_close(
    out,
    ref,
    atol=1e-2,
    rtol=1e-2,
)
```

Tests include:

* square matrices
* rectangular matrices
* large matrices
* small matrices
* odd dimensions
* different batch sizes
* different numbers of heads
* different `K` dimensions

The kernel uses FP32 accumulation, so numerical differences are expected when comparing against FP16 output.

---

# Benchmark Methodology

Timing uses CUDA events:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
```

Each configuration is warmed up before measurement.

The benchmark then runs multiple repetitions and averages the elapsed time.

TFLOPS are calculated as:

```text
FLOPs = 2 × B × H × M × N × K

TFLOPS = FLOPs / (time_ms × 10⁹)
```

This accounts for both the multiply and add operations in matrix multiplication.

---

# What This Project Was Intended to Teach

The main purpose of this kernel was to understand the relationship between:

```text
Tensor layout
      ↓
Memory access
      ↓
CTA mapping
      ↓
Cache reuse
      ↓
Tiling
      ↓
Warp execution
      ↓
Pipeline behavior
      ↓
Compiler code generation
      ↓
GPU performance
```

Rather than treating Triton as a black box, the optimization process was performed by changing one aspect at a time and measuring its effect across different shapes.

---

# Optimization Progression

The kernel evolved through the following stages:

```text
1. Basic tiled matmul
        ↓
2. FP32 accumulation
        ↓
3. Tile-size experiments
        ↓
4. num_warps experiments
        ↓
5. num_stages experiments
        ↓
6. Grouped program mapping
        ↓
7. Memory-layout/compiler hints
        ↓
8. Shape-dependent benchmarking
        ↓
9. Autotuning
```

The final implementation is therefore not just a matmul kernel, but an experiment in understanding how GPU kernel decisions interact.

---

# Hardware

Benchmarks were performed on:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
CUDA/Triton environment: local development environment
Datatype: FP16
Accumulator: FP32
```

Performance numbers are hardware- and environment-dependent and should not be interpreted as universal Triton or PyTorch benchmarks.

---

# Project Status

Completed:

* [x] Tiled Triton matmul
* [x] FP32 accumulation
* [x] Configurable tile sizes
* [x] Warp tuning
* [x] Pipeline-stage tuning
* [x] Grouped CTA mapping
* [x] `tl.multiple_of` experiments
* [x] `tl.max_contiguous` experiments
* [x] Shape-based benchmarking
* [x] Autotuning
* [x] Correctness testing
* [x] Performance comparison against PyTorch

The kernel is considered complete for this project.

Further optimization is intentionally not pursued indefinitely; the purpose is to understand the major performance mechanisms rather than maximize a single benchmark number.

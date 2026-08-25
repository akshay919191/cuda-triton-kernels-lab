# CUDA vs Triton Kernel Lab

## Overview

This section compares equivalent CUDA and Triton implementations of the same GPU kernels.

The goal is not simply to make one implementation faster than the other. The goal is to understand how different programming models expose:

* memory access patterns
* thread/program mapping
* tiling
* reductions
* register usage
* occupancy
* cache reuse
* kernel launch overhead
* compiler-generated optimizations

Each implementation is benchmarked against the PyTorch reference.

---

## Implementation Structure

```text
kernels/
├── cuda/
│   ├── <kernel>/
│   │   ├── kernel.cu
│   │   ├── binding.cpp
│   │   ├── setup.py
│   │   └── ...
│
└── triton/
    ├── <kernel>/
    │   ├── kernel.py
    │   └── ...
```

CUDA and Triton implementations intentionally solve the same mathematical operation using different execution models.

### CUDA

CUDA exposes the GPU execution model directly:

```text
grid
 └── blocks
      └── warps
           └── threads
```

The programmer explicitly controls:

* block dimensions
* thread indexing
* shared memory
* synchronization
* memory accesses
* warp-level behavior
* kernel launch configuration

### Triton

Triton exposes a higher-level programming model:

```text
grid
 └── program instances
      └── compiler-managed threads/warps
```

The programmer instead describes:

* tile shapes
* program IDs
* memory access patterns
* vectorized operations
* reductions
* tensor operations

The Triton compiler maps these operations onto GPU hardware.

---

# Matmul Case Study

The matmul kernel computes:

```text
C = A @ B
```

with:

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

The implementation uses tiled matrix multiplication.

For example:

```text
M = 512
N = 512
K = 512

128 x 128 output tile
```

The output is divided into:

```text
4 x 4 = 16 output tiles
```

Each program/CTA owns one output tile.

---

## Triton Program Mapping

The optimized Triton implementation uses grouped program ordering.

Instead of simply mapping:

```text
pid_m -> row tile
pid_n -> column tile
```

the program IDs are reordered into groups.

For example, with:

```text
GROUP = 2
```

the execution order becomes approximately:

```text
CTA 0 -> tile (0, 0)
CTA 1 -> tile (1, 0)

CTA 2 -> tile (0, 1)
CTA 3 -> tile (1, 1)

CTA 4 -> tile (2, 0)
CTA 5 -> tile (3, 0)

...
```

This changes the order in which tiles are processed without changing the mathematical result.

The purpose is cache reuse.

Neighboring programs work on tiles that reuse data from the same region of the input matrices, increasing the chance that useful data remains in cache.

---

# What To Analyze

For every optimized kernel, the important questions are not simply:

> "Is Triton faster?"

Instead, analyze the execution.

### 1. Which CTA/program owns this output tile?

For example:

```text
CTA/program -> C[0:128, 0:128]
```

### 2. Which elements does each lane/thread access?

Determine the mapping from:

```text
lane/thread
      ↓
matrix coordinates
      ↓
global memory address
```

### 3. Are the accesses contiguous?

Look at the physical layout of the tensor.

For a row-major matrix:

```text
A[i][k]
```

the `k` dimension is contiguous.

For:

```text
B[k][j]
```

the `j` dimension is contiguous.

Therefore, do not simply label one matrix "good" and the other "bad".

The correct question is:

> Which dimension is contiguous, and how are lanes mapped onto that dimension?

### 4. Which memory lines are touched?

Determine which neighboring lanes access the same cache lines.

### 5. What is reused between neighboring CTAs?

Ask whether neighboring programs reuse:

* A tiles
* B tiles
* cache lines
* shared memory data
* registers

This is the reasoning behind grouped/swizzled program ordering.

---

# Triton Matmul Optimization Experiments

The optimized matmul was developed incrementally.

## Baseline

Initial configuration:

```text
MBLOCK = 128
NBLOCK = 128
KBLOCK = 128
```

with simple:

```text
(pid_m, pid_n, pid_bh)
```

program mapping.

This provided a functional baseline but performed poorly for many larger matrices.

---

## K-Tile Reduction

The K tile was reduced:

```text
MBLOCK = 128
NBLOCK = 128
KBLOCK = 32
```

This improved performance significantly.

The kernel keeps the accumulator in FP32:

```python
acc = tl.zeros(
    (MBLOCK, NBLOCK),
    dtype=tl.float32,
)
```

while loading/storing FP16 data.

This preserves numerical stability without unnecessarily storing the accumulator in FP16.

---

## Explicit Warp Configuration

The kernel was tested with:

```text
num_warps = 4
num_warps = 8
```

The best configuration depends on the matrix shape.

There is no universally optimal warp count.

For example, one configuration may perform better for:

```text
1024 x 4096 x 1024
```

while another may perform better for:

```text
2048 x 2048 x 2048
```

This demonstrates why GPU kernel optimization must be evaluated across representative workloads rather than a single benchmark.

---

## Tile Size Experiments

The following configurations were tested:

```text
64  x 64  x 32
64  x 128 x 32
128 x 64  x 32
128 x 128 x 32
128 x 128 x 64
128 x 256 x 32
256 x 128 x 32
```

The larger tiles are not automatically faster.

Increasing tile size can increase:

* register usage
* number of values held by a program
* wasted computation on boundary tiles
* compiler resource requirements

Therefore tile size must be evaluated experimentally.

---

# Grouped Program Ordering

The kernel also tests grouped program ordering:

```text
GROUP = 1
GROUP = 2
GROUP = 4
GROUP = 8
```

The purpose is to change execution order so neighboring programs can reuse cache-resident data.

For example, a group can process multiple row tiles while keeping the same column tile active.

This is useful because matrix multiplication repeatedly accesses the same K regions.

However, grouping is workload-dependent.

For small matrices with very few output tiles, changing the grouping may produce little or no measurable improvement because there are not enough CTAs for cache locality to become a dominant factor.

For larger matrices, the difference can become more visible because the number of output tiles increases.

---

# Compiler Hints

The Triton implementation also experiments with:

```python
tl.multiple_of(...)
tl.max_contiguous(...)
```

These hints communicate layout/alignment information to the compiler.

For example:

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

The important observation is that these hints do not guarantee a speedup.

For some shapes:

```text
NONE
MULTIPLE_OF
MULTIPLE_OF + MAX_CONTIGUOUS
```

produce nearly identical timings.

For other shapes, the compiler can generate better code when the additional information is useful.

Therefore these hints are treated as compiler information rather than magic optimization switches.

---

# CUDA vs Triton: Bottleneck Analysis

The main bottleneck depends on the kernel family.

## Elementwise kernels

Examples:

```text
GELU
SiLU
```

Typical bottleneck:

```text
global memory bandwidth
```

The computation per element is relatively small compared with the amount of data moved.

Important questions:

```text
How many bytes are loaded?
How many bytes are stored?
How many FLOPs are performed?
Are accesses coalesced?
Can operations be fused?
```

---

## Reduction kernels

Examples:

```text
RMSNorm
LayerNorm
Softmax
```

Typical bottlenecks include:

```text
memory bandwidth
reduction efficiency
synchronization
register usage
```

The important question becomes:

> How efficiently can multiple threads cooperate to reduce a row/vector?

---

## Fused kernels

Examples:

```text
bias + GELU
residual + RMSNorm
```

The primary benefit is often reducing memory traffic.

Instead of:

```text
load
compute
store

load
compute
store
```

multiple operations can be performed while data is already resident in registers.

This can reduce intermediate global-memory reads and writes.

---

## Matmul

Matmul is primarily compute-intensive for sufficiently large matrices.

Important factors include:

```text
tensor-core utilization
tile size
K blocking
memory coalescing
L2 cache reuse
register pressure
warp count
occupancy
```

The key optimization question becomes:

> How much useful matrix computation can be performed per byte loaded from memory?

---

# Benchmarking Philosophy

Benchmarks are performed against:

```text
PyTorch
CUDA
Triton
```

using CUDA events rather than CPU wall-clock timing.

For each configuration:

1. Compile/warm up the kernel.
2. Synchronize the GPU.
3. Execute multiple repetitions.
4. Measure GPU execution time.
5. Compare numerical correctness.
6. Calculate TFLOPS.
7. Compare against the reference implementation.

Speedup is reported as:

```text
speedup = PyTorch time / kernel time
```

A value greater than:

```text
1.0x
```

means the custom kernel is faster.

A value below:

```text
1.0x
```

means the custom kernel is slower.

---

# Important Result

The objective of this project is not to claim:

> "Custom CUDA/Triton kernels always beat PyTorch."

PyTorch already uses highly optimized GPU libraries and compiler-generated kernels.

The purpose of the project is to understand why performance changes when modifying:

```text
tile size
warp count
program mapping
grouping
memory access patterns
compiler hints
fusion
reduction strategy
```

A slower kernel with a well-understood bottleneck is more valuable than an unexplained benchmark win.

---

# Limitations

Performance is hardware and workload dependent.

Results may change with:

* GPU architecture
* CUDA version
* Triton version
* PyTorch version
* tensor dtype
* matrix dimensions
* batch size
* sequence length
* memory layout
* compiler version

Therefore benchmark numbers should be treated as measurements for the tested environment rather than universal performance claims.

---

# Reproduction

Install the required environment and run the relevant benchmark scripts.

For CUDA kernels:

```bash
cd kernels/cuda/<kernel>
python setup.py build_ext --inplace
```

For Triton kernels:

```bash
python kernels/triton/<kernel>/benchmark.py
```

Each benchmark reports correctness and performance against PyTorch.

---

# What This Project Demonstrates

This project is intended to demonstrate practical GPU-kernel engineering rather than only API familiarity.

It covers:

```text
GPU memory hierarchy
thread/block/program mapping
warps
tiling
coalesced memory access
cache reuse
reductions
kernel fusion
register pressure
occupancy
CUDA extensions
Triton compilation
numerical correctness
GPU benchmarking
performance analysis
```

The central workflow is:

```text
Understand the operation
        ↓
Implement the kernel
        ↓
Verify correctness
        ↓
Benchmark
        ↓
Identify bottleneck
        ↓
Change one thing
        ↓
Benchmark again
        ↓
Explain the result
```

The goal is not to memorize optimizations.

The goal is to be able to look at a kernel and answer:

> Where does the data come from?

> Which threads/programs access it?

> Is the access coalesced?

> What is reused?

> Where is synchronization required?

> What resource is limiting performance?

> Why did this optimization help or fail?

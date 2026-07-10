#include "../common/common_helper.cuh"
#include "private_helper.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <cmath>
#define PADDING 8
#define FLOAT4(x)  (*reinterpret_cast<float4*>(&(x)))
#define CFLOAT4(x) (*reinterpret_cast<const float4*>(&(x)))


template<int Br>
__global__ void Silufwd_kernel(
    const __half* __restrict__ input,
          __half* __restrict__ output,
    int seqlen,
    int headdim,
    int numhead
)
{
    int tid = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base =
        (long long)batchid * numhead * seqlen * headdim +
        (long long)headid  * seqlen * headdim;

    const __half* INptr = input + base;
    __half* outptr = output + base;

    extern __shared__ char smem[];
    char* ptr = smem;

    const int rowstride = headdim + PADDING;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * rowstride * sizeof(__half);

    cpasynccopy<Br>(INptr, smemA, rowstride, tileid, seqlen, headdim);

    // Keep these only if cpasynccopy actually uses cp.async.
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");

    __syncthreads();

    doSILU<Br>(smemA, rowstride, headdim);

    __syncthreads();

    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int smemidx = r * rowstride + c;

        outptr[globalRow * headdim + c] = smemA[smemidx];
    }
}

template<int Br>
__global__ void Silubwd_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ dl_dy,
          __half* __restrict__ output,
    int seqlen , int headdim , int numhead
)
{
    int tid = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;


    const long long base = (long long)batchid * numhead * seqlen * headdim + 
                           (long long)headid * headdim * seqlen;

    const __half* INptr  = input + base;
    const __half* dy     = dl_dy + base;
    __half* outptr = output + base;
    

    extern __shared__ char smem[];
    char* ptr = smem;

    const int rowstride = headdim + PADDING;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * rowstride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smenB = reinterpret_cast<__half*>(ptr);
    ptr += Br * rowstride * sizeof(__half);

    cpasynccopy<Br>(INptr , smemA , rowstride , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");

    cpasynccopy<Br>(dy , smenB , rowstride , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");

    __syncthreads();
    doSILUbck<Br>(smemA, smenB, rowstride, headdim);
    __syncthreads();

    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int smemidx = r * rowstride + c;
        int globalRow = Br * tileid + r;

        if (globalRow >= seqlen) continue;

        outptr[globalRow * headdim + c] =
            smemA[smemidx];
    }
}

static inline size_t align32_bytes(size_t x) {
    return (x + 31) & ~size_t(31);
}

template<int Br>
static inline size_t silu_fwd_smem_bytes(int D) {
    size_t bytes = 0;

    const int rowStride = D + PADDING;

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // smemA

    bytes += 128;
    return bytes;
}

template<int Br>
static inline size_t silu_bwd_smem_bytes(int D) {
    size_t bytes = 0;

    const int rowStride = D + PADDING;

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // smemA / input

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // upstream dy

    bytes += 128;
    return bytes;
}


template<int Br>
std::vector<torch::Tensor> silu_forward_launch(torch::Tensor x) {
    CHECK_INPUT(x);

    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    auto y = torch::empty_like(x);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = silu_fwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            Silufwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    Silufwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N,  // seqlen
        D,  // headdim
        H   // numhead
    );

    CUDA_CHECK(cudaGetLastError());

    return {y};
}


template<int Br>
std::vector<torch::Tensor> silu_backward_launch(
    torch::Tensor dy,
    torch::Tensor x
) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);

    TORCH_CHECK(dy.scalar_type() == torch::kFloat16, "dy must be float16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");

    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    auto dx = torch::empty_like(x);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = silu_bwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            Silubwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    Silubwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dy.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(dx.data_ptr<at::Half>()),
        N,  // seqlen
        D,  // headdim
        H   // numhead
    );

    CUDA_CHECK(cudaGetLastError());

    return {dx};
}


std::vector<torch::Tensor> silu_forward_cuda(torch::Tensor x) {
    return silu_forward_launch<16>(x);
}


std::vector<torch::Tensor> silu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
) {
    return silu_backward_launch<16>(dy, x);
}
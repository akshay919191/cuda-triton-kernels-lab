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
#include <vector>

#define PADDING 8
#define FLOAT4(x)  (*reinterpret_cast<float4*>(&(x)))
#define CFLOAT4(x) (*reinterpret_cast<const float4*>(&(x)))

template<int Br>
__global__ void fusedgelufwd_kernel(
    const __half* __restrict__ input,
    const float*  __restrict__ bias,
          __half* __restrict__ output ,
    const int seqlen , const int headdim , const int numhead
)
{
    int tid    = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                                (long long)headid * headdim * seqlen;

    const __half* INptr  = input + base;
          __half* outptr = output + base;
    const float*  biasp  = bias;              // fixed: was float* (dropped const)
    const int rowStride = headdim + PADDING;

    extern __shared__ char smem[];

    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    float* biass = reinterpret_cast<float*>(ptr);
    ptr += headdim * sizeof(float);

    for(int i = tid ; i < headdim ; i += blockDim.x) biass[i] = biasp[i];  // fixed: was `= biasp;`

    __syncthreads();

    cpasynccopy<Br>(INptr , smemA , headdim + PADDING , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");

    __syncthreads();

    for(int i = tid ; i < Br * headdim ; i += blockDim.x){
        int r = i / headdim;
        int c = i % headdim;

        int smemidx = r * rowStride + c;   // fixed: was `rowstride`
        smemA[smemidx] = __float2half(__half2float(smemA[smemidx]) + biass[c]);
    }

    __syncthreads();

    performGelu<Br>(smemA , headdim + PADDING , headdim);
    __syncthreads();

    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int row = i / headdim;
        int col = i % headdim;

        int globalRow = tileid * Br + row;
        if (globalRow >= seqlen) continue;

        outptr[globalRow * headdim + col] =
            smemA[row * rowStride + col];
    }
}

template<int Br>
__global__ void fusedgelubwd_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ dl_dy,
    const float*  __restrict__ bias,
          __half* __restrict__ output ,
    const int seqlen , const int headdim , const int numhead
)
{
    int tid = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const int rowStride = headdim + PADDING;

    const long long base    = (long long)batchid * numhead * seqlen * headdim +
                        (long long)headid  * seqlen * headdim;

    const __half* INptr  = input  + base;
    const __half* prevv  = dl_dy  + base;
    const float*  biasp  = bias;              // fixed: was float* (dropped const)
    
          __half* outptr = output + base;

    extern __shared__ char smem[];
    
    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* upstream = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    float* biass = reinterpret_cast<float*>(ptr);
    ptr += headdim * sizeof(float);

    for(int i = tid ; i < headdim ; i += blockDim.x) biass[i] = biasp[i];  // fixed: was `= biasp;`

    __syncthreads();

    cpasynccopy<Br>(prevv , upstream , headdim + PADDING , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");

    cpasynccopy<Br>(INptr , smemA , headdim + PADDING , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n");

    __syncthreads();

    for(int i = tid ; i < Br * headdim ; i += blockDim.x){
        int r = i / headdim;
        int c = i % headdim;

        int smemidx = r * rowStride + c;   // fixed: was `rowstride`
        smemA[smemidx] = __float2half(__half2float(smemA[smemidx]) + biass[c]);
    }

    __syncthreads();

    performGelubck<Br>(smemA , upstream , headdim + PADDING , headdim);
    __syncthreads();

    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int row = i / headdim;
        int col = i % headdim;

        int globalRow = tileid * Br + row;
        if (globalRow >= seqlen) continue;

        outptr[globalRow * headdim + col] =
            smemA[row * rowStride + col];
    }
}

static inline size_t align32_bytes(size_t x) {
    return (x + 31) & ~size_t(31);
}

template<int Br>
static inline size_t gelu_fwd_smem_bytes(int D) {
    size_t bytes = 0;

    bytes = align32_bytes(bytes);
    bytes += Br * (D + PADDING) * sizeof(__half);

    bytes = align32_bytes(bytes);
    bytes += D * sizeof(float); 

    bytes += 128;
    return bytes;
}

template<int Br>
static inline size_t gelu_bwd_smem_bytes(int D) {
    size_t bytes = 0;

    bytes = align32_bytes(bytes);
    bytes += Br * (D + PADDING) * sizeof(__half);  // smemA

    bytes = align32_bytes(bytes);
    bytes += Br * (D + PADDING) * sizeof(__half);  // upstream

    bytes = align32_bytes(bytes);
    bytes += D * sizeof(float);  // bias staging

    bytes += 128;
    return bytes;
}

template<int Br>
std::vector<torch::Tensor> gelu_forward_launch(torch::Tensor x , torch::Tensor b) {
    CHECK_INPUT(x);
    CHECK_INPUT(b);   // added: bias was never validated

    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(b.dim() == 1 && b.size(0) == x.size(3), "bias must be [D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    auto y = torch::empty_like(x);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = gelu_fwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            fusedgelufwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    fusedgelufwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        b.data_ptr<float>(),                       // fixed: was reinterpret_cast<const float>(b.data_ptr)
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N,
        D,
        H
    );

    CUDA_CHECK(cudaGetLastError());

    return {y};
}

template<int Br>
std::vector<torch::Tensor> gelu_backward_launch(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor b
) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    CHECK_INPUT(b);   // added

    TORCH_CHECK(dy.scalar_type() == torch::kFloat16, "dy must be float16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(b.dim() == 1 && b.size(0) == x.size(3), "bias must be [D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    auto dx = torch::empty_like(x);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = gelu_bwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            fusedgelubwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    fusedgelubwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dy.data_ptr<at::Half>()),
        b.data_ptr<float>(),                       // fixed
        reinterpret_cast<__half*>(dx.data_ptr<at::Half>()),
        N,
        D,
        H
    );

    CUDA_CHECK(cudaGetLastError());

    return {dx};
}

std::vector<torch::Tensor> gelu_forward_cuda(torch::Tensor x , torch::Tensor b) {
    return gelu_forward_launch<16>(x , b);
}

std::vector<torch::Tensor> gelu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor b
) {
    return gelu_backward_launch<16>(dy, x , b);
}
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

template<int Br>
__global__ void softmaxfwd_kernel(
    const __half* __restrict__ input,
          __half* __restrict__ output,
    const int headdim , const int seqlen , const int numhead
)
{
    int tid = threadIdx.x;
    const int rowstride = PADDING + headdim;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                           (long long)headid  * headdim * seqlen;

    const __half* INptr  = input + base;
          __half* outptr = output + base;
    
    extern __shared__ char smem[];

    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * rowstride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* res1 = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    float* res2 = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    cpasynccopy<Br>(INptr , smemA , rowstride , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    dosoftmax<Br>(smemA , res1 , res2 , headdim , seqlen , rowstride);
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
__global__ void softmaxbwd_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ dl_dy,
          __half* __restrict__ output,
    const int headdim , const int seqlen , const int numhead
)
{
    int tid = threadIdx.x;
    const int rowstride = PADDING + headdim;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                           (long long)headid  * headdim * seqlen;

    const __half* INptr  = input + base;
    const __half* dy     = dl_dy + base;
          __half* outptr = output + base;
    
    extern __shared__ char smem[];

    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);
    ptr += Br * rowstride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemB = reinterpret_cast<__half*>(ptr);
    ptr += Br * rowstride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* buff = reinterpret_cast<__half*>(ptr);
    ptr += Br * rowstride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* res1 = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    float* res2 = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    cpasynccopy<Br>(INptr , smemA , rowstride , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");

    cpasynccopy<Br>(dy    , smemB , rowstride , tileid , seqlen , headdim);
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    /// smemA holds the softmax
    dosoftmax<Br>(smemA , res1 , res2 , headdim , seqlen , rowstride);
    __syncthreads();

    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        buff[r * rowstride + c] = __float2half(__half2float(smemA[r * rowstride + c]) * __half2float(smemB[r * rowstride + c]));
    }
    __syncthreads();

    multiWarpReductionSUM_softmax_plain(buff, res1, headdim, rowstride, Br);
    __syncthreads();

    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int smemidx = r * rowstride + c;
        int globalidx = globalRow * headdim + c;

        float y  = __half2float(smemA[smemidx]);
        float dy = __half2float(smemB[smemidx]);
        float s  = res1[r];

        outptr[globalidx] = __float2half(y * (dy - s));
    }
}

static inline size_t align32_bytes(size_t x) {
    return (x + 31) & ~size_t(31);
}

template<int Br>
static inline size_t softmax_fwd_smem_bytes(int D) {
    size_t bytes = 0;

    const int rowStride = D + PADDING;

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // smemA

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // res1

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // res2

    bytes += 128;
    return bytes;
}

template<int Br>
static inline size_t softmax_bwd_smem_bytes(int D) {
    size_t bytes = 0;

    const int rowStride = D + PADDING;

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half); 

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // res1

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // res2

    bytes += 128;
    return bytes;
}


template<int Br>
std::vector<torch::Tensor> softmax_forward_launch(torch::Tensor x) {
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

    size_t smem = softmax_fwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            softmaxfwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    softmaxfwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        D,  // headdim
        N,  // seqlen
        H   // numhead
    );

    CUDA_CHECK(cudaGetLastError());

    return {y};
}


template<int Br>
std::vector<torch::Tensor> softmax_backward_launch(
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

    size_t smem = softmax_bwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            softmaxbwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    softmaxbwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dy.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(dx.data_ptr<at::Half>()),
        D,  // headdim
        N,  // seqlen
        H   // numhead
    );

    CUDA_CHECK(cudaGetLastError());

    return {dx};
}


std::vector<torch::Tensor> softmax_forward_cuda(torch::Tensor x) {
    return softmax_forward_launch<16>(x);
}


std::vector<torch::Tensor> softmax_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
) {
    return softmax_backward_launch<16>(dy, x);
}
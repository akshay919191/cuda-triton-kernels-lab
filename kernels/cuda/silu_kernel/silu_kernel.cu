#include "../common/common_helper.cuh"
#include "private_helper.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

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


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

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

    const long long base = (long long)batchid * numhead * seqlen * headdim +
                           (long long)headid  * seqlen * headdim;

    const __half* INptr = input + base;
    __half* outptr = output + base;

    // Process elements directly from global memory
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        long long idx = (long long)globalRow * headdim + c;

        // Load
        float val = __half2float(INptr[idx]);

        // SiLU: x * sigmoid(x) -> x / (1 + exp(-x))
        float silu_val = val / (1.0f + expf(-val));

        // Store
        outptr[idx] = __float2half(silu_val);
    }
}

template<int Br>
__global__ void Silubwd_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ dl_dy,
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

    const long long base = (long long)batchid * numhead * seqlen * headdim +
                           (long long)headid  * seqlen * headdim;

    const __half* INptr = input + base;
    const __half* dy    = dl_dy + base;
    __half* outptr      = output + base;

    // Process elements directly from global memory
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        long long idx = (long long)globalRow * headdim + c;

        // Load input and upstream gradient
        float val  = __half2float(INptr[idx]);
        float grad = __half2float(dy[idx]);

        // SiLU derivative:
        // d/dx [ x * sigmoid(x) ] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        // = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float sig = 1.0f / (1.0f + expf(-val));
        float d_silu = sig * (1.0f + val * (1.0f - sig));
        
        float grad_val = grad * d_silu;

        // Store
        outptr[idx] = __float2half(grad_val);
    }
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

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    Silufwd_kernel<Br><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N,
        D,
        H
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

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    Silubwd_kernel<Br><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dy.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(dx.data_ptr<at::Half>()),
        N,
        D,
        H
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
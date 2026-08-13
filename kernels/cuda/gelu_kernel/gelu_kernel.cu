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

#define RSQRT_2 0.70710678118654752440f

template<int Br>
__global__ void gelufwd_kernel(
    const __half* __restrict__ input,
          __half* __restrict__ output ,
    const int seqlen , const int headdim , const int numhead
)
{
    int tid = threadIdx.x;
    
    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                                (long long)headid * headdim * seqlen;

    const __half2* INptr2  = reinterpret_cast<const __half2*>(input + base);
    __half2* outptr2       = reinterpret_cast<__half2*>(output + base);

    const int headdim2 = headdim >> 1;
    const int total_elements2 = Br * headdim2;

    for (int i = tid; i < total_elements2; i += blockDim.x)
    {
        int row = i / headdim2;
        int col = i % headdim2;

        int globalRow = tileid * Br + row;
        if (globalRow >= seqlen) continue;

        long long idx = (long long)globalRow * headdim2 + col;
        
        __half2 val_h2 = INptr2[idx];
        float2 val_f2 = __half22float2(val_h2);

        float out_x = 0.5f * val_f2.x * (1.0f + erff(val_f2.x * RSQRT_2));
        float out_y = 0.5f * val_f2.y * (1.0f + erff(val_f2.y * RSQRT_2));

        outptr2[idx] = __floats2half2_rn(out_x, out_y);
    }

    if (headdim & 1) {
        int col_odd = headdim - 1;
        for (int row = tid; row < Br; row += blockDim.x) {
            int globalRow = tileid * Br + row;
            if (globalRow >= seqlen) continue;
            
            long long idx = (long long)globalRow * headdim + col_odd;
            float val = __half2float((input + base)[idx]);
            float gelu_val = 0.5f * val * (1.0f + erff(val * RSQRT_2));
            (output + base)[idx] = __float2half(gelu_val);
        }
    }
}

template<int Br>
__global__ void gelubwd_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ dl_dy,
          __half* __restrict__ output ,
    const int seqlen , const int headdim , const int numhead
)
{
    int tid = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * seqlen * headdim +
                        (long long)headid  * seqlen * headdim;

    const __half2* INptr2  = reinterpret_cast<const __half2*>(input + base);
    const __half2* prevv2 = reinterpret_cast<const __half2*>(dl_dy + base);
    __half2* outptr2      = reinterpret_cast<__half2*>(output + base);

    const int headdim2 = headdim >> 1;
    const int total_elements2 = Br * headdim2;

    for (int i = tid; i < total_elements2; i += blockDim.x)
    {
        int row = i / headdim2;
        int col = i % headdim2;

        int globalRow = tileid * Br + row;
        if (globalRow >= seqlen) continue;

        long long idx = (long long)globalRow * headdim2 + col;

        __half2 val_h2  = INptr2[idx];
        __half2 grad_h2 = prevv2[idx];
        
        float2 val_f2  = __half22float2(val_h2);
        float2 grad_f2 = __half22float2(grad_h2);

        float cdf_x = 0.5f * (1.0f + erff(val_f2.x * RSQRT_2));
        float pdf_x = 0.39894228040143267793f * __expf(-0.5f * val_f2.x * val_f2.x);
        float out_x = grad_f2.x * __fmaf_rn(val_f2.x, pdf_x, cdf_x);

        float cdf_y = 0.5f * (1.0f + erff(val_f2.y * RSQRT_2));
        float pdf_y = 0.39894228040143267793f * __expf(-0.5f * val_f2.y * val_f2.y);
        float out_y = grad_f2.y * __fmaf_rn(val_f2.y, pdf_y, cdf_y);

        // Vectorized store
        outptr2[idx] = __floats2half2_rn(out_x, out_y);
    }

    if (headdim & 1) {
        int col_odd = headdim - 1;
        for (int row = tid; row < Br; row += blockDim.x) {
            int globalRow = tileid * Br + row;
            if (globalRow >= seqlen) continue;
            
            long long idx = (long long)globalRow * headdim + col_odd;
            float val  = __half2float((input + base)[idx]);
            float grad = __half2float((dl_dy + base)[idx]);

            float cdf = 0.5f * (1.0f + erff(val * RSQRT_2));
            float pdf = 0.39894228040143267793f * __expf(-0.5f * val * val);
            float grad_val = grad * __fmaf_rn(val, pdf, cdf);

            (output + base)[idx] = __float2half(grad_val);
        }
    }
}

template<int Br>
std::vector<torch::Tensor> gelu_forward_launch(torch::Tensor x) {
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

    gelufwd_kernel<Br><<<grid, block, 0, stream>>>(
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
std::vector<torch::Tensor> gelu_backward_launch(
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

    gelubwd_kernel<Br><<<grid, block, 0, stream>>>(
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


std::vector<torch::Tensor> gelu_forward_cuda(torch::Tensor x) {
    return gelu_forward_launch<16>(x);
}


std::vector<torch::Tensor> gelu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x
) {
    return gelu_backward_launch<16>(dy, x);
}
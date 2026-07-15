#include "../common/common_helper.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <vector>

#define RSQRT_2 0.70710678118654752440f

template<int Br>
__global__ void fusedgelufwd_kernel(
    const __half* __restrict__ input,
    const float*  __restrict__ bias,
          __half* __restrict__ output,
    const int seqlen,
    const int headdim,
    const int numhead
)
{
    int tid = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                           (long long)headid * headdim * seqlen;

    const __half* INptr  = input + base;
          __half* outptr = output + base;

    // Process elements directly from global memory
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        long long idx = (long long)globalRow * headdim + c;

        float val = __half2float(INptr[idx]) + bias[c];

        float gelu_val = 0.5f * val * (1.0f + erff(val * RSQRT_2));

        // 3. Store output
        outptr[idx] = __float2half(gelu_val);
    }
}

template<int Br>
__global__ void fusedgelubwd_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ dl_dy,
    const float*  __restrict__ bias,
          __half* __restrict__ output,
          float*  __restrict__ dl_bias,
    const int seqlen,
    const int headdim,
    const int numhead
)
{
    int tid = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base = (long long)batchid * numhead * seqlen * headdim +
                           (long long)headid  * seqlen * headdim;

    const __half* INptr  = input  + base;
    const __half* prevv  = dl_dy  + base;
          __half* outptr = output + base;

    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        long long idx = (long long)globalRow * headdim + c;

        float val = __half2float(INptr[idx]) + bias[c];

        // 2. Load upstream gradient
        float grad = __half2float(prevv[idx]);

        float cdf = 0.5f * (1.0f + erff(val * RSQRT_2));
        float pdf = 0.39894228040143267793f * expf(-0.5f * val * val); 
        float d_gelu = grad * (cdf + val * pdf);

        atomicAdd(&dl_bias[c], d_gelu);

        outptr[idx] = __float2half(d_gelu);
    }
}


template<int Br>
std::vector<torch::Tensor> fused_bias_gelu_forward_launch(
    torch::Tensor x,
    torch::Tensor bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);

    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(bias.dim() == 1, "bias must be [D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    TORCH_CHECK(bias.size(0) == D, "bias.size(0) must match D");

    auto y = torch::empty_like(x);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fusedgelufwd_kernel<Br><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        bias.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N,
        D,
        H
    );

    CUDA_CHECK(cudaGetLastError());

    return {y};
}


template<int Br>
std::vector<torch::Tensor> fused_bias_gelu_backward_launch(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor bias
) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    CHECK_INPUT(bias);

    TORCH_CHECK(dy.scalar_type() == torch::kFloat16, "dy must be float16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");
    TORCH_CHECK(bias.dim() == 1, "bias must be [D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    TORCH_CHECK(bias.size(0) == D, "bias.size(0) must match D");

    auto dx = torch::empty_like(x);

    auto opts_f32 = torch::TensorOptions()
        .device(x.device())
        .dtype(torch::kFloat32);

    auto dbias = torch::zeros({D}, opts_f32);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fusedgelubwd_kernel<Br><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dy.data_ptr<at::Half>()),
        bias.data_ptr<float>(),
        reinterpret_cast<__half*>(dx.data_ptr<at::Half>()),
        dbias.data_ptr<float>(),
        N,
        D,
        H
    );

    CUDA_CHECK(cudaGetLastError());

    return {dx, dbias};
}


std::vector<torch::Tensor> fused_bias_gelu_forward_cuda(
    torch::Tensor x,
    torch::Tensor bias
) {
    return fused_bias_gelu_forward_launch<16>(x, bias);
}


std::vector<torch::Tensor> fused_bias_gelu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor bias
) {
    return fused_bias_gelu_backward_launch<16>(dy, x, bias);
}


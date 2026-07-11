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
__global__ void layernormfwd_kernel(
    const __half* __restrict__ input,
    const float*  __restrict__ gamma,
    const float*  __restrict__ betaa,
          __half* __restrict__ output,
    float eps , const int headdim , int seqlen , int numhead
){
    int tid = threadIdx.x;

    const int rowstride = headdim + PADDING;
    const int batchid   = blockIdx.x;
    const int headid    = blockIdx.y;
    const int tileid    = blockIdx.z;

    long long base = (long long)batchid * numhead * headdim * seqlen + 
                     (long long)headid  * headdim * seqlen;
    
    const __half* INptr = input + base;
          __half* outptr= output + base;

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

    float* gam = reinterpret_cast<float*>(ptr);
    ptr += headdim * sizeof(float);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* beta = reinterpret_cast<float*>(ptr);
    ptr += headdim * sizeof(float);

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

    for(int i = tid ; i < headdim ; i += blockDim.x)
    {
        gam[i] = gamma[i]; beta[i] = betaa[i];
    }
    __syncthreads();

    multiWarpReductionMEAN(smemA , res1 , headdim , rowstride , Br);
    __syncthreads();

    multiWarpReductionSIGMA2(smemA , res1 , res2 , headdim , rowstride , Br);
    __syncthreads();

    //// now we have sigma square and U and x , gamma and beta
    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int global_row = (tileid * Br + r);
        if (global_row >= seqlen) continue;

        int smemidx = r * rowstride + c;
        float val = gam[c] * (__half2float(smemA[smemidx]) - res1[r]) / (sqrtf(res2[r] + eps)) + beta[c];
        outptr[global_row * headdim + c] = val;

    }
}



template<int Br>
__global__ void layernormbwd_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ dl_dy,
    const float*  __restrict__ gamma,
    const float*  __restrict__ betaa,   // unused in backward
          __half* __restrict__ output,
          float*  __restrict__ dl_gama,
          float*  __restrict__ dl_beta,
    float eps,
    const int headdim,
    int seqlen,
    int numhead
)
{
    int tid = threadIdx.x;

    const int rowstride = headdim + PADDING;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const long long base =
        (long long)batchid * numhead * seqlen * headdim +
        (long long)headid  * seqlen * headdim;

    const __half* INptr  = input + base;
    const __half* dy     = dl_dy + base;
          __half* outptr = output + base;

    float* betaout = dl_beta;   // [D]
    float* gamaout = dl_gama;   // [D]

    extern __shared__ char smem[];
    char* ptr = smem;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* smemA = reinterpret_cast<__half*>(ptr);   // x, later xhat
    ptr += Br * rowstride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* dysm = reinterpret_cast<__half*>(ptr);    // dy, later delta*xhat
    ptr += Br * rowstride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* buff = reinterpret_cast<__half*>(ptr);    // delta = dy * gamma
    ptr += Br * rowstride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* gam = reinterpret_cast<float*>(ptr);        // gamma
    ptr += headdim * sizeof(float);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* res1 = reinterpret_cast<float*>(ptr);       // mean(x), later mean(delta)
    ptr += Br * sizeof(float);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* res2 = reinterpret_cast<float*>(ptr);       // var(x)
    ptr += Br * sizeof(float);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* res3 = reinterpret_cast<float*>(ptr);       // mean(delta * xhat)
    ptr += Br * sizeof(float);

    // load x
    cpasynccopy<Br>(INptr, smemA, rowstride, tileid, seqlen, headdim);
    asm volatile("cp.async.commit_group;\n");

    // load dy
    cpasynccopy<Br>(dy, dysm, rowstride, tileid, seqlen, headdim);
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // load gamma
    for (int i = tid; i < headdim; i += blockDim.x)
    {
        gam[i] = gamma[i];
    }
    __syncthreads();

    // res1 = mean(x)
    multiWarpReductionMEAN(smemA, res1, headdim, rowstride, Br);
    __syncthreads();

    // res2 = variance(x)
    multiWarpReductionSIGMA2(smemA, res1, res2, headdim, rowstride, Br);
    __syncthreads();

    // smemA = xhat
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int global_row = tileid * Br + r;
        int smemidx = r * rowstride + c;

        if (global_row >= seqlen) {
            smemA[smemidx] = __float2half(0.0f);
            continue;
        }

        float x_val = __half2float(smemA[smemidx]);
        float mean = res1[r];
        float inv_std = rsqrtf(res2[r] + eps);

        float xhat = (x_val - mean) * inv_std;
        smemA[smemidx] = __float2half(xhat);
    }
    __syncthreads();

    // atomic dgamma/dbeta + buff = delta = dy * gamma
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int global_row = tileid * Br + r;
        int smemidx = r * rowstride + c;

        if (global_row >= seqlen) {
            buff[smemidx] = __float2half(0.0f);
            dysm[smemidx] = __float2half(0.0f);
            continue;
        }

        float dy_val = __half2float(dysm[smemidx]);
        float xhat   = __half2float(smemA[smemidx]);

        atomicAdd(&betaout[c], dy_val);
        atomicAdd(&gamaout[c], dy_val * xhat);

        float delta = dy_val * gam[c];
        buff[smemidx] = __float2half(delta);
    }
    __syncthreads();

    // res1 = mean(delta)
    multiWarpReductionMEAN(buff, res1, headdim, rowstride, Br);
    __syncthreads();

    // dysm = delta * xhat
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int global_row = tileid * Br + r;
        int smemidx = r * rowstride + c;

        if (global_row >= seqlen) {
            dysm[smemidx] = __float2half(0.0f);
            continue;
        }

        float delta = __half2float(buff[smemidx]);
        float xhat  = __half2float(smemA[smemidx]);

        dysm[smemidx] = __float2half(delta * xhat);
    }
    __syncthreads();

    // res3 = mean(delta * xhat)
    multiWarpReductionMEAN(dysm, res3, headdim, rowstride, Br);
    __syncthreads();

    // dx = inv_std * (delta - mean(delta) - xhat * mean(delta*xhat))
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int global_row = tileid * Br + r;
        if (global_row >= seqlen) continue;

        int smemidx = r * rowstride + c;

        float inv_std = rsqrtf(res2[r] + eps);

        float delta = __half2float(buff[smemidx]);
        float xhat  = __half2float(smemA[smemidx]);

        float mean_delta      = res1[r];
        float mean_delta_xhat = res3[r];

        float dx = inv_std * (
            delta
            - mean_delta
            - xhat * mean_delta_xhat
        );

        outptr[global_row * headdim + c] = __float2half(dx);
    }
}

static inline size_t align32_bytes(size_t x) {
    return (x + 31) & ~size_t(31);
}


template<int Br>
static inline size_t layernorm_fwd_smem_bytes(int D) {
    size_t bytes = 0;

    const int rowStride = D + PADDING;

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // smemA

    bytes = align32_bytes(bytes);
    bytes += D * sizeof(float);                // gamma

    bytes = align32_bytes(bytes);
    bytes += D * sizeof(float);                // beta

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // res1 mean

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // res2 variance

    bytes += 128;
    return bytes;
}


template<int Br>
static inline size_t layernorm_bwd_smem_bytes(int D) {
    size_t bytes = 0;

    const int rowStride = D + PADDING;

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // smemA

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // dysm

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // buff

    bytes = align32_bytes(bytes);
    bytes += D * sizeof(float);                // gamma

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // res1

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // res2

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // res3

    bytes += 128;
    return bytes;
}


template<int Br>
std::vector<torch::Tensor> layernorm_forward_launch(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);

    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "beta must be float32");

    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [D]");
    TORCH_CHECK(beta.dim() == 1, "beta must be [D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    TORCH_CHECK(gamma.size(0) == D, "gamma shape must match D");
    TORCH_CHECK(beta.size(0) == D, "beta shape must match D");

    auto y = torch::empty_like(x);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = layernorm_fwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            layernormfwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    layernormfwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        static_cast<float>(eps),
        D,  // headdim
        N,  // seqlen
        H   // numhead
    );

    CUDA_CHECK(cudaGetLastError());

    return {y};
}


template<int Br>
std::vector<torch::Tensor> layernorm_backward_launch(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);

    TORCH_CHECK(dy.scalar_type() == torch::kFloat16, "dy must be float16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "beta must be float32");

    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    TORCH_CHECK(gamma.size(0) == D, "gamma shape must match D");
    TORCH_CHECK(beta.size(0) == D, "beta shape must match D");

    auto dx = torch::empty_like(x);

    auto opts_f32 = torch::TensorOptions()
        .device(x.device())
        .dtype(torch::kFloat32);

    auto dgamma = torch::zeros({D}, opts_f32);
    auto dbeta  = torch::zeros({D}, opts_f32);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = layernorm_bwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            layernormbwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    layernormbwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dy.data_ptr<at::Half>()),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        reinterpret_cast<__half*>(dx.data_ptr<at::Half>()),
        dgamma.data_ptr<float>(),
        dbeta.data_ptr<float>(),
        static_cast<float>(eps),
        D,  // headdim
        N,  // seqlen
        H   // numhead
    );

    CUDA_CHECK(cudaGetLastError());

    return {dx, dgamma, dbeta};
}


std::vector<torch::Tensor> layernorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
) {
    return layernorm_forward_launch<16>(x, gamma, beta, eps);
}


std::vector<torch::Tensor> layernorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
) {
    return layernorm_backward_launch<16>(dy, x, gamma, beta, eps);
}
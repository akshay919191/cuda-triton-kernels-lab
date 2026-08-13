#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>



/// philox is LLM generated---
struct Philox {
    uint4 counter;
    uint2 key;
    uint4 state;
    int state_idx;

    __device__ __forceinline__ Philox(uint64_t seed, uint64_t offset, uint32_t tid) {
        key.x = (uint32_t)seed;
        key.y = (uint32_t)(seed >> 32);
        
        counter.x = (uint32_t)offset;
        counter.y = (uint32_t)(offset >> 32);
        counter.z = tid;    
        counter.w = 0;
        
        state_idx = 4;
    }

    __device__ __forceinline__ uint4 single_round(uint4 ctr, uint2 k) {
        uint64_t p0 = (uint64_t)ctr.x * 0xD2511F53ULL;
        uint64_t p1 = (uint64_t)ctr.z * 0xCD9E8D57ULL;
        
        uint32_t hi0 = (uint32_t)(p0 >> 32);
        uint32_t lo0 = (uint32_t)p0;
        uint32_t hi1 = (uint32_t)(p1 >> 32);
        uint32_t lo1 = (uint32_t)p1;

        uint4 ret;
        ret.x = hi1 ^ ctr.y ^ k.x;
        ret.y = lo1;
        ret.z = hi0 ^ ctr.w ^ k.y;
        ret.w = lo0;
        return ret;
    }

    __device__ __forceinline__ uint4 next_uint4() {
        uint4 ctr = counter;
        uint2 k = key;
        
        // 10 rounds (Philox4x32-10 standard)
        #pragma unroll
        for (int i = 0; i < 10; ++i) {
            ctr = single_round(ctr, k);
            k.x += 0x9E3779B9UL; 
            k.y += 0xBB67AE85UL;
        }
        
        if (++counter.x == 0) {
            if (++counter.y == 0) {
                if (++counter.z == 0) {
                    ++counter.w;
                }
            }
        }
        return ctr;
    }

    __device__ __forceinline__ float4 rng4() {
        uint4 bits = next_uint4();
        float4 out;
        out.x = (float)(bits.x >> 8) * (1.0f / 16777216.0f);
        out.y = (float)(bits.y >> 8) * (1.0f / 16777216.0f);
        out.z = (float)(bits.z >> 8) * (1.0f / 16777216.0f);
        out.w = (float)(bits.w >> 8) * (1.0f / 16777216.0f);
        return out;
    }
};

template <typename T>
__global__ void dropout_kernel_optimized(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t N,
    float p,
    uint64_t seed,
    uint64_t offset
) {
    int64_t tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int64_t stride =
        (int64_t)blockDim.x * gridDim.x;

    const float scale =
        1.0f / (1.0f - p);

    if constexpr (std::is_same_v<T, float>) {

        const int64_t N_vec = N >> 2;
        int64_t idx = tid;

        Philox rng(
            seed,
            offset,
            (uint32_t)tid
        );

        while (idx < N_vec) {

            float4 x_vec =
                reinterpret_cast<const float4*>(x)[idx];

            float4 y_vec;

            float4 r = rng.rng4();

            y_vec.x =
                (r.x >= p)
                ? x_vec.x * scale
                : 0.0f;

            y_vec.y =
                (r.y >= p)
                ? x_vec.y * scale
                : 0.0f;

            y_vec.z =
                (r.z >= p)
                ? x_vec.z * scale
                : 0.0f;

            y_vec.w =
                (r.w >= p)
                ? x_vec.w * scale
                : 0.0f;

            reinterpret_cast<float4*>(y)[idx] =
                y_vec;

            idx += stride;
        }

        // Remaining elements
        int64_t tail_idx =
            N_vec * 4 + tid;

        while (tail_idx < N) {

            float4 r = rng.rng4();

            float val = x[tail_idx];

            y[tail_idx] =
                (r.x >= p)
                ? val * scale
                : 0.0f;

            tail_idx += stride;
        }

    } else {

        // __half path
        int64_t idx = tid;

        Philox rng(
            seed,
            offset,
            (uint32_t)tid
        );

        while (idx < N) {

            float4 r = rng.rng4();

            float val =
                __half2float(x[idx]);

            float out =
                (r.x >= p)
                ? val * scale
                : 0.0f;

            y[idx] =
                __float2half(out);

            idx += stride;
        }
    }
}


template <typename T>
void launch_dropout(
    const torch::Tensor& x,
    torch::Tensor& y,
    double p,
    uint64_t seed,
    uint64_t offset
) {
    int64_t N = x.numel();

    constexpr int threads = 256;

    int blocks = (N + threads - 1) / threads;

    if constexpr (std::is_same_v<T, float>) {

        dropout_kernel_optimized<float><<<blocks, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            N,
            static_cast<float>(p),
            seed,
            offset
        );

    } else {

        dropout_kernel_optimized<__half><<<blocks, threads>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N,
            static_cast<float>(p),
            seed,
            offset
        );
    }
}

torch::Tensor dropout_cuda(
    torch::Tensor x,
    double p,
    uint64_t seed,
    uint64_t offset
) {
    auto y = torch::empty_like(x);

    int64_t N = x.numel();

    constexpr int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (x.scalar_type() == torch::kFloat32) {

        dropout_kernel_optimized<float><<<blocks, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            N,
            static_cast<float>(p),
            seed,
            offset
        );

    } else if (x.scalar_type() == torch::kFloat16) {

        dropout_kernel_optimized<__half><<<blocks, threads>>>(
            reinterpret_cast<const __half*>(
                x.data_ptr<at::Half>()
            ),
            reinterpret_cast<__half*>(
                y.data_ptr<at::Half>()
            ),
            N,
            static_cast<float>(p),
            seed,
            offset
        );

    } else {
        TORCH_CHECK(
            false,
            "Only float32 and float16 are supported"
        );
    }

    cudaError_t err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        "CUDA kernel failed: ",
        cudaGetErrorString(err)
    );

    return y;
}
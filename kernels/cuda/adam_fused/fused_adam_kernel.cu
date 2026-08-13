#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <type_traits>
#include <cmath>
#include <algorithm>
template <typename T>
__device__ __forceinline__ void adam_step(
    T& p, const T& g, T& m, T& v,
    const float lr, const float beta1, const float beta2, const float eps,
    const float weight_decay, const float bc1, const float bc2)
{
    float gf = static_cast<float>(g);
    float mf = static_cast<float>(m);
    float vf = static_cast<float>(v);
    float pf = static_cast<float>(p);

    mf = beta1 * mf + (1.0f - beta1) * gf;
    vf = beta2 * vf + (1.0f - beta2) * gf * gf;

    const float step_size = lr / bc1;
    const float denom     = sqrtf(vf) / sqrtf(bc2) + eps;

    pf = pf * (1.0f - lr * weight_decay);
    pf = pf - step_size * mf / denom;

    m = static_cast<T>(mf);
    v = static_cast<T>(vf);
    p = static_cast<T>(pf);
}

template <typename T>
__global__ void ADAMfused(
    T* __restrict__       param,
    const T* __restrict__ grad,
    T* __restrict__       exp_avg,
    T* __restrict__       exp_avg_sq,
    int64_t N,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1, 
    float bias_correction2   
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    if constexpr (std::is_same_v<T, float>) {
        const int64_t N_vec = N >> 2; // N / 4
        int64_t idx = tid;
        
        while (idx < N_vec) {
            float4 p_vec = reinterpret_cast<float4*>(param)[idx];
            float4 g_vec = reinterpret_cast<const float4*>(grad)[idx];
            float4 m_vec = reinterpret_cast<float4*>(exp_avg)[idx];
            float4 v_vec = reinterpret_cast<float4*>(exp_avg_sq)[idx];

            adam_step(p_vec.x, g_vec.x, m_vec.x, v_vec.x, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
            adam_step(p_vec.y, g_vec.y, m_vec.y, v_vec.y, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
            adam_step(p_vec.z, g_vec.z, m_vec.z, v_vec.z, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
            adam_step(p_vec.w, g_vec.w, m_vec.w, v_vec.w, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);

            reinterpret_cast<float4*>(param)[idx]     = p_vec;
            reinterpret_cast<float4*>(exp_avg)[idx]  = m_vec;
            reinterpret_cast<float4*>(exp_avg_sq)[idx]= v_vec;
            
            idx += stride;
        }

        int64_t tail_idx = N_vec * 4 + tid;
        if (tail_idx < N) {
            adam_step(param[tail_idx], grad[tail_idx], exp_avg[tail_idx], exp_avg_sq[tail_idx], lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
        }
    } 
    else if constexpr (std::is_same_v<T, __half>) {
        const int64_t N_vec = N >> 1;
        int64_t idx = tid;
        
        while (idx < N_vec) {
            __half2 p_vec = reinterpret_cast<__half2*>(param)[idx];
            __half2 g_vec = reinterpret_cast<const __half2*>(grad)[idx];
            __half2 m_vec = reinterpret_cast<__half2*>(exp_avg)[idx];
            __half2 v_vec = reinterpret_cast<__half2*>(exp_avg_sq)[idx];

            float2 p_f2 = __half22float2(p_vec);
            float2 g_f2 = __half22float2(g_vec);
            float2 m_f2 = __half22float2(m_vec);
            float2 v_f2 = __half22float2(v_vec);

            adam_step(p_f2.x, g_f2.x, m_f2.x, v_f2.x, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
            adam_step(p_f2.y, g_f2.y, m_f2.y, v_f2.y, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);

            reinterpret_cast<__half2*>(param)[idx]      = __float22half2_rn(p_f2);
            reinterpret_cast<__half2*>(exp_avg)[idx]   = __float22half2_rn(m_f2);
            reinterpret_cast<__half2*>(exp_avg_sq)[idx] = __float22half2_rn(v_f2);
            
            idx += stride;
        }

        if ((N & 1) != 0 && tid == 0) {
            int64_t tail_idx = N - 1;
            adam_step(param[tail_idx], grad[tail_idx], exp_avg[tail_idx], exp_avg_sq[tail_idx], lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
        }
    }
    else {
        for (int64_t idx = tid; idx < N; idx += stride) {
            adam_step(param[idx], grad[idx], exp_avg[idx], exp_avg_sq[idx], lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
        }
    }
}

void adam_fused_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step
) {
    int64_t N = param.numel();

    float bc1 = 1.0f - std::pow(
        static_cast<float>(beta1),
        static_cast<float>(step)
    );

    float bc2 = 1.0f - std::pow(
        static_cast<float>(beta2),
        static_cast<float>(step)
    );

    const int block_size = 256;

    const int grid_size =
        std::min<int>(
            (N + block_size - 1) / block_size,
            1024
        );

    ADAMfused<float><<<grid_size, block_size>>>(
        param.data_ptr<float>(),
        grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        N,
        static_cast<float>(lr),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(eps),
        static_cast<float>(weight_decay),
        bc1,
        bc2
    );

    cudaError_t err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        "ADAMfused kernel launch failed: ",
        cudaGetErrorString(err)
    );
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>


__global__ void precompute_rope_cache_kernel(
    float* __restrict__ cos_cache,
    float* __restrict__ sin_cache,
    int max_seq_len,
    int rotary_dim,
    float base
) {
    const int position = blockIdx.x;
    const int d = threadIdx.x;
    const int half_rotary = rotary_dim / 2;

    if (
        position >= max_seq_len ||
        d >= half_rotary
    ) {
        return;
    }

    const float inv_freq = powf(
        base,
        -2.0f * static_cast<float>(d) /
        static_cast<float>(rotary_dim)
    );

    const float angle =
        static_cast<float>(position) * inv_freq;

    float sin_value;
    float cos_value;

    sincosf(
        angle,
        &sin_value,
        &cos_value
    );

    const int64_t index =
        static_cast<int64_t>(position) *
        half_rotary +
        d;

    cos_cache[index] = cos_value;
    sin_cache[index] = sin_value;
}


template<
    typename scalar_t,
    bool Backward
>
__global__ void rope_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int32_t* __restrict__ position_ids,
    int rotary_dim,
    int max_seq_len,
    int head_dim,
    int seq_len,
    int num_heads,
    int64_t position_offset
) {
    const int batch_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int sequence_id = blockIdx.z;
    const int d = threadIdx.x;

    const int half_rotary = rotary_dim / 2;
    const int tail_size = head_dim - rotary_dim;

    const int64_t position =
        position_ids != nullptr
            ? static_cast<int64_t>(
                  position_ids[
                      static_cast<int64_t>(batch_id) *
                          seq_len +
                      sequence_id
                  ]
              )
            : position_offset + sequence_id;

    if (
        position < 0 ||
        position >= max_seq_len
    ) {
        return;
    }

    const int64_t base_index =
        (
            (
                static_cast<int64_t>(batch_id) *
                    num_heads +
                head_id
            ) *
            seq_len +
            sequence_id
        ) *
        head_dim;

    if (d < half_rotary) {
        const int64_t cache_index =
            position * half_rotary + d;

        const float cos_value =
            cos_cache[cache_index];

        float sin_value =
            sin_cache[cache_index];

        if constexpr (Backward) {
            sin_value = -sin_value;
        }

        const int64_t first_index =
            base_index + d;

        const int64_t second_index =
            base_index + d + half_rotary;

        const float first =
            static_cast<float>(x[first_index]);

        const float second =
            static_cast<float>(x[second_index]);

        const float first_output =
            first * cos_value -
            second * sin_value;

        const float second_output =
            second * cos_value +
            first * sin_value;

        out[first_index] =
            static_cast<scalar_t>(first_output);

        out[second_index] =
            static_cast<scalar_t>(second_output);
    }

    if (d < tail_size) {
        const int64_t tail_index =
            base_index + rotary_dim + d;

        out[tail_index] = x[tail_index];
    }
}


std::vector<torch::Tensor> build_rope_cache_cuda(
    const torch::Tensor& reference,
    int64_t max_seq_len,
    int64_t rotary_dim,
    double base
) {
    TORCH_CHECK(
        reference.is_cuda(),
        "reference must be a CUDA tensor"
    );

    TORCH_CHECK(
        max_seq_len > 0,
        "max_seq_len must be positive"
    );

    TORCH_CHECK(
        max_seq_len <=
            std::numeric_limits<int>::max(),
        "max_seq_len exceeds int32 range"
    );

    TORCH_CHECK(
        rotary_dim > 0,
        "rotary_dim must be positive"
    );

    TORCH_CHECK(
        rotary_dim <= 2048,
        "rotary_dim is too large"
    );

    TORCH_CHECK(
        (rotary_dim & 1) == 0,
        "rotary_dim must be even"
    );

    TORCH_CHECK(
        base > 0.0,
        "base must be positive"
    );

    c10::cuda::CUDAGuard device_guard(
        reference.device()
    );

    const int max_seq_len_i =
        static_cast<int>(max_seq_len);

    const int rotary_dim_i =
        static_cast<int>(rotary_dim);

    const int half_rotary =
        rotary_dim_i / 2;

    TORCH_CHECK(
        half_rotary <= 1024,
        "rotary_dim / 2 exceeds CUDA block limit"
    );

    auto options = torch::TensorOptions()
        .device(reference.device())
        .dtype(torch::kFloat32);

    torch::Tensor cos_cache =
        torch::empty(
            {max_seq_len, half_rotary},
            options
        );

    torch::Tensor sin_cache =
        torch::empty(
            {max_seq_len, half_rotary},
            options
        );

    const int threads =
        std::max(
            32,
            ((half_rotary + 31) / 32) * 32
        );

    const dim3 grid(
        static_cast<unsigned int>(max_seq_len_i)
    );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    precompute_rope_cache_kernel
        <<<grid, threads, 0, stream>>>(
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            max_seq_len_i,
            rotary_dim_i,
            static_cast<float>(base)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        cos_cache,
        sin_cache
    };
}


void validate_rope_inputs(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset
) {
    TORCH_CHECK(
        input.is_cuda(),
        "input must be a CUDA tensor"
    );

    TORCH_CHECK(
        input.dim() == 4,
        "input must have shape [B,H,S,D]"
    );

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous"
    );

    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 ||
        input.scalar_type() == torch::kFloat16 ||
        input.scalar_type() == torch::kBFloat16,
        "input must be float32, float16 or bfloat16"
    );

    TORCH_CHECK(
        cos_cache.is_cuda() &&
        sin_cache.is_cuda(),
        "cos_cache and sin_cache must be CUDA tensors"
    );

    TORCH_CHECK(
        cos_cache.device() == input.device() &&
        sin_cache.device() == input.device(),
        "input and caches must be on the same device"
    );

    TORCH_CHECK(
        cos_cache.scalar_type() == torch::kFloat32 &&
        sin_cache.scalar_type() == torch::kFloat32,
        "cos_cache and sin_cache must be float32"
    );

    TORCH_CHECK(
        cos_cache.is_contiguous() &&
        sin_cache.is_contiguous(),
        "cos_cache and sin_cache must be contiguous"
    );

    TORCH_CHECK(
        cos_cache.dim() == 2 &&
        sin_cache.dim() == 2,
        "caches must have shape [max_seq_len, rotary_dim/2]"
    );

    TORCH_CHECK(
        cos_cache.sizes() == sin_cache.sizes(),
        "cos_cache and sin_cache shapes must match"
    );

    const int64_t batch_size = input.size(0);
    const int64_t num_heads = input.size(1);
    const int64_t seq_len = input.size(2);
    const int64_t head_dim = input.size(3);

    TORCH_CHECK(
        batch_size <=
            std::numeric_limits<int>::max(),
        "batch size exceeds int32 range"
    );

    TORCH_CHECK(
        num_heads <=
            std::numeric_limits<int>::max(),
        "num_heads exceeds int32 range"
    );

    TORCH_CHECK(
        seq_len <= 65535,
        "seq_len exceeds CUDA grid.z limit"
    );

    TORCH_CHECK(
        head_dim <=
            std::numeric_limits<int>::max(),
        "head_dim exceeds int32 range"
    );

    TORCH_CHECK(
        rotary_dim > 0,
        "rotary_dim must be positive"
    );

    TORCH_CHECK(
        (rotary_dim & 1) == 0,
        "rotary_dim must be even"
    );

    TORCH_CHECK(
        rotary_dim <= head_dim,
        "rotary_dim cannot exceed head_dim"
    );

    TORCH_CHECK(
        cos_cache.size(1) ==
            rotary_dim / 2,
        "cache width must equal rotary_dim / 2"
    );

    if (!position_ids.has_value()) {
        TORCH_CHECK(
            position_offset >= 0,
            "position_offset cannot be negative"
        );

        TORCH_CHECK(
            position_offset + seq_len <=
                cos_cache.size(0),
            "implicit positions exceed cache size"
        );

        return;
    }

    const torch::Tensor& positions =
        position_ids.value();

    TORCH_CHECK(
        positions.is_cuda(),
        "position_ids must be a CUDA tensor"
    );

    TORCH_CHECK(
        positions.device() == input.device(),
        "position_ids must be on input's device"
    );

    TORCH_CHECK(
        positions.is_contiguous(),
        "position_ids must be contiguous"
    );

    TORCH_CHECK(
        positions.scalar_type() == torch::kInt32,
        "position_ids must be int32"
    );

    TORCH_CHECK(
        positions.dim() == 2 &&
        positions.size(0) == batch_size &&
        positions.size(1) == seq_len,
        "position_ids must have shape [B,S]"
    );
}


template<
    typename scalar_t,
    bool Backward
>
void launch_rope_kernel(
    const torch::Tensor& input,
    const int32_t* position_ptr,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    torch::Tensor& output,
    int rotary_dim,
    int64_t position_offset
) {
    const int batch_size =
        static_cast<int>(input.size(0));

    const int num_heads =
        static_cast<int>(input.size(1));

    const int seq_len =
        static_cast<int>(input.size(2));

    const int head_dim =
        static_cast<int>(input.size(3));

    const int max_seq_len =
        static_cast<int>(cos_cache.size(0));

    const int half_rotary =
        rotary_dim / 2;

    const int tail_size =
        head_dim - rotary_dim;

    const int threads_needed =
        std::max(
            half_rotary,
            tail_size
        );

    const int threads =
        std::max(
            32,
            ((threads_needed + 31) / 32) * 32
        );

    TORCH_CHECK(
        threads <= 1024,
        "head_dim or rotary_dim requires more than 1024 threads"
    );

    const dim3 grid(
        static_cast<unsigned int>(batch_size),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>(seq_len)
    );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    rope_kernel<
        scalar_t,
        Backward
    ><<<grid, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(),
        position_ptr,
        rotary_dim,
        max_seq_len,
        head_dim,
        seq_len,
        num_heads,
        position_offset
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<bool Backward>
torch::Tensor rope_apply_cuda(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset
) {
    validate_rope_inputs(
        input,
        position_ids,
        cos_cache,
        sin_cache,
        rotary_dim,
        position_offset
    );

    torch::Tensor output =
        torch::empty_like(input);

    if (input.numel() == 0) {
        return output;
    }

    c10::cuda::CUDAGuard device_guard(
        input.device()
    );

    const int rotary_dim_i =
        static_cast<int>(rotary_dim);

    const int32_t* position_ptr =
        position_ids.has_value()
            ? position_ids->data_ptr<int32_t>()
            : nullptr;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        Backward
            ? "rope_backward_cuda"
            : "rope_forward_cuda",
        [&] {
            launch_rope_kernel<
                scalar_t,
                Backward
            >(
                input,
                position_ptr,
                cos_cache,
                sin_cache,
                output,
                rotary_dim_i,
                position_offset
            );
        }
    );

    return output;
}


torch::Tensor rope_forward_cuda(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset
) {
    return rope_apply_cuda<false>(
        input,
        position_ids,
        cos_cache,
        sin_cache,
        rotary_dim,
        position_offset
    );
}


torch::Tensor rope_backward_cuda(
    const torch::Tensor& grad_out,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset
) {
    return rope_apply_cuda<true>(
        grad_out,
        position_ids,
        cos_cache,
        sin_cache,
        rotary_dim,
        position_offset
    );
}
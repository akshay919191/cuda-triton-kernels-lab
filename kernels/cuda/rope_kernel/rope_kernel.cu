#include "../common/common_helper.cuh"
#include "private_helper.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#define FPAD 1

/*
Tensor layouts:

x / grad_out:
    [B, H, S, D]

position_ids:
    [B, S], int32 or int64
    May be None.

cos_cache / sin_cache:
    [cache_len, rotary_dim / 2], float32

output / grad_input:
    [B, H, S, D]
    Same dtype as input.
*/


__global__ void build_inv_freq_kernel(
    float* __restrict__ inv_freq,
    int rotary_dim,
    float base
) {
    const int d =
        static_cast<int>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const int halfrot = rotary_dim / 2;

    if (d >= halfrot) {
        return;
    }

    inv_freq[d] = powf(
        base,
        -2.0f * static_cast<float>(d) /
        static_cast<float>(rotary_dim)
    );
}


__global__ void build_rope_cache_kernel(
    const float* __restrict__ inv_freq,
    float* __restrict__ cos_cache,
    float* __restrict__ sin_cache,
    int cache_len,
    int halfrot
) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const int64_t total =
        static_cast<int64_t>(cache_len) * halfrot;

    if (idx >= total) {
        return;
    }

    const int position =
        static_cast<int>(idx / halfrot);

    const int d =
        static_cast<int>(idx % halfrot);

    const float theta =
        static_cast<float>(position) * inv_freq[d];

    float sn;
    float cs;

    sincosf(theta, &sn, &cs);

    sin_cache[idx] = sn;
    cos_cache[idx] = cs;
}


template<typename index_t>
__device__ __forceinline__ int64_t get_rope_position(
    const index_t* __restrict__ position_ids,
    int batch_id,
    int global_row,
    int seq_len,
    int64_t position_offset
) {
    if (position_ids != nullptr) {
        return static_cast<int64_t>(
            position_ids[
                static_cast<int64_t>(batch_id) * seq_len +
                global_row
            ]
        );
    }

    return position_offset + global_row;
}


template<typename index_t, int Br>
__device__ __forceinline__ void load_rope_cache_tile(
    const index_t* __restrict__ position_ids,
    const float* __restrict__ sin_cached,
    const float* __restrict__ cos_cached,
    float* __restrict__ sin_smem,
    float* __restrict__ cos_smem,
    int batch_id,
    int tile_id,
    int seq_len,
    int cache_len,
    int halfrot,
    int64_t position_offset
) {
    const int tid = threadIdx.x;

    /*
    Vector path.

    Safe when halfrot is divisible by four because:

        cache row stride = halfrot floats
        c                = multiple of four
        torch allocation = sufficiently aligned
    */
    if ((halfrot & 3) == 0) {
        const int vecs_per_row = halfrot / 4;
        const int total_vecs = Br * vecs_per_row;

        for (
            int i = tid;
            i < total_vecs;
            i += blockDim.x
        ) {
            const int row = i / vecs_per_row;
            const int c = (i % vecs_per_row) * 4;

            const int global_row =
                tile_id * Br + row;

            /*
            Defaults ensure that invalid/tail rows do not leave
            uninitialized shared memory.

            sin = 0, cos = 1 means identity rotation.
            */
            float4 sn =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            float4 cs =
                make_float4(1.0f, 1.0f, 1.0f, 1.0f);

            if (global_row < seq_len) {
                const int64_t position =
                    get_rope_position(
                        position_ids,
                        batch_id,
                        global_row,
                        seq_len,
                        position_offset
                    );

                if (
                    position >= 0 &&
                    position < cache_len
                ) {
                    const int64_t src =
                        position * halfrot + c;

                    sn =
                        *reinterpret_cast<const float4*>(
                            sin_cached + src
                        );

                    cs =
                        *reinterpret_cast<const float4*>(
                            cos_cached + src
                        );
                }
            }

            const int dst =
                row * halfrot + c;

            *reinterpret_cast<float4*>(
                sin_smem + dst
            ) = sn;

            *reinterpret_cast<float4*>(
                cos_smem + dst
            ) = cs;
        }

        return;
    }

    // Scalar fallback.
    const int total = Br * halfrot;

    for (
        int i = tid;
        i < total;
        i += blockDim.x
    ) {
        const int row = i / halfrot;
        const int d = i % halfrot;

        const int global_row =
            tile_id * Br + row;

        float sn = 0.0f;
        float cs = 1.0f;

        if (global_row < seq_len) {
            const int64_t position =
                get_rope_position(
                    position_ids,
                    batch_id,
                    global_row,
                    seq_len,
                    position_offset
                );

            if (
                position >= 0 &&
                position < cache_len
            ) {
                const int64_t src =
                    position * halfrot + d;

                sn = sin_cached[src];
                cs = cos_cached[src];
            }
        }

        sin_smem[i] = sn;
        cos_smem[i] = cs;
    }
}


template<
    bool Backward,
    typename scalar_t,
    typename index_t,
    int Br
>
__global__ void rope_apply_kernel(
    const scalar_t* __restrict__ input,
    const index_t* __restrict__ position_ids,
    const float* __restrict__ sin_cached,
    const float* __restrict__ cos_cached,
    scalar_t* __restrict__ output,
    int seq_len,
    int cache_len,
    int rotary_dim,
    int head_dim,
    int num_heads,
    int64_t position_offset
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    const int batch_id =
        static_cast<int>(blockIdx.x);

    const int head_id =
        static_cast<int>(blockIdx.y);

    const int tile_id =
        static_cast<int>(blockIdx.z);

    const int halfrot = rotary_dim / 2;

    const int row_stride =
        (head_dim + FPAD + 3) & ~3;

    const int64_t tensor_base =
        (
            static_cast<int64_t>(batch_id) *
                num_heads +
            head_id
        ) *
        static_cast<int64_t>(seq_len) *
        head_dim;

    extern __shared__ char smem_raw[];

    char* ptr = smem_raw;

    // Input tile alignment.
    ptr = reinterpret_cast<char*>(
        (
            reinterpret_cast<uintptr_t>(ptr) +
            15ULL
        ) &
        ~static_cast<uintptr_t>(15ULL)
    );

    float* input_smem =
        reinterpret_cast<float*>(ptr);

    ptr +=
        static_cast<size_t>(Br) *
        row_stride *
        sizeof(float);

    // Sin tile alignment.
    ptr = reinterpret_cast<char*>(
        (
            reinterpret_cast<uintptr_t>(ptr) +
            15ULL
        ) &
        ~static_cast<uintptr_t>(15ULL)
    );

    float* sin_smem =
        reinterpret_cast<float*>(ptr);

    ptr +=
        static_cast<size_t>(Br) *
        halfrot *
        sizeof(float);

    // Cos tile alignment.
    ptr = reinterpret_cast<char*>(
        (
            reinterpret_cast<uintptr_t>(ptr) +
            15ULL
        ) &
        ~static_cast<uintptr_t>(15ULL)
    );

    float* cos_smem =
        reinterpret_cast<float*>(ptr);

    load_rope_cache_tile<index_t, Br>(
        position_ids,
        sin_cached,
        cos_cached,
        sin_smem,
        cos_smem,
        batch_id,
        tile_id,
        seq_len,
        cache_len,
        halfrot,
        position_offset
    );

    __syncthreads();

    copy_to_float_smem<scalar_t, Br>(
        input + tensor_base,
        input_smem,
        row_stride,
        tile_id,
        seq_len,
        head_dim
    );

    if constexpr (
        std::is_same_v<scalar_t, float>
    ) {
        if ((head_dim & 3) == 0) {
            asm volatile(
                "cp.async.commit_group;\n"
                ::
            );

            asm volatile(
                "cp.async.wait_group 0;\n"
                :::
                "memory"
            );
        }
    }

    __syncthreads();

    for (
        int row = warp_id;
        row < Br;
        row += num_warps
    ) {
        const int global_row =
            tile_id * Br + row;

        if (global_row >= seq_len) {
            continue;
        }

        for (
            int d = lane;
            d < head_dim;
            d += 32
        ) {
            const float value =
                input_smem[
                    row * row_stride + d
                ];

            float result = value;

            // Dimensions after rotary_dim stay unchanged.
            if (d < rotary_dim) {
                int freq_idx;
                float partner;

                if (d < halfrot) {
                    freq_idx = d;

                    const float second_half =
                        input_smem[
                            row * row_stride +
                            d + halfrot
                        ];

                    if constexpr (Backward) {
                        // dx1 = dy1*c + dy2*s
                        partner = second_half;
                    } else {
                        // y1 = x1*c - x2*s
                        partner = -second_half;
                    }
                } else {
                    freq_idx = d - halfrot;

                    const float first_half =
                        input_smem[
                            row * row_stride +
                            d - halfrot
                        ];

                    if constexpr (Backward) {
                        // dx2 = dy2*c - dy1*s
                        partner = -first_half;
                    } else {
                        // y2 = x2*c + x1*s
                        partner = first_half;
                    }
                }

                const float sn =
                    sin_smem[
                        row * halfrot + freq_idx
                    ];

                const float cs =
                    cos_smem[
                        row * halfrot + freq_idx
                    ];

                result =
                    value * cs +
                    partner * sn;
            }

            const int64_t output_idx =
                tensor_base +
                static_cast<int64_t>(global_row) *
                    head_dim +
                d;

            output[output_idx] =
                static_cast<scalar_t>(result);
        }
    }
}


std::vector<torch::Tensor> build_rope_cache_cuda(
    const torch::Tensor& reference,
    int64_t cache_len,
    int64_t rotary_dim,
    double base
) {
    TORCH_CHECK(
        reference.is_cuda(),
        "reference must be a CUDA tensor"
    );

    TORCH_CHECK(
        cache_len > 0,
        "cache_len must be positive"
    );

    TORCH_CHECK(
        cache_len <= std::numeric_limits<int>::max(),
        "cache_len is too large"
    );

    TORCH_CHECK(
        rotary_dim > 0,
        "rotary_dim must be positive"
    );

    TORCH_CHECK(
        rotary_dim <= std::numeric_limits<int>::max(),
        "rotary_dim is too large"
    );

    TORCH_CHECK(
        (rotary_dim & 1) == 0,
        "rotary_dim must be even"
    );

    TORCH_CHECK(
        std::isfinite(base) && base > 0.0,
        "base must be finite and positive"
    );

    c10::cuda::CUDAGuard device_guard(
        reference.device()
    );

    const int cache_len_i =
        static_cast<int>(cache_len);

    const int rotary_dim_i =
        static_cast<int>(rotary_dim);

    const int halfrot =
        rotary_dim_i / 2;

    auto options = torch::TensorOptions()
        .device(reference.device())
        .dtype(torch::kFloat32);

    torch::Tensor inv_freq =
        torch::empty(
            {halfrot},
            options
        );

    torch::Tensor cos_cache =
        torch::empty(
            {cache_len, halfrot},
            options
        );

    torch::Tensor sin_cache =
        torch::empty(
            {cache_len, halfrot},
            options
        );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    constexpr int threads = 256;

    const int inv_blocks =
        (halfrot + threads - 1) /
        threads;

    build_inv_freq_kernel
        <<<inv_blocks, threads, 0, stream>>>(
            inv_freq.data_ptr<float>(),
            rotary_dim_i,
            static_cast<float>(base)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t total =
        cache_len *
        static_cast<int64_t>(halfrot);

    const int64_t cache_blocks_64 =
        (total + threads - 1) /
        threads;

    int device = 0;
    C10_CUDA_CHECK(
        cudaGetDevice(&device)
    );

    cudaDeviceProp properties{};

    C10_CUDA_CHECK(
        cudaGetDeviceProperties(
            &properties,
            device
        )
    );

    TORCH_CHECK(
        cache_blocks_64 <=
            properties.maxGridSize[0],
        "RoPE cache is too large for one grid launch"
    );

    const int cache_blocks =
        static_cast<int>(cache_blocks_64);

    build_rope_cache_kernel
        <<<cache_blocks, threads, 0, stream>>>(
            inv_freq.data_ptr<float>(),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            cache_len_i,
            halfrot
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        cos_cache,
        sin_cache,
        inv_freq
    };
}


template<
    bool Backward,
    typename scalar_t,
    typename index_t,
    int Br
>
void launch_rope_apply(
    const torch::Tensor& input,
    const index_t* position_ptr,
    const torch::Tensor& sin_cache,
    const torch::Tensor& cos_cache,
    torch::Tensor& output,
    int rotary_dim,
    int64_t position_offset
) {
    const int batch =
        static_cast<int>(input.size(0));

    const int num_heads =
        static_cast<int>(input.size(1));

    const int seq_len =
        static_cast<int>(input.size(2));

    const int head_dim =
        static_cast<int>(input.size(3));

    const int cache_len =
        static_cast<int>(sin_cache.size(0));

    const int halfrot =
        rotary_dim / 2;

    const int tiles =
        (seq_len + Br - 1) /
        Br;

    constexpr int threads = 128;

    dim3 grid(
        static_cast<unsigned int>(batch),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>(tiles)
    );

    const int row_stride =
        (head_dim + FPAD + 3) & ~3;

    const size_t smem_bytes =
        48 +
        static_cast<size_t>(Br) *
            row_stride *
            sizeof(float) +
        static_cast<size_t>(Br) *
            halfrot *
            sizeof(float) *
            2;

    int device = 0;

    C10_CUDA_CHECK(
        cudaGetDevice(&device)
    );

    cudaDeviceProp properties{};

    C10_CUDA_CHECK(
        cudaGetDeviceProperties(
            &properties,
            device
        )
    );

    TORCH_CHECK(
        batch <= properties.maxGridSize[0],
        "Batch size exceeds CUDA grid.x limit"
    );

    TORCH_CHECK(
        num_heads <= properties.maxGridSize[1],
        "Number of heads exceeds CUDA grid.y limit"
    );

    TORCH_CHECK(
        tiles <= properties.maxGridSize[2],
        "Number of sequence tiles exceeds CUDA grid.z limit"
    );

    const size_t maximum_dynamic_smem =
        std::max(
            static_cast<size_t>(
                properties.sharedMemPerBlock
            ),
            static_cast<size_t>(
                properties.sharedMemPerBlockOptin
            )
        );

    TORCH_CHECK(
        smem_bytes <= maximum_dynamic_smem,
        "Requested shared memory ",
        smem_bytes,
        " bytes exceeds device limit ",
        maximum_dynamic_smem,
        " bytes. Reduce Br."
    );

    TORCH_CHECK(
        smem_bytes <=
            static_cast<size_t>(
                std::numeric_limits<int>::max()
            ),
        "Shared-memory request is too large"
    );

    if (
        smem_bytes >
        static_cast<size_t>(
            properties.sharedMemPerBlock
        )
    ) {
        C10_CUDA_CHECK(
            cudaFuncSetAttribute(
                rope_apply_kernel<
                    Backward,
                    scalar_t,
                    index_t,
                    Br
                >,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)
            )
        );
    }

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    rope_apply_kernel<
        Backward,
        scalar_t,
        index_t,
        Br
    ><<<grid, threads, smem_bytes, stream>>>(
        input.data_ptr<scalar_t>(),
        position_ptr,
        sin_cache.data_ptr<float>(),
        cos_cache.data_ptr<float>(),
        output.data_ptr<scalar_t>(),
        seq_len,
        cache_len,
        rotary_dim,
        head_dim,
        num_heads,
        position_offset
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<
    bool Backward,
    typename scalar_t,
    typename index_t
>
void dispatch_rope_br(
    const torch::Tensor& input,
    const index_t* position_ptr,
    const torch::Tensor& sin_cache,
    const torch::Tensor& cos_cache,
    torch::Tensor& output,
    int rotary_dim,
    int64_t position_offset,
    int64_t Br
) {
    switch (Br) {
        case 16:
            launch_rope_apply<
                Backward,
                scalar_t,
                index_t,
                16
            >(
                input,
                position_ptr,
                sin_cache,
                cos_cache,
                output,
                rotary_dim,
                position_offset
            );
            break;

        case 32:
            launch_rope_apply<
                Backward,
                scalar_t,
                index_t,
                32
            >(
                input,
                position_ptr,
                sin_cache,
                cos_cache,
                output,
                rotary_dim,
                position_offset
            );
            break;

        case 64:
            launch_rope_apply<
                Backward,
                scalar_t,
                index_t,
                64
            >(
                input,
                position_ptr,
                sin_cache,
                cos_cache,
                output,
                rotary_dim,
                position_offset
            );
            break;

        default:
            TORCH_CHECK(
                false,
                "Supported Br values are 16, 32 and 64"
            );
    }
}


void validate_rope_inputs(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset,
    int64_t Br
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
        input.size(0) <= std::numeric_limits<int>::max() &&
        input.size(1) <= std::numeric_limits<int>::max() &&
        input.size(2) <= std::numeric_limits<int>::max() &&
        input.size(3) <= std::numeric_limits<int>::max(),
        "input dimensions exceed int32 kernel limits"
    );

    TORCH_CHECK(
        cos_cache.is_cuda() &&
        sin_cache.is_cuda(),
        "RoPE caches must be CUDA tensors"
    );

    TORCH_CHECK(
        cos_cache.device() == input.device() &&
        sin_cache.device() == input.device(),
        "input and caches must be on the same device"
    );

    TORCH_CHECK(
        cos_cache.scalar_type() == torch::kFloat32 &&
        sin_cache.scalar_type() == torch::kFloat32,
        "RoPE caches must be float32"
    );

    TORCH_CHECK(
        cos_cache.is_contiguous() &&
        sin_cache.is_contiguous(),
        "RoPE caches must be contiguous"
    );

    TORCH_CHECK(
        cos_cache.dim() == 2 &&
        sin_cache.dim() == 2,
        "RoPE caches must have shape [cache_len, halfrot]"
    );

    TORCH_CHECK(
        cos_cache.size(0) == sin_cache.size(0) &&
        cos_cache.size(1) == sin_cache.size(1),
        "sin and cos cache shapes must match"
    );

    TORCH_CHECK(
        cos_cache.size(0) > 0,
        "cache_len must be positive"
    );

    TORCH_CHECK(
        cos_cache.size(0) <=
            std::numeric_limits<int>::max(),
        "cache_len exceeds int32 kernel limits"
    );

    const int64_t batch =
        input.size(0);

    const int64_t seq_len =
        input.size(2);

    const int64_t head_dim =
        input.size(3);

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
        "cache second dimension must equal rotary_dim / 2"
    );

    TORCH_CHECK(
        Br == 16 ||
        Br == 32 ||
        Br == 64,
        "Br must be 16, 32 or 64"
    );

    if (!position_ids.has_value()) {
        TORCH_CHECK(
            position_offset >= 0,
            "position_offset cannot be negative"
        );

        TORCH_CHECK(
            position_offset + seq_len <=
                cos_cache.size(0),
            "implicit positions exceed RoPE cache"
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
        "position_ids must be on the same device as input"
    );

    TORCH_CHECK(
        positions.is_contiguous(),
        "position_ids must be contiguous"
    );

    TORCH_CHECK(
        positions.dim() == 2 &&
        positions.size(0) == batch &&
        positions.size(1) == seq_len,
        "position_ids must have shape [B,S]"
    );

    TORCH_CHECK(
        positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64,
        "position_ids must be int32 or int64"
    );
}


template<bool Backward>
torch::Tensor rope_apply_cuda_impl(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset,
    int64_t Br
) {
    validate_rope_inputs(
        input,
        position_ids,
        cos_cache,
        sin_cache,
        rotary_dim,
        position_offset,
        Br
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

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        Backward
            ? "rope_backward_cuda"
            : "rope_forward_cuda",
        [&] {
            if (!position_ids.has_value()) {
                dispatch_rope_br<
                    Backward,
                    scalar_t,
                    int32_t
                >(
                    input,
                    nullptr,
                    sin_cache,
                    cos_cache,
                    output,
                    rotary_dim_i,
                    position_offset,
                    Br
                );

                return;
            }

            const torch::Tensor& positions =
                position_ids.value();

            if (
                positions.scalar_type() ==
                torch::kInt32
            ) {
                dispatch_rope_br<
                    Backward,
                    scalar_t,
                    int32_t
                >(
                    input,
                    positions.data_ptr<int32_t>(),
                    sin_cache,
                    cos_cache,
                    output,
                    rotary_dim_i,
                    position_offset,
                    Br
                );
            } else {
                dispatch_rope_br<
                    Backward,
                    scalar_t,
                    int64_t
                >(
                    input,
                    positions.data_ptr<int64_t>(),
                    sin_cache,
                    cos_cache,
                    output,
                    rotary_dim_i,
                    position_offset,
                    Br
                );
            }
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
    int64_t position_offset,
    int64_t Br
) {
    return rope_apply_cuda_impl<false>(
        input,
        position_ids,
        cos_cache,
        sin_cache,
        rotary_dim,
        position_offset,
        Br
    );
}


torch::Tensor rope_backward_cuda(
    const torch::Tensor& grad_out,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset,
    int64_t Br
) {
    return rope_apply_cuda_impl<true>(
        grad_out,
        position_ids,
        cos_cache,
        sin_cache,
        rotary_dim,
        position_offset,
        Br
    );
}
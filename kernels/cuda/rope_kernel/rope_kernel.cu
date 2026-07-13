#include "../common/common_helper.cuh"
#include "private_helper.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#define FPAD 1

// x            : [B, H, S, D]       FP32 / FP16 / BF16
// position_ids : [B, S]             int32 
// inv_freq     : [rotary_dim / 2]   FP32
// output       : [B, H, S, D]       same dtype as x

template<int Base>
__global__ void build_inv_freq_kernel(
    float* __restrict__ inv_freq,
    int rotary_dim
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int halfrot = rotary_dim / 2;

    if (d >= halfrot)
        return;

    inv_freq[d] = powf(
        static_cast<float>(Base),
        -2.0f * static_cast<float>(d) /
        static_cast<float>(rotary_dim)
    );
}

__global__ void build_rope_cache_kernel(
    const float* __restrict__ inv_freq, // [halfrot]
    float* __restrict__ cos_cache,      // [cache_len, halfrot]
    float* __restrict__ sin_cache,
    int cache_len,
    int halfrot
) {
    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    int64_t total =
        static_cast<int64_t>(cache_len) * halfrot;

    if (idx >= total)
        return;

    int position = static_cast<int>(idx / halfrot);
    int d = static_cast<int>(idx % halfrot);

    float theta =
        static_cast<float>(position) * inv_freq[d];

    float sn, cs;
    sincosf(theta, &sn, &cs);

    sin_cache[idx] = sn;
    cos_cache[idx] = cs;
}

template<typename scalar_t, typename index_t, int Br>
__global__ void rope_forward_kernel(
    const scalar_t* __restrict__ matrix,       // [B,H,S,D]
    const index_t* __restrict__ position_ids,  // [B,S] or nullptr
    const float* __restrict__ sin_cached,      // [cache_len,halfrot]
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

    const int batch_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int tile_id = blockIdx.z;

    const int halfrot = rotary_dim / 2;
    const int row_stride = head_dim + FPAD;

    const int64_t tensor_base =
        (static_cast<int64_t>(batch_id) * num_heads + head_id) *
        static_cast<int64_t>(seq_len) * head_dim;

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;

    // Align input shared-memory region.
    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15ULL) &
        ~static_cast<uintptr_t>(15ULL)
    );

    float* smem_x = reinterpret_cast<float*>(ptr);
    ptr += static_cast<size_t>(Br) *
           row_stride * sizeof(float);

    // Align sin shared-memory region.
    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15ULL) &
        ~static_cast<uintptr_t>(15ULL)
    );

    float* sin_smem = reinterpret_cast<float*>(ptr);
    ptr += static_cast<size_t>(Br) *
           halfrot * sizeof(float);

    // Align cos shared-memory region.
    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15ULL) &
        ~static_cast<uintptr_t>(15ULL)
    );

    float* cos_smem = reinterpret_cast<float*>(ptr);


    if ((halfrot & 3) == 0) {
        const int vecs_per_row = halfrot / 4;
        const int total_vecs = Br * vecs_per_row;

        for (int i = tid; i < total_vecs; i += blockDim.x) {
            int row = i / vecs_per_row;
            int c = (i % vecs_per_row) * 4;

            int global_row = tile_id * Br + row;

            float4 sin_value =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            float4 cos_value =
                make_float4(1.0f, 1.0f, 1.0f, 1.0f);

            if (global_row < seq_len) {
                int64_t position =
                    position_ids
                        ? static_cast<int64_t>(
                              position_ids[
                                  static_cast<int64_t>(batch_id) *
                                      seq_len +
                                  global_row
                              ]
                          )
                        : position_offset + global_row;

                // Prevent an out-of-bounds cache access.
                if (position >= 0 && position < cache_len) {
                    int64_t src =
                        position * halfrot + c;

                    sin_value =
                        *reinterpret_cast<const float4*>(
                            sin_cached + src
                        );

                    cos_value =
                        *reinterpret_cast<const float4*>(
                            cos_cached + src
                        );
                }
            }

            int dst = row * halfrot + c;

            *reinterpret_cast<float4*>(sin_smem + dst) =
                sin_value;

            *reinterpret_cast<float4*>(cos_smem + dst) =
                cos_value;
        }
    } else {
        for (
            int i = tid;
            i < Br * halfrot;
            i += blockDim.x
        ) {
            int row = i / halfrot;
            int d = i % halfrot;

            int global_row = tile_id * Br + row;

            float sn = 0.0f;
            float cs = 1.0f;

            if (global_row < seq_len) {
                int64_t position =
                    position_ids
                        ? static_cast<int64_t>(
                              position_ids[
                                  static_cast<int64_t>(batch_id) *
                                      seq_len +
                                  global_row
                              ]
                          )
                        : position_offset + global_row;

                if (position >= 0 && position < cache_len) {
                    int64_t src =
                        position * halfrot + d;

                    sn = sin_cached[src];
                    cs = cos_cached[src];
                }
            }

            sin_smem[i] = sn;
            cos_smem[i] = cs;
        }
    }

    __syncthreads();

    
    copy_to_float_smem<scalar_t, Br>(
        matrix + tensor_base,
        smem_x,
        row_stride,
        tile_id,
        seq_len,
        head_dim
    );

    if constexpr (std::is_same_v<scalar_t, float>) {
        if ((head_dim & 3) == 0) {
            asm volatile("cp.async.commit_group;\n" ::);
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
        int global_row = tile_id * Br + row;

        if (global_row >= seq_len)
            continue;

        for (
            int d = lane;
            d < head_dim;
            d += 32
        ) {
            float x =
                smem_x[row * row_stride + d];

            float result = x;

            if (d < rotary_dim) {
                int freq_idx;
                float rotated;

                if (d < halfrot) {
                    freq_idx = d;

                    rotated = -smem_x[
                        row * row_stride +
                        d + halfrot
                    ];
                } else {
                    freq_idx = d - halfrot;

                    rotated = smem_x[
                        row * row_stride +
                        d - halfrot
                    ];
                }

                float sn =
                    sin_smem[
                        row * halfrot + freq_idx
                    ];

                float cs =
                    cos_smem[
                        row * halfrot + freq_idx
                    ];

                result = x * cs + rotated * sn;
            }

            int64_t output_idx =
                tensor_base +
                static_cast<int64_t>(global_row) *
                    head_dim +
                d;

            output[output_idx] =
                static_cast<scalar_t>(result);
        }
    }
}

/// backward


std::vector<torch::Tensor> build_rope_cache_cuda(
    const torch::Tensor& reference,
    int64_t cache_len,
    int64_t rotary_dim
) {
    TORCH_CHECK(reference.is_cuda(),
                "reference must be a CUDA tensor");

    TORCH_CHECK(cache_len > 0,
                "cache_len must be positive");

    TORCH_CHECK(rotary_dim > 0,
                "rotary_dim must be positive");

    TORCH_CHECK((rotary_dim & 1) == 0,
                "rotary_dim must be even");

    c10::cuda::CUDAGuard device_guard(
        reference.device()
    );

    int halfrot = static_cast<int>(rotary_dim / 2);

    auto options = torch::TensorOptions()
        .device(reference.device())
        .dtype(torch::kFloat32);

    torch::Tensor inv_freq =
        torch::empty({halfrot}, options);

    torch::Tensor cos_cache =
        torch::empty({cache_len, halfrot}, options);

    torch::Tensor sin_cache =
        torch::empty({cache_len, halfrot}, options);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    constexpr int threads = 256;

    int inv_blocks =
        (halfrot + threads - 1) / threads;

    build_inv_freq_kernel<10000>
        <<<inv_blocks, threads, 0, stream>>>(
            inv_freq.data_ptr<float>(),
            static_cast<int>(rotary_dim)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t total =
        cache_len * static_cast<int64_t>(halfrot);

    int cache_blocks =
        static_cast<int>(
            (total + threads - 1) / threads
        );

    build_rope_cache_kernel
        <<<cache_blocks, threads, 0, stream>>>(
            inv_freq.data_ptr<float>(),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            static_cast<int>(cache_len),
            halfrot
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Keep inv_freq too, especially if you later expand the cache.
    return {
        cos_cache,
        sin_cache,
        inv_freq
    };
}

template<
    typename scalar_t,
    typename index_t,
    int Br
>
void launch_rope_forward(
    const torch::Tensor& x,
    const index_t* position_ptr,
    const torch::Tensor& sin_cache,
    const torch::Tensor& cos_cache,
    torch::Tensor& output,
    int rotary_dim,
    int64_t position_offset
) {
    int batch = static_cast<int>(x.size(0));
    int heads = static_cast<int>(x.size(1));
    int seq_len = static_cast<int>(x.size(2));
    int head_dim = static_cast<int>(x.size(3));

    int cache_len =
        static_cast<int>(sin_cache.size(0));

    int halfrot = rotary_dim / 2;
    int tiles = (seq_len + Br - 1) / Br;

    constexpr int threads = 128;

    dim3 grid(batch, heads, tiles);

    size_t smem_bytes =
        48 + // alignment slack
        static_cast<size_t>(Br) *
            (head_dim + FPAD) * sizeof(float) +
        static_cast<size_t>(Br) *
            halfrot * sizeof(float) * 2;

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    // Necessary when dynamic shared memory exceeds the default limit.
    if (smem_bytes > 48 * 1024) {
        C10_CUDA_CHECK(
            cudaFuncSetAttribute(
                rope_forward_kernel<
                    scalar_t,
                    index_t,
                    Br
                >,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)
            )
        );
    }

    rope_forward_kernel<
        scalar_t,
        index_t,
        Br
    ><<<grid, threads, smem_bytes, stream>>>(
        x.data_ptr<scalar_t>(),
        position_ptr,
        sin_cache.data_ptr<float>(),
        cos_cache.data_ptr<float>(),
        output.data_ptr<scalar_t>(),
        seq_len,
        cache_len,
        rotary_dim,
        head_dim,
        heads,
        position_offset
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor rope_forward_cuda(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& position_ids,
    const torch::Tensor& cos_cache,
    const torch::Tensor& sin_cache,
    int64_t rotary_dim,
    int64_t position_offset,
    int64_t Br
) {
    TORCH_CHECK(x.is_cuda(),
                "x must be a CUDA tensor");

    TORCH_CHECK(x.dim() == 4,
                "x must have shape [B,H,S,D]");

    TORCH_CHECK(x.is_contiguous(),
                "x must be contiguous");

    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32 ||
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16,
        "x must be float32, float16 or bfloat16"
    );

    TORCH_CHECK(cos_cache.is_cuda() &&
                sin_cache.is_cuda(),
                "RoPE caches must be CUDA tensors");

    TORCH_CHECK(
        cos_cache.device() == x.device() &&
        sin_cache.device() == x.device(),
        "x and caches must be on the same device"
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
        cos_cache.sizes() == sin_cache.sizes(),
        "sin and cos cache shapes must match"
    );

    TORCH_CHECK(cos_cache.dim() == 2,
                "cache must have shape [cache_len, halfrot]");

    int64_t batch = x.size(0);
    int64_t seq_len = x.size(2);
    int64_t head_dim = x.size(3);

    TORCH_CHECK(rotary_dim > 0,
                "rotary_dim must be positive");

    TORCH_CHECK((rotary_dim & 1) == 0,
                "rotary_dim must be even");

    TORCH_CHECK(rotary_dim <= head_dim,
                "rotary_dim cannot exceed head_dim");

    TORCH_CHECK(
        cos_cache.size(1) == rotary_dim / 2,
        "cache second dimension must equal rotary_dim / 2"
    );

    if (!position_ids.has_value()) {
        TORCH_CHECK(position_offset >= 0,
                    "position_offset cannot be negative");

        TORCH_CHECK(
            position_offset + seq_len <=
                cos_cache.size(0),
            "implicit positions exceed RoPE cache"
        );
    } else {
        const torch::Tensor& pos =
            position_ids.value();

        TORCH_CHECK(pos.is_cuda(),
                    "position_ids must be CUDA");

        TORCH_CHECK(pos.device() == x.device(),
                    "position_ids must be on x's device");

        TORCH_CHECK(pos.is_contiguous(),
                    "position_ids must be contiguous");

        TORCH_CHECK(
            pos.dim() == 2 &&
            pos.size(0) == batch &&
            pos.size(1) == seq_len,
            "position_ids must have shape [B,S]"
        );

        TORCH_CHECK(
            pos.scalar_type() == torch::kInt32 ||
            pos.scalar_type() == torch::kInt64,
            "position_ids must be int32 or int64"
        );
    }

    torch::Tensor output = torch::empty_like(x);

    if (x.numel() == 0)
        return output;

    c10::cuda::CUDAGuard device_guard(x.device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "rope_forward_cuda",
        [&] {
            auto launch_for_br = [&](auto* pos_ptr) {
                using index_t = std::remove_pointer_t<
                    decltype(pos_ptr)
                >;

                switch (Br) {
                    case 16:
                        launch_rope_forward<
                            scalar_t, index_t, 16
                        >(
                            x,
                            pos_ptr,
                            sin_cache,
                            cos_cache,
                            output,
                            static_cast<int>(rotary_dim),
                            position_offset
                        );
                        break;

                    case 32:
                        launch_rope_forward<
                            scalar_t, index_t, 32
                        >(
                            x,
                            pos_ptr,
                            sin_cache,
                            cos_cache,
                            output,
                            static_cast<int>(rotary_dim),
                            position_offset
                        );
                        break;

                    case 64:
                        launch_rope_forward<
                            scalar_t, index_t, 64
                        >(
                            x,
                            pos_ptr,
                            sin_cache,
                            cos_cache,
                            output,
                            static_cast<int>(rotary_dim),
                            position_offset
                        );
                        break;

                    default:
                        TORCH_CHECK(
                            false,
                            "Supported Br values are 16, 32 and 64"
                        );
                }
            };

            if (!position_ids.has_value()) {
                launch_for_br(
                    static_cast<const int32_t*>(nullptr)
                );
            } else if (
                position_ids->scalar_type() ==
                torch::kInt32
            ) {
                launch_for_br(
                    position_ids->data_ptr<int32_t>()
                );
            } else {
                launch_for_br(
                    position_ids->data_ptr<int64_t>()
                );
            }
        }
    );

    return output;
}
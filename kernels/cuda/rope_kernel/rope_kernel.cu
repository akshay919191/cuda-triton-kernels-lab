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

            // Dimensions outside rotary_dim remain unchanged.
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
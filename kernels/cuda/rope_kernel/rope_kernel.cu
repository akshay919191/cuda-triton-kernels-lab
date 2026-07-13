#include "../common/common_helper.cuh"
#include "private_helper.cuh"

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <vector>

#define PADDING 8
#define fPAD 1

// x            : [B, H, S, D]       FP32 / FP16 / BF16
// position_ids : [B, S]             int32 
// inv_freq     : [rotary_dim / 2]   FP32
// output       : [B, H, S, D]       same dtype as x

template<int Base>
__global__ void build_rope_cache_nopos(
    float* __restrict__ cos_cache,  // [cache_len, halfrot]
    float* __restrict__ sin_cache,
    int cache_len,
    int rotary_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfrot = rotary_dim / 2;
    int total = cache_len * halfrot;

    if (idx >= total)
        return;

    int position = idx / halfrot;
    int d = idx % halfrot;

    float inv_freq = powf(
        static_cast<float>(Base),
        -2.0f * static_cast<float>(d) /
        static_cast<float>(rotary_dim)
    );

    float theta = static_cast<float>(position) * inv_freq;

    float sn, cs;
    sincosf(theta, &sn, &cs);

    sin_cache[idx] = sn;
    cos_cache[idx] = cs;
}

template<typename scalar_t, int Br>
__global__ void ropefwd_kernel(
    const scalar_t* __restrict__ matrix,      // [B,H,S,D]
    const int32_t* __restrict__ position_ids, // [B,S]
    const float* __restrict__ sin_cached,     // [cache_len,halfrot]
    const float* __restrict__ cos_cached,
    scalar_t* __restrict__ output,
    int seq_len,
    int cache_len,
    int rotary_dim,
    int head_dim,
    int num_heads
) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int num_warps = blockDim.x >> 5;

    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tile_id = blockIdx.z;

    int halfrot = rotary_dim / 2;
    int row_stride = head_dim + fPAD;

    int64_t tensor_base =
        (static_cast<int64_t>(batch_id) * num_heads + head_id) *
        seq_len * head_dim;

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    float* smem_x = reinterpret_cast<float*>(ptr);
    ptr += Br * row_stride * sizeof(float);

    // Only halfrot values are needed per row.
    float* sin_smem = reinterpret_cast<float*>(ptr);
    ptr += Br * halfrot * sizeof(float);

    float* cos_smem = reinterpret_cast<float*>(ptr);

    /*
     * Load cached sin/cos for this batch's actual position_ids.
     * Assumes halfrot is divisible by 4.
     */
    for (
        int i = tid;
        i < Br * (halfrot / 4);
        i += blockDim.x
    ) {
        int row = i / (halfrot / 4);
        int c = (i % (halfrot / 4)) * 4;

        int global_row = tile_id * Br + row;

        if (global_row < seq_len) {
            int position =
                position_ids[batch_id * seq_len + global_row];

            // Prefer validating this in the launcher.
            if (position >= 0 && position < cache_len) {
                int src = position * halfrot + c;
                int dst = row * halfrot + c;

                *reinterpret_cast<float4*>(sin_smem + dst) =
                    *reinterpret_cast<const float4*>(
                        sin_cached + src
                    );

                *reinterpret_cast<float4*>(cos_smem + dst) =
                    *reinterpret_cast<const float4*>(
                        cos_cached + src
                    );
            }
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
        asm volatile("cp.async.commit_group;\n");
        asm volatile(
            "cp.async.wait_group 0;\n"
            ::: "memory"
        );
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
            float x = smem_x[row * row_stride + d];
            float y = x;

            if (d < rotary_dim) {
                int freq_idx;
                float rotated;

                if (d < halfrot) {
                    freq_idx = d;
                    rotated = -smem_x[
                        row * row_stride + d + halfrot
                    ];
                } else {
                    freq_idx = d - halfrot;
                    rotated = smem_x[
                        row * row_stride + d - halfrot
                    ];
                }

                float sn =
                    sin_smem[row * halfrot + freq_idx];

                float cs =
                    cos_smem[row * halfrot + freq_idx];

                y = x * cs + rotated * sn;
            }

            output[
                tensor_base +
                static_cast<int64_t>(global_row) * head_dim +
                d
            ] = static_cast<scalar_t>(y);
        }
    }
}
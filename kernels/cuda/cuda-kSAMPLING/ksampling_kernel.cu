#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


template<int K>
__forceinline__ __device__ void insert_topk(
    const float value,
    const int val_idx,
    float* top,
    int* idx,
    float& min_value,
    int& min_position
) {
    if (value > min_value) {

        top[min_position] = value;
        idx[min_position] = val_idx;

        min_value = top[0];
        min_position = 0;

        #pragma unroll
        for (int j = 1; j < K; ++j) {
            if (top[j] < min_value) {
                min_value = top[j];
                min_position = j;
            }
        }
    }
}


template<int K>
__forceinline__ __device__ void warp_reduce_topk(
    float* top,
    int* idx,
    float& min_value,
    int& min_position
) {
    constexpr unsigned FULL_MASK = 0xffffffff;

    int lane = threadIdx.x & 31;


    for (int delta = 16; delta > 0; delta >>= 1) {

        #pragma unroll
        for (int k = 0; k < K; ++k) {

            float recv_val =
                __shfl_down_sync(
                    FULL_MASK,
                    top[k],
                    delta
                );

            int recv_idx =
                __shfl_down_sync(
                    FULL_MASK,
                    idx[k],
                    delta
                );

            // Only lower lanes merge.
            if (lane < delta) {
                insert_topk<K>(
                    recv_val,
                    recv_idx,
                    top,
                    idx,
                    min_value,
                    min_position
                );
            }
        }
    }
}


template<int K>
__global__ void ksmapling(
    const float* __restrict__ vocab,
    const int max_vocab_size,
    float* __restrict__ out_vals,
    int* __restrict__ out_idxs
) {
    float top[K];
    int idx[K];

    float min_value = -INFINITY;
    int min_position = 0;

    #pragma unroll
    for (int k = 0; k < K; ++k) {
        top[k] = -INFINITY;
        idx[k] = -1;
    }

    const int row = blockIdx.x;
    const int tid = threadIdx.x;


    for (
        int v = tid;
        v < max_vocab_size;
        v += blockDim.x
    ) {
        float val = vocab[
            row * max_vocab_size + v
        ];

        insert_topk<K>(
            val,
            v,
            top,
            idx,
            min_value,
            min_position
        );
    }


    warp_reduce_topk<K>(
        top,
        idx,
        min_value,
        min_position
    );


    const int warp_id = tid / 32;
    const int lane = tid & 31;

    constexpr int NUM_WARPS = 256 / 32;


    extern __shared__ char smem[];

    float* s_vals = reinterpret_cast<float*>(smem);

    int* s_idxs =
        reinterpret_cast<int*>(
            s_vals + NUM_WARPS * K
        );


    if (lane == 0) {

        #pragma unroll
        for (int k = 0; k < K; ++k) {

            s_vals[
                warp_id * K + k
            ] = top[k];

            s_idxs[
                warp_id * K + k
            ] = idx[k];
        }
    }

    __syncthreads();


    if (tid == 0) {

        float final_top[K];
        int final_idx[K];

        float final_min_value = -INFINITY;
        int final_min_position = 0;


        #pragma unroll
        for (int k = 0; k < K; ++k) {
            final_top[k] = -INFINITY;
            final_idx[k] = -1;
        }


        const int total_warp_elements =
            NUM_WARPS * K;


        for (
            int i = 0;
            i < total_warp_elements;
            ++i
        ) {

            insert_topk<K>(
                s_vals[i],
                s_idxs[i],
                final_top,
                final_idx,
                final_min_value,
                final_min_position
            );
        }

        #pragma unroll
        for (int k = 0; k < K; ++k) {

            out_vals[
                row * K + k
            ] = final_top[k];

            out_idxs[
                row * K + k
            ] = final_idx[k];
        }
    }
}



void ksmapling_cuda(
    torch::Tensor vocab,
    int64_t K,
    torch::Tensor out_vals,
    torch::Tensor out_idxs
) {
    const int batch =
        static_cast<int>(vocab.size(0));

    const int V =
        static_cast<int>(vocab.size(1));


    constexpr int THREADS = 256;
    constexpr int NUM_WARPS = THREADS / 32;


    dim3 grid(batch);
    dim3 block(THREADS);


    size_t smem =
        NUM_WARPS * K * sizeof(float)
        +
        NUM_WARPS * K * sizeof(int);


    cudaStream_t stream =
        at::cuda::getDefaultCUDAStream();


    switch (K) {

        case 1:

            ksmapling<1>
                <<<grid, block, smem, stream>>>(
                    vocab.data_ptr<float>(),
                    V,
                    out_vals.data_ptr<float>(),
                    out_idxs.data_ptr<int>()
                );

            break;


        case 3:

            ksmapling<3>
                <<<grid, block, smem, stream>>>(
                    vocab.data_ptr<float>(),
                    V,
                    out_vals.data_ptr<float>(),
                    out_idxs.data_ptr<int>()
                );

            break;


        case 5:

            ksmapling<5>
                <<<grid, block, smem, stream>>>(
                    vocab.data_ptr<float>(),
                    V,
                    out_vals.data_ptr<float>(),
                    out_idxs.data_ptr<int>()
                );

            break;


        case 10:

            ksmapling<10>
                <<<grid, block, smem, stream>>>(
                    vocab.data_ptr<float>(),
                    V,
                    out_vals.data_ptr<float>(),
                    out_idxs.data_ptr<int>()
                );

            break;


        default:

            TORCH_CHECK(
                false,
                "Unsupported K. "
                "Supported K: 1, 3, 5, 10"
            );
    }


    cudaError_t err =
        cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        "CUDA kernel launch failed: ",
        cudaGetErrorString(err)
    );
}
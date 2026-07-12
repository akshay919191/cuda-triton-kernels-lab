#include "../common/common_helper.cuh"  /// common among most of kernels
#include "private_helper.cuh"           /// specific for this 
/// includes needed
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <cmath>
#define PADDING 8
#define FLOAT4(x)  (*reinterpret_cast<float4*>(&(x)))
#define CFLOAT4(x) (*reinterpret_cast<const float4*>(&(x)))
/*
mean of the whole data , wrt to col --- row wise
*/
template<int Br>
__global__ void rmsfwd_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    float eps,
    const __half* __restrict__ gammaGlobal,
    const int seqlen,
    const int headdim,
    const int numhead
)
{
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warpid = tid >> 5; 
    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;
        /// as it is used in attentions in transformers so we need to use batchid and all
    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                                (long long)headid * headdim * seqlen;
    const __half* INptr  = input + base;
    __half* outptr = output + base;
        /// now allocate the mem
    extern __shared__ char smen_raw[];
    char* ptr = smen_raw;
    const int rowStride = headdim + PADDING;   /// real row width padded, used for every Asmem index below
    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );
    /// for A

    __half* Asmem = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    /// for B    for gamma
    __half* gamma = reinterpret_cast<__half*>(ptr);
    ptr += (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    /// due to multi warp , we will store the mean here 
    float* result = reinterpret_cast<float*>(ptr);

    ptr += Br * sizeof(float);

    float* denoSmem = reinterpret_cast<float*>(ptr);   // [Br], one slot per row

    ptr += Br * sizeof(float);
        /// launch with number of rows actually needed so no loop
        /// const int rowitr = tileid;   use tile id instead of new var
    int vec_elems = headdim / 8;  

    for (int i = tid; i < vec_elems; i += blockDim.x) {
        FLOAT4(gamma[i * 8]) = CFLOAT4(gammaGlobal[i * 8]);
    }

    for (int i = vec_elems * 8 + tid; i < headdim; i += blockDim.x) {
        gamma[i] = gammaGlobal[i];
    }
    __syncthreads();
    cpasynccopy<Br>(
        INptr,
        Asmem,
        headdim + PADDING,
        tileid,
        seqlen,
        headdim
    );
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
        /// loaded the whole Br * headim 
        /// now we need that reduce but but but , we need x ** 2 mean not x ; x is the whole row
    multiWarpReductionSUM_RMS(Asmem, result, headdim, rowStride, Br);    //// it gives us sum not mean so divide by total numbers in a row means headdim

    __syncthreads();

    for (int row = tid; row < Br; row += blockDim.x)
        denoSmem[row] = sqrtf(result[row] / static_cast<float>(headdim) + eps);

    __syncthreads(); 
        /// now each row mean is in result , can be accesed by index
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
    int row = i / headdim;
    int col = i % headdim;
    float deno = denoSmem[row];   // shared read, no sqrtf here anymore
            Asmem[row * rowStride + col] = __float2half(
                (__half2float(Asmem[row * rowStride + col]) / deno) * __half2float(gamma[col])
            );
        }
    __syncthreads();
        /// write the normalized tile back to global — same row/col this tile owns,
        /// shifted by tileid * Br to land at the right block of seqlen. vectorized
        /// float4 so it's 8 halves a store instead of 1, same trick as the gamma load.
    for (int i = tid; i < Br * (headdim / 8); i += blockDim.x)
    {
    int row = i / (headdim / 8);
    int c8  = i % (headdim / 8);
    int globalRow = tileid * Br + row;
    if (globalRow >= seqlen) continue;   /// last tile can be partial if seqlen % Br != 0
            FLOAT4(outptr[globalRow * headdim + c8 * 8]) = CFLOAT4(Asmem[row * rowStride + c8 * 8]);
        }
}

template<int Br>
__global__ void rmsbwd_kernel(
    const __half* __restrict__ dl_final_out,
    const __half* __restrict__ input,
    const __half* __restrict__ gammaGlobal,
    __half* __restrict__ dl_dx,
    float* __restrict__ dl_gamma,
    float eps,
    const int seqlen,
    const int headdim,
    const int numhead
)
{
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warpid = tid >> 5; 
    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    extern __shared__ char smem[];
    char* ptr = smem;
    const int rowStride = headdim + PADDING;

    /*
    we need space for DO , dl_final , input , result(for rms deno saving)
    */

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31
    );

    __half* dl_in = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31
    );
    
    __half* dl_final = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31
    );
    
    __half* dl_O = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    //// we need atleast 2-3 element wise multiplication so we will write it down in our helper header
    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );
    /// for B    for gamma
    __half* gamma = reinterpret_cast<__half*>(ptr);

    ptr += (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );
    
    float* result = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    float* summat = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                                (long long)headid * headdim * seqlen;
    const __half* INptr  = input + base;
    const __half* Fptr   = dl_final_out + base;
          __half* out    = dl_dx + base;
          float* gmma_bk= base + dl_gamma;


    int vec_elems = headdim / 8;  // Number of full 8-element chunks

    for (int i = tid; i < vec_elems; i += blockDim.x) {
        FLOAT4(gamma[i * 8]) = CFLOAT4(gammaGlobal[i * 8]);
    }

    for (int i = vec_elems * 8 + tid; i < headdim; i += blockDim.x) {
        gamma[i] = gammaGlobal[i];
    }

    cpasynccopy<Br>(
        INptr,
        dl_in,
        headdim + PADDING,
        tileid,
        seqlen,
        headdim
    );
    asm volatile("cp.async.commit_group;\n");

    cpasynccopy<Br>(
        Fptr,
        dl_final,
        headdim + PADDING,
        tileid,
        seqlen,
        headdim
    );
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();


    multiWarpReductionSUM_RMS(dl_in, result, headdim, rowStride, Br);
    __syncthreads();

    for (int row = tid; row < Br; row += blockDim.x)
        result[row] = sqrtf(result[row] / static_cast<float>(headdim) + eps);

    __syncthreads(); 

    // dgamma[c] += dy[row, c] * x[row, c] / rms[row]
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int smem_idx = r * rowStride + c;

        float dy_val = __half2float(dl_final[smem_idx]);
        float x_val  = __half2float(dl_in[smem_idx]);
        float rms    = result[r];

        atomicAdd(&dl_gamma[c], dy_val * x_val / rms);
    }
    __syncthreads();

    // dl_O = dl_final * gamma
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int idx = r * rowStride + c;

        dl_O[idx] = __float2half(
            __half2float(dl_final[idx]) * __half2float(gamma[c])
        );
    }
    __syncthreads();

    /// now we need only dl/dx  

    //// now d_final hold the summation
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int idx = r * rowStride + c;

        dl_final[idx] = __float2half(
            __half2float(dl_O[idx]) * __half2float(dl_in[idx])
        );
    }
    __syncthreads();

    multiWarpReductionSUM_plain_RMS(dl_final, summat, headdim, rowStride, Br);
    __syncthreads();

    for(int i = tid ; i < Br * headdim ; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        float val = result[r];
        int idx = r * rowStride + c;

        dl_O[idx]  = __float2half(__half2float(dl_O[idx]) / val);
        dl_in[idx] = __float2half(__half2float(dl_in[idx]) / (val * val * val));
    }
    __syncthreads();


    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int smem_idx = r * rowStride + c;

        float a = __half2float(dl_O[smem_idx]);   // dy * gamma / rms
        float b = __half2float(dl_in[smem_idx]);  // x / rms^3

        float s = summat[r] / static_cast<float>(headdim);

        out[globalRow * headdim + c] = __float2half(a - b * s);
    }
}


static inline size_t align32_bytes(size_t x) {
    return (x + 31) & ~size_t(31);
}

template<int Br>
static inline size_t rmsnorm_fwd_smem_bytes(int D) {
    size_t bytes = 0;

    bytes = align32_bytes(bytes);
    bytes += Br * (D + PADDING) * sizeof(__half);

    bytes = align32_bytes(bytes);
    bytes += (D + PADDING) * sizeof(__half);

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);

    bytes += 128;
    return bytes;
}

template<int Br>
static inline size_t rmsnorm_bwd_smem_bytes(int D) {
    size_t bytes = 0;

    bytes = align32_bytes(bytes);
    bytes += Br * (D + PADDING) * sizeof(__half);

    bytes = align32_bytes(bytes);
    bytes += Br * (D + PADDING) * sizeof(__half);

    bytes = align32_bytes(bytes);
    bytes += Br * (D + PADDING) * sizeof(__half);

    bytes = align32_bytes(bytes);
    bytes += (D + PADDING) * sizeof(__half);

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);

    bytes += 128;
    return bytes;
}

template<int Br>
std::vector<torch::Tensor> rmsnorm_forward_launch(
    torch::Tensor x,
    torch::Tensor gamma,
    double eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);

    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat16, "gamma must be float16");

    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    TORCH_CHECK(gamma.size(0) == D, "gamma.size(0) must match D");

    // Keep this because your final global store is still FLOAT4 vectorized.
    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8 because kernel uses FLOAT4 stores");

    auto y = torch::empty_like(x);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = rmsnorm_fwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            rmsfwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    rmsfwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        static_cast<float>(eps),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        N,
        D,
        H
    );

    CUDA_CHECK(cudaGetLastError());

    return {y};
}

template<int Br>
std::vector<torch::Tensor> rmsnorm_backward_launch(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor gamma,
    double eps
) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);

    TORCH_CHECK(dy.scalar_type() == torch::kFloat16, "dy must be float16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat16, "gamma must be float16");

    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    TORCH_CHECK(gamma.size(0) == D, "gamma.size(0) must match D");
    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8 because kernel uses vectorized ops");

    auto dx = torch::empty_like(x);
    auto dgamma = torch::zeros({D}, x.options().dtype(torch::kFloat32));

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = rmsnorm_bwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            rmsbwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    rmsbwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(dy.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(dx.data_ptr<at::Half>()),
        dgamma.data_ptr<float>(),
        static_cast<float>(eps),
        N,
        D,
        H
    );

    CUDA_CHECK(cudaGetLastError());

    return {dx, dgamma};
}

std::vector<torch::Tensor> rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    double eps
) {
    return rmsnorm_forward_launch<16>(x, gamma, eps);
}

std::vector<torch::Tensor> rmsnorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor gamma,
    double eps
) {
    return rmsnorm_backward_launch<16>(dy, x, gamma, eps);
}
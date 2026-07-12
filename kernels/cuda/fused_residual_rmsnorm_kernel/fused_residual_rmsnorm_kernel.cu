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
#define FLOAT4(x)  (*reinterpret_cast<float4*>(&(x)))
#define CFLOAT4(x) (*reinterpret_cast<const float4*>(&(x)))

template<int Br>
__global__ void fusedrmsfwd_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ residual,
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
    const long long base = (long long)batchid * numhead * headdim * seqlen + 
                                (long long)headid * headdim * seqlen;

    const __half* INptr  = input + base;
    const __half* Reptr  = residual + base;
    __half* outptr = output + base;

        /// now allocate the mem
    extern __shared__ char smen_raw[];
    char* ptr = smen_raw;
    const int rowStride = headdim + PADDING;   /// real row width padded, used for every Asmem index below

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* Asmem = reinterpret_cast<__half*>(ptr);
    ptr += Br * (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* res = reinterpret_cast<__half*>(ptr);
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

    cpasynccopy<Br>(
        Reptr,
        res,
        headdim + PADDING,
        tileid,
        seqlen,
        headdim
    );
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    for(int i = tid ; i < Br * headdim ; i += blockDim.x) {
        int r = i / headdim;
        int c = i % headdim;
        int idx = r * rowStride + c;
        Asmem[idx] = __float2half(__half2float(Asmem[idx]) + __half2float(res[idx]));
    }
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
__global__ void fusedrmsbwd_kernel(
    const __half* __restrict__ dl_final_out,
    const __half* __restrict__ input,
    const __half* __restrict__ residual,
    const __half* __restrict__ gammaGlobal,
          __half* __restrict__ dl_dx,
          __half* __restrict__ dl_residual,
          float*  __restrict__ dl_gamma,
    float eps,
    const int seqlen,
    const int headdim,
    const int numhead
)
{
    int tid = threadIdx.x;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    extern __shared__ char smem[];
    char* ptr = smem;

    const int rowStride = headdim + PADDING;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* dl_in = reinterpret_cast<__half*>(ptr);    
    ptr += Br * rowStride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* dl_final = reinterpret_cast<__half*>(ptr);  // dy
    ptr += Br * rowStride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* res = reinterpret_cast<__half*>(ptr);       
    ptr += Br * rowStride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* dl_O = reinterpret_cast<__half*>(ptr);      
    ptr += Br * rowStride * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    __half* gamma = reinterpret_cast<__half*>(ptr);
    ptr += (headdim + PADDING) * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* result = reinterpret_cast<float*>(ptr);     
    ptr += Br * sizeof(float);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 31) & ~31ULL
    );

    float* summat = reinterpret_cast<float*>(ptr);      
    ptr += Br * sizeof(float);

    const long long base =
        (long long)batchid * numhead * seqlen * headdim +
        (long long)headid  * seqlen * headdim;

    const __half* INptr = input + base;
    const __half* Reptr = residual + base;
    const __half* Fptr  = dl_final_out + base;

    __half* out    = dl_dx + base;
    __half* resout = dl_residual + base;

    // load gamma
    int vec_elems = headdim / 8;

    for (int i = tid; i < vec_elems; i += blockDim.x) {
        FLOAT4(gamma[i * 8]) = CFLOAT4(gammaGlobal[i * 8]);
    }

    for (int i = vec_elems * 8 + tid; i < headdim; i += blockDim.x) {
        gamma[i] = gammaGlobal[i];
    }

    // load x
    cpasynccopy<Br>(
        INptr,
        dl_in,
        rowStride,
        tileid,
        seqlen,
        headdim
    );
    asm volatile("cp.async.commit_group;\n");

    // load residual
    cpasynccopy<Br>(
        Reptr,
        res,
        rowStride,
        tileid,
        seqlen,
        headdim
    );
    asm volatile("cp.async.commit_group;\n");

    // load dy
    cpasynccopy<Br>(
        Fptr,
        dl_final,
        rowStride,
        tileid,
        seqlen,
        headdim
    );
    asm volatile("cp.async.commit_group;\n");

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    // dl_in = z = x + residual
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int idx = r * rowStride + c;

        float x_val = __half2float(dl_in[idx]);
        float r_val = __half2float(res[idx]);

        dl_in[idx] = __float2half(x_val + r_val);
    }
    __syncthreads();

    // result = sum(z^2)
    multiWarpReductionSUM_RMS(dl_in, result, headdim, rowStride, Br);
    __syncthreads();

    // result = rms
    for (int row = tid; row < Br; row += blockDim.x) {
        result[row] = sqrtf(result[row] / static_cast<float>(headdim) + eps);
    }
    __syncthreads();

    // dgamma[c] += dy * z / rms
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int idx = r * rowStride + c;

        float dy_val = __half2float(dl_final[idx]);
        float z_val  = __half2float(dl_in[idx]);
        float rms    = result[r];

        atomicAdd(&dl_gamma[c], dy_val * z_val / rms);
    }
    __syncthreads();

    // dl_O = delta = dy * gamma
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int idx = r * rowStride + c;

        float dy_val = __half2float(dl_final[idx]);
        float g_val  = __half2float(gamma[c]);

        dl_O[idx] = __float2half(dy_val * g_val);
    }
    __syncthreads();

    // dl_final = delta * z
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int idx = r * rowStride + c;

        float delta = __half2float(dl_O[idx]);
        float z_val = __half2float(dl_in[idx]);

        dl_final[idx] = __float2half(delta * z_val);
    }
    __syncthreads();

    // summat = sum(delta * z)
    multiWarpReductionSUM_plain_RMS(dl_final, summat, headdim, rowStride, Br);
    __syncthreads();

    // dl_O = delta / rms
    // dl_in = z / rms^3
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int idx = r * rowStride + c;

        float rms = result[r];

        float delta = __half2float(dl_O[idx]);
        float z_val = __half2float(dl_in[idx]);

        dl_O[idx]  = __float2half(delta / rms);
        dl_in[idx] = __float2half(z_val / (rms * rms * rms));
    }
    __syncthreads();

    // dz = delta/rms - z/rms^3 * mean(delta*z)
    // dx = dz
    // dresidual = dz
    for (int i = tid; i < Br * headdim; i += blockDim.x)
    {
        int r = i / headdim;
        int c = i % headdim;

        int globalRow = tileid * Br + r;
        if (globalRow >= seqlen) continue;

        int idx = r * rowStride + c;

        float a = __half2float(dl_O[idx]);   // delta / rms
        float b = __half2float(dl_in[idx]);  // z / rms^3

        float s = summat[r] / static_cast<float>(headdim);

        float dz = a - b * s;

        out[globalRow * headdim + c]    = __float2half(dz);
        resout[globalRow * headdim + c] = __float2half(dz);
    }
}

static inline size_t align32_bytes(size_t x) {
    return (x + 31) & ~size_t(31);
}

template<int Br>
static inline size_t fused_rms_fwd_smem_bytes(int D) {
    size_t bytes = 0;

    const int rowStride = D + PADDING;

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // Asmem: x, later z/output

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // res: residual

    bytes = align32_bytes(bytes);
    bytes += (D + PADDING) * sizeof(__half);   // gamma

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // result

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // denoSmem

    bytes += 128;
    return bytes;
}

template<int Br>
static inline size_t fused_rms_bwd_smem_bytes(int D) {
    size_t bytes = 0;

    const int rowStride = D + PADDING;

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // dl_in: z = x + residual

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // dl_final: dy

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // res: residual temp

    bytes = align32_bytes(bytes);
    bytes += Br * rowStride * sizeof(__half);  // dl_O: delta

    bytes = align32_bytes(bytes);
    bytes += (D + PADDING) * sizeof(__half);   // gamma

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // result/rms

    bytes = align32_bytes(bytes);
    bytes += Br * sizeof(float);               // summat

    bytes += 128;
    return bytes;
}

template<int Br>
std::vector<torch::Tensor> fused_residual_rmsnorm_forward_launch(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    double eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(residual);
    CHECK_INPUT(gamma);

    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "residual must be float16");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat16, "gamma must be float16");

    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    TORCH_CHECK(gamma.size(0) == D, "gamma.size(0) must match D");
    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8 because kernel uses FLOAT4 vectorized stores");

    auto y = torch::empty_like(x);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = fused_rms_fwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            fusedrmsfwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    fusedrmsfwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
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
std::vector<torch::Tensor> fused_residual_rmsnorm_backward_launch(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    double eps
) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    CHECK_INPUT(residual);
    CHECK_INPUT(gamma);

    TORCH_CHECK(dy.scalar_type() == torch::kFloat16, "dy must be float16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "residual must be float16");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat16, "gamma must be float16");

    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [D]");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);

    TORCH_CHECK(gamma.size(0) == D, "gamma.size(0) must match D");
    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8 because kernel uses vectorized ops");

    auto dx = torch::empty_like(x);
    auto dresidual = torch::empty_like(x);
    auto dgamma = torch::zeros({D}, x.options().dtype(torch::kFloat32));

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(256);

    size_t smem = fused_rms_bwd_smem_bytes<Br>(D);

    if (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            fusedrmsbwd_kernel<Br>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
    }

    fusedrmsbwd_kernel<Br><<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(dy.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(dx.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(dresidual.data_ptr<at::Half>()),
        dgamma.data_ptr<float>(),
        static_cast<float>(eps),
        N,
        D,
        H
    );

    CUDA_CHECK(cudaGetLastError());

    return {dx, dresidual, dgamma};
}


std::vector<torch::Tensor> fused_residual_rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    double eps
) {
    return fused_residual_rmsnorm_forward_launch<16>(x, residual, gamma, eps);
}


std::vector<torch::Tensor> fused_residual_rmsnorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    double eps
) {
    return fused_residual_rmsnorm_backward_launch<16>(dy, x, residual, gamma, eps);
}
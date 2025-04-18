#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "normalize.cuh"

constexpr double NS_A_CONST = 3.4445;
constexpr double NS_B_CONST = -4.7750;
constexpr double NS_C_CONST = 2.0315;

// Kernels ---------------------------------------------------------------------

template <typename scalar_t>
__global__ void update_B_kernel(const scalar_t *__restrict__ A_in,   // (K, K)
                                const scalar_t *__restrict__ A_2_in, // (K, K)
                                scalar_t *__restrict__ B_out,        // (K, K)
                                const int K, // K = min(M, N)
                                const scalar_t ns_B, const scalar_t ns_C) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride_col = gridDim.x * blockDim.x;
  const int stride_row = gridDim.y * blockDim.y;

  for (int current_row = row; current_row < K; current_row += stride_row) {
    for (int current_col = col; current_col < K; current_col += stride_col) {
      int tid = current_row * K + current_col;
      B_out[tid] = ns_B * A_in[tid] + ns_C * A_2_in[tid];
    }
  }
}

template <typename scalar_t>
__global__ void update_X_kernel(const scalar_t *__restrict__ X_in,   // (K, L)
                                const scalar_t *__restrict__ B_X_in, // (K, L)
                                scalar_t *__restrict__ X_out,        // (K, L)
                                const int K, // K = min(M, N)
                                const int L, // L = max(M, N)
                                const scalar_t ns_A) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride_col = gridDim.x * blockDim.x;
  const int stride_row = gridDim.y * blockDim.y;

  for (int current_row = row; current_row < K; current_row += stride_row) {
    for (int current_col = col; current_col < L; current_col += stride_col) {
      int tid = current_row * L + current_col;
      X_out[tid] = ns_A * X_in[tid] + B_X_in[tid];
    }
  }
}

// Launchers -------------------------------------------------------------------

template <typename scalar_t>
void update_B_launcher(const scalar_t *__restrict__ A_ptr,   // (K, K)
                       const scalar_t *__restrict__ A_2_ptr, // (K, K)
                       scalar_t *B_ptr,                      // (K, K)
                       int K,                                // K = min(M, N)
                       cudaStream_t stream) {
  if (K == 0)
    return;

  const scalar_t ns_B = static_cast<scalar_t>(NS_B_CONST);
  const scalar_t ns_C = static_cast<scalar_t>(NS_C_CONST);

  dim3 blockDim(16, 16);
  dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
               (K + blockDim.y - 1) / blockDim.y);

  update_B_kernel<scalar_t>
      <<<gridDim, blockDim, 0, stream>>>(A_ptr, A_2_ptr, B_ptr, K, ns_B, ns_C);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel launch error (update_B_kernel): ") +
        cudaGetErrorString(launch_err));
  }
}

template <typename scalar_t>
void update_X_launcher(const scalar_t *X_in_ptr, // (K, L)
                       const scalar_t *B_X_ptr,  // (K, L)
                       scalar_t *X_output_ptr,   // (K, L)
                       int K,                    // K = min(M, N)
                       int L,                    // L = max(M, N)
                       cudaStream_t stream) {
  if (K == 0 || L == 0)
    return;

  const scalar_t ns_A = static_cast<scalar_t>(NS_A_CONST);

  dim3 blockDim(16, 16);
  dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
               (K + blockDim.y - 1) / blockDim.y);

  update_X_kernel<scalar_t><<<gridDim, blockDim, 0, stream>>>(
      X_in_ptr, B_X_ptr, X_output_ptr, K, L, ns_A);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel launch error (update_X_kernel): ") +
        cudaGetErrorString(launch_err));
  }
}

#define DECLARE_B_LAUNCHER(type)                                               \
  template void update_B_launcher<type>(const type *, const type *, type *,    \
                                        int, cudaStream_t);

#define DECLARE_X_LAUNCHER(type)                                               \
  template void update_X_launcher<type>(const type *, const type *, type *,    \
                                        int, int, cudaStream_t);

#define DECLARE_LAUNCHERS(type)                                                \
  DECLARE_B_LAUNCHER(type)                                                     \
  DECLARE_X_LAUNCHER(type)

DECLARE_LAUNCHERS(double)
DECLARE_LAUNCHERS(float)
DECLARE_LAUNCHERS(c10::BFloat16)
DECLARE_LAUNCHERS(c10::Half)

// Newton Schulz ---------------------------------------------------------------

torch::Tensor newton_schulz(const torch::Tensor &G, const int ns_steps = 5) {
  TORCH_CHECK(G.is_cuda(), "Input tensor G must be a CUDA tensor.");
  TORCH_CHECK(G.dim() == 2, "Input tensor G must be 2 dimensional.")

  const int M = G.size(0);
  const int N = G.size(1);
  TORCH_CHECK(M > 0, "Input dim M must be positive.")
  TORCH_CHECK(N > 0, "Input dim N must be positive.")

  auto G_cont = G.contiguous();
  auto options = G_cont.options();

  torch::Tensor X_current;
  bool needs_transpose = false;
  if (M > N) {
    X_current = G_cont.t().contiguous();
    needs_transpose = true;
  } else {
    X_current = G_cont;
  }

  const int K = X_current.size(0);
  const int L = X_current.size(1);

  normalize_(X_current);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto A = torch::empty({K, K}, G_cont.options());
  auto A_2 = torch::empty({K, K}, G_cont.options());
  auto B = torch::empty({K, K}, G_cont.options());
  auto B_X = torch::empty({K, L}, G_cont.options());
  auto X_next = torch::empty({K, L}, G_cont.options());

  auto X_current_T = torch::empty({L, K}, options);

  for (auto i = 0; i < ns_steps; ++i) {
    X_current_T = X_current.t().contiguous();
    torch::matmul_out(A, X_current, X_current_T);

    torch::matmul_out(A_2, A, A);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        options.dtype().toScalarType(), "update_B_dispatch", [&] {
          update_B_launcher<scalar_t>(A.data_ptr<scalar_t>(),
                                      A_2.data_ptr<scalar_t>(),
                                      B.data_ptr<scalar_t>(), K, stream);
        });

    torch::matmul_out(B_X, B, X_current);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        options.dtype().toScalarType(), "update_X_dispatch", [&] {
          update_X_launcher<scalar_t>(
              X_current.data_ptr<scalar_t>(), B_X.data_ptr<scalar_t>(),
              X_next.data_ptr<scalar_t>(), K, L, stream);
        });

    X_current.copy_(X_next, true);
  }

  if (needs_transpose) {
    X_current = X_current.t().contiguous();
  }

  return X_current;
}

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// #include "newton_schulz.cuh"
#include "normalize.cuh"

#define CUDA_CHECK_HOST(call)                                                                                                                        \
  do {                                                                                                                                               \
    cudaError_t err = call;                                                                                                                          \
    if (err != cudaSuccess) {                                                                                                                        \
      throw std::runtime_error(std::string("CUDA Error at ") + __FILE__ + ":" + std::to_string(__LINE__) + " - Code: " + std::to_string(err) +       \
                               " (" + cudaGetErrorString(err) + ")");                                                                                \
    }                                                                                                                                                \
  } while (0)

void normalize_bf16_cuda(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor.");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input tensor must be BFloat16.");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");
  TORCH_CHECK(input.is_non_overlapping_and_dense(), "In-place requires non-overlapping and dense tensor.");

  const int64_t N = input.numel();
  if (N == 0) {
    return;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int reduce1_block_dim = 256;
  const int reduce1_vec_size = 8;
  const int elements_per_thread_approx = 8;
  int reduce1_grid_dim = std::min((1 << 16), div_ceil(N, reduce1_block_dim * elements_per_thread_approx * reduce1_vec_size));
  reduce1_grid_dim = std::max(1, reduce1_grid_dim);

  torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
  torch::Tensor partial_sums = torch::empty({reduce1_grid_dim}, float_options);
  const size_t reduce1_shared_mem_size = reduce1_block_dim * sizeof(float);

  reduce_sum_squares_partial_kernel_launcher(reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()), partial_sums.data_ptr<float>(),
                                             N, reduce1_grid_dim, reduce1_block_dim, reduce1_shared_mem_size, stream);
  CUDA_CHECK_HOST(cudaGetLastError());

  const int n_partial_sums = reduce1_grid_dim;
  torch::Tensor total_sum_squared = torch::empty({1}, float_options);
  if (n_partial_sums > 0) {
    const int MAX_REDUCE2_BLOCK_DIM = 1024;
    int reduce2_block_dim = std::min(MAX_REDUCE2_BLOCK_DIM, div_ceil(n_partial_sums, 1));
    size_t reduce2_shared_mem_size = reduce2_block_dim * sizeof(float);

    reduce_final_sum_kernel_launcher(partial_sums.data_ptr<float>(), total_sum_squared.data_ptr<float>(), n_partial_sums, 1, reduce2_block_dim,
                                     reduce2_shared_mem_size, stream);
    CUDA_CHECK_HOST(cudaGetLastError());
  } else {
    CUDA_CHECK_HOST(cudaMemsetAsync(total_sum_squared.data_ptr(), 0, sizeof(float), stream));
  }

  const int scale_block_dim = 256;
  const int scale_vec_size = 8;
  int scale_grid_dim = std::min((1 << 16), div_ceil(N, scale_block_dim * scale_vec_size));
  scale_grid_dim = std::max(1, scale_grid_dim);

  scale_by_inv_norm_inplace_kernel_launcher(reinterpret_cast<__nv_bfloat16 *>(input.data_ptr<at::BFloat16>()), total_sum_squared.data_ptr<float>(), N,
                                            scale_grid_dim, scale_block_dim, stream);
  CUDA_CHECK_HOST(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("newton_schulz", &newton_schulz, "Newton-Schulz iteration", py::arg("G"), py::arg("ns_steps") = 5);
  m.def("normalize_", &normalize_bf16_cuda, "RMS Normalization", py::arg("X"));
  // m.def("normalize_", &normalize_, "RMS Normalization.", py::arg("X"));
}

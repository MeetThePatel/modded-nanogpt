#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

torch::Tensor fused_muon(torch::Tensor G, int ns_steps) {
  TORCH_CHECK(G.is_cuda(), "Gradient tensor must be on device.");
  TORCH_CHECK(G.dim() >= 2, "Gradient tensor must be at least 2-dimensional.");

  size_t M = G.size(-2);
  size_t N = G.size(-1);

  size_t batch_size = 1;
  for (auto i = 0; i < G.dim() - 2; i++) {
    batch_size *= G.size(i);
  }

  auto output = torch::empty_like(G);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(G.scalar_type(), "fused_newton_schulz_cuda", ([&] {
                               fused_muon_launcher<scalar_t>(
                                   G.data_ptr<scalar_t>(),
                                   output.data_ptr<scalar_t>(), batch_size, M,
                                   N, ns_steps, stream);
                             }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_newton_schulz", &fused_muon, "CUDA kernel for NS iteration.");
}

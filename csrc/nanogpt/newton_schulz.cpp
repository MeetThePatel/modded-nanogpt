#include <torch/extension.h>

extern void
newton_schulz_kernel(const float* G_d, float* X_d, int rows, int cols);

torch::Tensor
newton_schulz(torch::Tensor G)
{
  TORCH_CHECK(G.is_cuda(), "newton_schulz: input tensor must be a CUDA tensor.")
  TORCH_CHECK(G.scalar_type() == torch::kFloat32,
              "newton_schulz: input tensor must be float32.")
  G = G.contiguous();

  auto X = G.clone();

  int rows = G.size(-2);
  int cols = G.size(-1);

  newton_schulz_kernel(G.data_ptr<float>(), X.data_ptr<float>(), rows, cols);

  return X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("newton_schulz", &newton_schulz, "Newton-Schulz CUDA Extension.");
}
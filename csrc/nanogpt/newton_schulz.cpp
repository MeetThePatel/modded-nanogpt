#include <torch/extension.h>

torch::Tensor newton_schulz_cuda(const torch::Tensor &G, const int ns_steps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("newton_schulz", &newton_schulz_cuda, "NS Iteration", py::arg("G"),
        py::arg("ns_steps") = 5);
}

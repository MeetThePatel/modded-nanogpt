#include <torch/extension.h>

torch::Tensor newton_schulz_cuda(const torch::Tensor &G, const int ns_steps);

void _fused_muon_kernel_cuda_(at::TensorList params, at::TensorList grads, at::TensorList momentum_buffer_list, const double momentum,
                              const double lr, const int ns_steps, const bool is_first_step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("newton_schulz", &newton_schulz_cuda, "Newton-Schulz iteration", py::arg("G"), py::arg("ns_steps") = 5);
    m.def("_fused_muon_cuda", &_fused_muon_kernel_cuda_, "Fused Muon Step", py::arg("params"), py::arg("grads"), py::arg("momentum_buffer_list"),
          py::arg("momentum"), py::arg("lr"), py::arg("ns_steps"), py::arg("is_first_step"));
}

#include <torch/extension.h>

torch::Tensor newton_schulz_dispatch(const torch::Tensor &G, int ns_steps);

void fused_muon_kernel_cuda(const std::vector<at::Tensor> &params, const std::vector<at::Tensor> &grads,
                            const std::vector<at::Tensor> &momentum_buffer_list, const double momentum, const double lr, const int ns_steps,
                            const bool is_first_step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("newton_schulz", &newton_schulz_dispatch, "Newton-Schulz iteration", py::arg("G"), py::arg("ns_steps") = 5);
    m.def("_fused_muon_cuda", &fused_muon_kernel_cuda, "Fused Muon Step", py::arg("params"), py::arg("grads"), py::arg("momentum_buffer_list"),
          py::arg("momentum"), py::arg("lr"), py::arg("ns_steps"), py::arg("is_first_step"));
}

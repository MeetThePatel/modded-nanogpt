#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "newton_schulz.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("newton_schulz", &newton_schulz, "Newton-Schulz iteration", py::arg("G"), py::arg("ns_steps") = 5); }

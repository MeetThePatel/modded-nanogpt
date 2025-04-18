#include <torch/extension.h>

#include "newton_schulz.cuh"
#include "normalize.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("newton_schulz", &newton_schulz, "Newton-Schulz iteration",
        py::arg("G"), py::arg("ns_steps") = 5);
  m.def("normalize_", &normalize_, "RMS Normalization.", py::arg("X"));
}

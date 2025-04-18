#pragma once

#include <torch/extension.h>

torch::Tensor newton_schulz(const torch::Tensor &G, const int ns_steps = 5);

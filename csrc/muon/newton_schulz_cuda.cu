#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/layout.h>
#include <torch/extension.h>

constexpr float NS_A = 3.4445;
constexpr float NS_B = -4.7750;
constexpr float NS_C = 2.0315;

torch::Tensor newton_schulz_cuda(const torch::Tensor &G, const int ns_steps) {
    TORCH_CHECK(G.is_cuda(), "G must be a CUDA tensor.");
    TORCH_CHECK(G.is_contiguous(), "G must be contiguous.");
    TORCH_CHECK(G.scalar_type() == torch::kBFloat16, "G must have dtype bfloat16.");

    auto X = G;
    bool flipped = false;

    int B = (X.dim() == 3) ? X.size(0) : 1;
    int M = X.size(-2);
    int N = X.size(-1);

    if (X.dim() == 2) {
        X = X.unsqueeze(0);
        B = 1;
    }

    if (M > N) {
        X = X.transpose(-2, -1).contiguous();
        std::swap(M, N);
        flipped = true;
    }

    // Normalization
    auto norm = X.to(torch::kFloat32).pow(2).sum({-2, -1}, true).add(1e-7f).sqrt();
    X = X.to(torch::kFloat32).div(norm).to(torch::kBFloat16);

    for (int i = 0; i < ns_steps; ++i) {
        auto A = at::bmm(X, X.transpose(1, 2));
        auto C = at::bmm(A, A);

        torch::Tensor B = A.mul(NS_B).add(C.mul(NS_C));

        torch::Tensor BX = at::bmm(B, X);

        X = X.mul(NS_A).add(BX);
    }

    if (flipped) {
        X = X.transpose(-2, -1).contiguous();
    }
    return (B == 1) ? X.squeeze(0) : X;
}

torch::Tensor newton_schulz_dispatch(const torch::Tensor &G, int ns_steps) {
    TORCH_CHECK(G.is_cuda(), "G must be a CUDA tensor.");
    TORCH_CHECK(G.is_contiguous(), "G must be contiguous.");
    TORCH_CHECK(G.scalar_type() == torch::kBFloat16, "G must have dtype bfloat16.");

    if (G.dim() == 2) {
        // Shape (M, N)
        return newton_schulz_cuda(G.unsqueeze(0), ns_steps).squeeze(0);
    } else if (G.dim() == 3) {
        // Shape (H, M, N)
        return newton_schulz_cuda(G, ns_steps);
    } else if (G.dim() == 4) {
        // Conv filter: (O, I, kH, kW) → flatten to (O, I * kH * kW)
        int O = G.size(0);
        auto G_flat = G.view({O, -1});                                   // [O, I*kH*kW]
        auto result = newton_schulz_cuda(G_flat.unsqueeze(0), ns_steps); // shape [1, O, D]
        return result.squeeze(0).flatten();                              // flatten like original Python
    } else {
        TORCH_CHECK(false, "Unsupported tensor dimensionality: ", G.dim());
    }
}

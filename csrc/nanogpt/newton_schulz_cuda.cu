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
    // Assume that G is already normalized.
    TORCH_CHECK(G.is_cuda(), "G must be a CUDA tensor.");
    TORCH_CHECK(G.is_contiguous(), "G must be contiguous.");
    TORCH_CHECK(G.scalar_type() == torch::kBFloat16, "G must have dtype bfloat16.");

    auto X = G;

    int M = G.size(-2);
    int N = G.size(-1);

    bool flipped = false;
    if (M > N) {
        X = X.transpose(-2, -1).contiguous();
        std::swap(M, N);
        flipped = true;
    }

    auto sq = X.to(torch::kFloat32).pow(2).sum({-2, -1}, true);
    auto denom = sq.add(1e-7f).sqrt();
    X = X.to(torch::kFloat32).div(denom).to(torch::kBFloat16);

    auto opts = X.options();

    torch::Tensor A = torch::empty({M, M}, opts);
    torch::Tensor C = torch::empty({M, M}, opts);
    torch::Tensor X_new = torch::empty_like(X);

    for (int i = 0; i < ns_steps; ++i) {
        A = at::mm(X, X.transpose(-2, -1));
        C = at::mm(A, A);

        torch::Tensor B = A.mul(NS_B).add_(C.mul(NS_C));

        torch::Tensor BX = at::mm(B, X);
        X_new = X.mul(NS_A).add_(BX);

        X = X_new;
    }

    if (flipped) {
        X = X.transpose(-2, -1).contiguous();
    }
    return X;
}

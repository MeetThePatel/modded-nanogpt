import torch
import torch.utils.benchmark as benchmark
from nanogpt_kernels import newton_schulz

from nanogpt.muon.newton_schulz import newton_schulz as reference_newton_schulz


if __name__ == '__main__':
    for dtype in [torch.float32, torch.bfloat16, torch.half]:
        G = torch.randn(128, 256, device='cuda', dtype=dtype)
        X = newton_schulz(G, 5)
        print(X.dtype)

        X_reference = reference_newton_schulz(G, 5).to(torch.float32)
        print(X_reference.dtype)

        torch.testing.assert_close(X, X_reference)
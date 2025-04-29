import lovely_tensors as lt

import torch
from torch.nn import functional as F
import torch.utils.benchmark as benchmark

from nanogpt_kernels import newton_schulz
from nanogpt.muon.newton_schulz import newton_schulz as reference_newton_schulz


lt.monkey_patch()
torch._dynamo.config.cache_size_limit = 64


def test_my_kernel(shape, dtype):
    G = torch.randn(shape, device="cuda", dtype=dtype)
    G = F.normalize(G, dim=(0, 1), p=2)
    newton_schulz(G)


def test_torch_compile(shape, dtype):
    G = torch.randn(shape, device="cuda", dtype=dtype)
    G = F.normalize(G, dim=(0, 1), p=2)
    reference_newton_schulz(G, 5)


if __name__ == "__main__":
    # speed
    # for shape in [(128, 256), (512, 512), (1024, 2048), (2048, 1024)]:
    # for dtype in [torch.bfloat16]:
    #     for shape in [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)]:
    #         time_my_kernel = (
    #             benchmark.Timer(
    #                 stmt="test_my_kernel(shape, dtype)",
    #                 setup="from __main__ import test_my_kernel",
    #                 globals={"shape": shape, "dtype": dtype},
    #             )
    #             .timeit(1000)
    #             .mean
    #             * 1e6
    #         )
    #         time_torch_compile = (
    #             benchmark.Timer(
    #                 stmt="test_torch_compile(shape, dtype)",
    #                 setup="from __main__ import test_torch_compile",
    #                 globals={"shape": shape, "dtype": dtype},
    #             )
    #             .timeit(1000)
    #             .mean
    #             * 1e6
    #         )

    #         speedup = time_torch_compile / time_my_kernel

    #         print(f"Shape: {shape} \t dtype: {dtype}")
    #         print(f"torch.compile: {time_torch_compile:0.4f} (µs)")
    #         print(f"my kernels: {time_my_kernel:0.4f} (µs)")
    #         print(f"Speedup: {speedup: 0.4f} x")

    #         print("\n")

    # # condition number
    # for dtype in [torch.float32, torch.half, torch.bfloat16]:
    #     for shape in [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)]:
    #         G = torch.randn(shape, device="cuda", dtype=dtype)
    #         X: torch.Tensor = newton_schulz(G, 5).to(torch.float32)
    #         reference_X = reference_newton_schulz(G, 5).to(torch.float32)
    #         G = G.to(torch.float32)

    #         print(f"Shape: {shape} \t dtype: {dtype}")
    #         print(f"G: cond = {torch.linalg.cond(G): 0.3f}")
    #         print(f"X: cond = {torch.linalg.cond(X): 0.3f}")
    #         print(f"X_reference: cond = {torch.linalg.cond(reference_X): 0.3f}")

    G = torch.randn((1024, 1024), dtype=torch.bfloat16, device="cuda")
    G = F.normalize(G, p=2, dim=(0, 1))
    print(f"Starting: {G}")

    reference_result = reference_newton_schulz(G, 5)
    print(f"reference: {reference_result}")

    my_result = newton_schulz(G)
    print(f"my: {my_result}")

    print(f"diff: {(reference_result - my_result)}")

    torch.testing.assert_close(G, reference_result)

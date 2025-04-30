import lovely_tensors as lt

import torch

from muon import newton_schulz
from nanogpt.muon.newton_schulz import newton_schulz as reference_newton_schulz


lt.monkey_patch()
torch._dynamo.config.cache_size_limit = 64


def test_my_kernel(shape, dtype):
    G = torch.randn(shape, device="cuda", dtype=dtype)
    _ = newton_schulz(G, 5)


def test_torch_compile(shape, dtype):
    G = torch.randn(shape, device="cuda", dtype=dtype)
    _ = reference_newton_schulz(G, 5)


if __name__ == "__main__":
    # # Speed
    # for shape in [(512, 512), (1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)]:
    #     time_my_kernel = (
    #         benchmark.Timer(
    #             stmt="test_my_kernel(shape, dtype)",
    #             setup="from __main__ import test_my_kernel",
    #             globals={"shape": shape, "dtype": torch.bfloat16},
    #         )
    #         .timeit(1000)
    #         .mean
    #         * 1e6
    #     )
    #     time_torch_compile = (
    #         benchmark.Timer(
    #             stmt="test_torch_compile(shape, dtype)",
    #             setup="from __main__ import test_torch_compile",
    #             globals={"shape": shape, "dtype": torch.bfloat16},
    #         )
    #         .timeit(1000)
    #         .mean
    #         * 1e6
    #     )

    #     speedup = time_torch_compile / time_my_kernel

    #     print(f"Shape: {shape} \t dtype: {torch.bfloat16}")
    #     print(f"torch.compile: {time_torch_compile:0.4f} (µs)")
    #     print(f"my kernels: {time_my_kernel:0.4f} (µs)")
    #     print(f"Speedup: {speedup: 0.4f} x")

    #     print("\n")

    # condition number
    for dtype in [torch.bfloat16]:
        for shape in [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)]:
            G = torch.randn(shape, device="cuda", dtype=dtype)

            X = newton_schulz(G, 5).to(torch.float32)
            X_ref = reference_newton_schulz(G, 5).to(torch.float32)

            print(f"Shape: {shape} \t dtype: {dtype}")
            print(f"G: cond = {torch.linalg.cond(G.to(torch.float32)): 0.3f}")
            print(f"X: cond = {torch.linalg.cond(X): 0.3f}")
            print(f"X_reference: cond = {torch.linalg.cond(X_ref): 0.3f}")

    # # Correctness
    # G = torch.randn((1024, 1024), dtype=torch.bfloat16, device="cuda")
    # print(f"Starting: {G}")

    # reference_result = reference_newton_schulz(G, 5)
    # print(f"reference: {reference_result}")

    # my_result = newton_schulz(G, 5)
    # print(f"my: {my_result}")

    # print(f"diff: {(reference_result - my_result)}")

    # torch.testing.assert_close(my_result, reference_result)

import lovely_tensors as lt

import torch

from muon import newton_schulz
from nanogpt.muon.newton_schulz import newton_schulz as reference_newton_schulz
from torch.utils import benchmark


lt.monkey_patch()
torch._dynamo.config.cache_size_limit = 64


def generate_tensor(shape, dtype=torch.bfloat16, device="cuda"):
    """Handles 2D, 3D, 4D tensor generation with valid sizes."""
    if len(shape) == 2:
        return torch.randn(shape, dtype=dtype, device=device).contiguous()
    elif len(shape) == 3:
        return torch.randn(shape, dtype=dtype, device=device).contiguous()
    elif len(shape) == 4:
        # Flatten [O, I, kH, kW] to [O, -1]
        O = shape[0]  # noqa: E741
        return torch.randn(shape, dtype=dtype, device=device).contiguous().view(O, -1)
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def test_my_kernel(shape, dtype):
    G = generate_tensor(shape, dtype)
    return newton_schulz(G, 5)


def test_torch_compile(shape, dtype):
    G = generate_tensor(shape, dtype)
    return reference_newton_schulz(G, 5)


if __name__ == "__main__":
    test_shapes = [
        (512, 512),
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
        (2048, 2048),
        (16, 64, 128),  # 3D: batch of matrices
        (64, 64, 64),  # 3D: square batch
        (32, 16, 32, 32),  # 4D conv filters: [O, I, kH, kW]
        (128, 3, 3, 3),  # 4D conv filters
    ]

    # Speed Benchmark
    for shape in test_shapes:
        time_my_kernel = (
            benchmark.Timer(
                stmt="test_my_kernel(shape, dtype)",
                setup="from __main__ import test_my_kernel",
                globals={"shape": shape, "dtype": torch.bfloat16},
            )
            .timeit(100)
            .mean
            * 1e6
        )
        time_torch_compile = (
            benchmark.Timer(
                stmt="test_torch_compile(shape, dtype)",
                setup="from __main__ import test_torch_compile",
                globals={"shape": shape, "dtype": torch.bfloat16},
            )
            .timeit(100)
            .mean
            * 1e6
        )

        speedup = time_torch_compile / time_my_kernel
        print(f"Shape: {shape} \t dtype: {torch.bfloat16}")
        print(f"torch.compile: {time_torch_compile:0.4f} (µs)")
        print(f"my kernel   : {time_my_kernel:0.4f} (µs)")
        print(f"Speedup     : {speedup:0.4f} x\n")

    # Condition Number Check
    for shape in test_shapes:
        G = generate_tensor(shape, dtype=torch.bfloat16)
        X = newton_schulz(G, 5).to(torch.float32)
        X_ref = reference_newton_schulz(G, 5).to(torch.float32)

        print(f"Shape: {shape} \t dtype: bfloat16")

        if X.ndim == 2:
            print(f"G: cond = {torch.linalg.cond(G.to(torch.float32)): 0.3f}")
            print(f"X: cond = {torch.linalg.cond(X.to(torch.float32)): 0.3f}")
            print(f"X_ref: cond = {torch.linalg.cond(X_ref.to(torch.float32)): 0.3f}")
        elif X.ndim == 3:
            print(f"Mean cond (X): {torch.linalg.cond(X[0].to(torch.float32))} ... (just first slice)")
        print()

    # Correctness on a single shape
    G = generate_tensor((1024, 1024), dtype=torch.bfloat16)
    print(f"Sample Input G:\n{G}")

    reference_result = reference_newton_schulz(G, 5)
    print(f"\nReference Output:\n{reference_result}")

    my_result = newton_schulz(G, 5)
    print(f"\nMy Output:\n{my_result}")

    print(f"\nDiff:\n{(reference_result - my_result).abs().max()}")
    torch.testing.assert_close(my_result, reference_result, atol=1e-1, rtol=1e-2)

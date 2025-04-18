import torch
from torch.nn import functional as F
import torch.utils.benchmark as benchmark
from nanogpt_kernels import normalize_

torch._dynamo.config.cache_size_limit = 64


def test_my_kernel(shape, dtype):
    X = torch.randn(shape, dtype=dtype, device="cuda")
    normalize_(X)


@torch.compile
def test_torch(shape, dtype):
    X = torch.randn(shape, dtype=dtype, device="cuda")
    F.normalize(X, dim=(0, 1))


if __name__ == "__main__":
    for dtype in [torch.float32, torch.bfloat16]:
        for shape in [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)]:
            my_kernel_time_us = (
                benchmark.Timer(
                    stmt="test_my_kernel(shape, dtype)",
                    setup="from __main__ import test_my_kernel",
                    globals={"shape": shape, "dtype": dtype},
                )
                .timeit(1000)
                .mean
                * 1e6
            )
            torch_time_us = (
                benchmark.Timer(
                    stmt="test_torch(shape, dtype)",
                    setup="from __main__ import test_torch",
                    globals={"shape": shape, "dtype": dtype},
                )
                .timeit(1000)
                .mean
                * 1e6
            )

            speedup = torch_time_us / my_kernel_time_us

            print(f"Shape: {shape} \t dtype: {dtype}")
            print(f"torch: {torch_time_us:0.4f} (µs)")
            print(f"my kernels: {my_kernel_time_us:0.4f} (µs)")
            print(f"Speedup: {speedup: 0.4f} x")

            print("\n")

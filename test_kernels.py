import torch
import torch.utils.benchmark as benchmark

# from nanogpt_kernels import newton_schulz
# from nanogpt.muon.newton_schulz import newton_schulz as reference_newton_schulz


if __name__ == "__main__":
    for shape in [(128, 256), (512, 512), (1024, 2048), (2048, 1024)]:
        for dtype in [torch.float32, torch.half, torch.bfloat16]:
            G = torch.rand(128, 526, device="cuda", dtype=dtype)

            t0 = benchmark.Timer(
                stmt="newton_schulz(G, 5)",
                setup="from nanogpt_kernels import newton_schulz",
                globals={"G": G},
            )
            t1 = benchmark.Timer(
                stmt="reference_newton_schulz(G, 5)",
                setup="from nanogpt.muon.newton_schulz import newton_schulz as reference_newton_schulz",
                globals={"G": G},
            )

            torch_compile_us = t1.timeit(1000).mean * 1e6
            custom_kernels_us = t0.timeit(1000).mean * 1e6
            speedup = torch_compile_us / custom_kernels_us

            print(f"Shape: {shape} \t dtype: {dtype}")
            print(f"torch.compile: {torch_compile_us:0.4f} (µs)")
            print(f"my kernels: {custom_kernels_us:0.4f} (µs)")
            print(f"Speedup: {speedup: 0.4f} x")

            print("\n")

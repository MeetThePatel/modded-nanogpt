import torch
from rich.console import Console

from nanogpt.muon.newton_schulz import newton_schulz


def shape_to_str(shape):
    return "x".join(map(str, shape))


if __name__ == "__main__":
    console = Console()

    TARGET_SHAPES = [
        torch.Size([3072, 768]),
        torch.Size([768, 3072]),
        torch.Size([3, 768, 768]),
    ]
    NS_STEPS = 5
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    WARMUP_RUNS = 5
    PROFILE_RUNS = 10

    nvtx_range_push = torch.cuda.nvtx.range_push
    nvtx_range_pop = torch.cuda.nvtx.range_pop

    for shape in TARGET_SHAPES:
        shape_str = shape_to_str(shape)

        G_input = torch.randn(shape, dtype=DTYPE, device=DEVICE, requires_grad=False)

        console.print("Warming up.")
        for _ in range(WARMUP_RUNS):
            _ = newton_schulz(G_input, steps=NS_STEPS)
            torch.cuda.synchronize(device="cuda")
        console.print("Warmup complete.")

        total_gpu_time = 0.0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device="cuda")

        for i in range(PROFILE_RUNS):
            nvtx_range_push(f"ProfileRun_{shape_str}_run_{i + 1}")
            start_event.record()
            nvtx_range_push(f"zeropower_via_newtonschulz5_{shape_str}_run_{i + 1}")
            result = newton_schulz(G_input, steps=5)
            nvtx_range_pop()
            end_event.record()
            torch.cuda.synchronize(device="cuda")
            nvtx_range_pop()
            total_gpu_time += start_event.elapsed_time(end_event)

        console.print(f"{shape_str} - Average GPU time: {total_gpu_time / PROFILE_RUNS:.2f} ms")

        del G_input
        del result
        torch.cuda.empty_cache()

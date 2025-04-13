import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Generator, NoReturn

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from nanogpt import Muon, NanoGPT
from nanogpt.dataloader import distributed_data_generator
from nanogpt.helpers import get_window_size_blocks
from nanogpt.logging import DistributedLogger, collect_code_snapshot, log_system_info
from nanogpt.nanogpt import NanoGPTParams

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.manual_seed(69)  # nice
torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min


RUN_NAME = "train_gpt_rewrite;bs=64;hyperclone=1"


@dataclass(kw_only=True, frozen=True)
class TrainerParams:
    adam_head_lr: float
    adam_embed_lr: float
    adam_scalar_lr: float
    adam_betas: tuple[float, float]

    muon_mlp_lr: float
    muon_attn_lr: float
    muon_momentum: float
    muon_ns_steps: int

    n_steps: int
    warmup_steps: int
    validate_every: int = 125
    save_checkpoint: bool = False


@dataclass(kw_only=True, frozen=True)
class DataParams:
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    train_seq_len: int = 64 * 1024

    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_seq_len: int = 64 * 1024
    val_tokens: int = 10485760


def train_stage(
    model: NanoGPT,
    train_params: TrainerParams,
    data_params: DataParams,
    *,
    global_step: int,
    rank: int,
    world_size: int,
    train_loader: Generator[tuple[torch.Tensor, torch.Tensor], Any, NoReturn],
    logger: DistributedLogger,
    tensorboard_writer: SummaryWriter,
):
    model = model.cuda()
    for m in model.modules():
        m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    adam_head_params = [model.lm_head.weight]
    adam_embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    adam_scalar_params = [p for p in model.parameters() if p.ndim < 2]
    muon_mlp_params = [(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "mlp" in n]
    muon_attn_params = [(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "attn" in n]

    adam_optimizer = torch.optim.Adam(
        params=[
            dict(params=adam_head_params, lr=train_params.adam_head_lr),
            dict(params=adam_embed_params, lr=train_params.adam_embed_lr),
            dict(params=adam_scalar_params, lr=train_params.adam_scalar_lr),
        ],
        betas=train_params.adam_betas,
        eps=1e-10,
        foreach=True,
    )
    muon_optimizer = Muon(
        params=[
            dict(params=muon_mlp_params, lr=train_params.muon_mlp_lr),
            dict(params=muon_attn_params, lr=train_params.muon_attn_lr),
        ],
        momentum=train_params.muon_momentum,
        rank=rank,
        world_size=world_size,
    )
    optimizers = [adam_optimizer, muon_optimizer]

    model: nn.Module = torch.compile(model, dynamic=False)

    # Warmup
    initial_state = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers],
    )
    for _ in range(train_params.warmup_steps):
        inputs = targets = torch.randint(0, model.vocab_size, size=(data_params.train_seq_len,), device="cuda")
        model(inputs.to(torch.int32), targets, get_window_size_blocks(0, train_params.n_steps)).backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state

    training_time_ms = 0

    def validate(step: int):
        torch.cuda.synchronize()
        model.eval()
        val_batch_size = world_size * data_params.val_seq_len
        assert data_params.val_tokens % val_batch_size == 0
        val_steps = data_params.val_tokens // val_batch_size
        val_loader = distributed_data_generator(data_params.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step, train_params.n_steps))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        tensorboard_writer.add_scalar("val_loss", val_loss.item(), step)
        logger.log(
            f"step:{step}/{train_params.n_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms",
            print_to_console=True,
        )
        model.train()
        torch.cuda.synchronize()

    for step in range(train_params.n_steps + 1):
        if step % train_params.validate_every == 0:
            validate(global_step)

        if step == train_params.n_steps:
            validate(global_step)
            break

        step_start_time = time.perf_counter()

        inputs, targets = next(train_loader)
        loss = model(inputs, targets, get_window_size_blocks(step, train_params.n_steps))
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        for opt in optimizers:
            opt.step()

        model.zero_grad(set_to_none=True)

        step_time = time.perf_counter() - step_start_time
        training_time_ms += 1000 * step_time

        tensorboard_writer.add_scalar("train_loss", loss.item(), global_step=global_step)
        global_step += 1
        logger.log(
            f"step:{step + 1}/{train_params.n_steps} step_train_time:{step_time * 1000:.0f} total_train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / (step + 1):.2f}ms",
            print_to_console=True,
        )


def main(logger: DistributedLogger, tensorboard_writer: SummaryWriter):
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()

    data_params = DataParams()
    train_loader = distributed_data_generator(data_params.train_files, world_size * data_params.train_seq_len, rank, world_size)

    stage_0_model_params = NanoGPTParams(
        num_layers=12,
        num_heads=6,
        head_dim=32,
        model_dim=192,
        max_seq_len=max(data_params.train_seq_len, data_params.val_seq_len),
    )
    model = NanoGPT(stage_0_model_params)

    stage_0_trainer_params = TrainerParams(
        adam_head_lr=0.22,
        adam_embed_lr=0.6,
        adam_scalar_lr=0.04,
        adam_betas=[0.8, 0.95],
        #
        muon_attn_lr=0.25,
        muon_mlp_lr=0.35,
        muon_momentum=0.85,
        muon_ns_steps=5,
        #
        n_steps=250,
        warmup_steps=10,
        validate_every=50,
    )

    stage_1_trainer_params = TrainerParams(
        adam_head_lr=0.22,
        adam_embed_lr=0.6,
        adam_scalar_lr=0.04,
        adam_betas=[0.8, 0.95],
        #
        muon_attn_lr=0.25,
        muon_mlp_lr=0.35,
        muon_momentum=0.85,
        muon_ns_steps=5,
        #
        n_steps=250,
        warmup_steps=10,
        validate_every=50,
    )

    global_step = 0
    train_stage(
        model,
        stage_0_trainer_params,
        data_params,
        global_step=global_step,
        rank=rank,
        world_size=world_size,
        train_loader=train_loader,
        logger=logger,
        tensorboard_writer=tensorboard_writer,
    )

    model.hyperclone_(dim=-1)

    train_stage(
        model,
        stage_1_trainer_params,
        data_params,
        global_step=global_step,
        rank=rank,
        world_size=world_size,
        train_loader=train_loader,
        logger=logger,
        tensorboard_writer=tensorboard_writer,
    )

    logger.log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
        print_to_console=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    logger = DistributedLogger(name=RUN_NAME)
    tensorboard_writer = SummaryWriter(log_dir=f"./logs/tensorboard/{RUN_NAME}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    system_information = log_system_info()
    logger.log("\n" + system_information)

    code = collect_code_snapshot(base_dir)
    logger.log("\n" + code)

    main(logger, tensorboard_writer)

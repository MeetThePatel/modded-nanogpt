__all__ = ["TrainerParams", "train_stage"]

import copy
import time
from dataclasses import dataclass
from typing import Any, Generator, NoReturn

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from nanogpt import Muon, NanoGPT
from nanogpt.dataloader import DataParams, distributed_data_generator
from nanogpt.helpers import get_window_size_blocks
from nanogpt.logging import DistributedLogger


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


def train_stage(
    model: NanoGPT,
    train_params: TrainerParams,
    data_params: DataParams,
    *,
    global_step: int,
    rank: int,
    world_size: int,
    train_loader: Generator[tuple[torch.Tensor, torch.Tensor], Any, NoReturn],
    logger: DistributedLogger | None = None,
    tensorboard_writer: SummaryWriter | None = None,
):
    model = model.to(rank)

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
        ns_steps=5,
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
        _, loss = model(inputs.to(torch.int32), targets, get_window_size_blocks(0, train_params.n_steps))
        loss.backward()
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
                val_loss += model(inputs, targets, get_window_size_blocks(step, train_params.n_steps))[1]
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        if tensorboard_writer:
            tensorboard_writer.add_scalar("val_loss", val_loss.item(), step)
        if logger:
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
        _, loss = model(inputs, targets, get_window_size_blocks(step, train_params.n_steps))
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        # for opt in optimizers:
        #     opt.step()

        with torch.cuda.nvtx.range("Adam Step"):
            optimizers[0].step()

        with torch.cuda.nvtx.range("Muon Step"):
            optimizers[1].step()

        model.zero_grad(set_to_none=True)

        step_time = time.perf_counter() - step_start_time
        training_time_ms += 1000 * step_time

        if tensorboard_writer:
            tensorboard_writer.add_scalar("train_loss", loss.item(), global_step=global_step)
        global_step += 1
        if logger:
            logger.log(
                f"step:{step + 1}/{train_params.n_steps} step_train_time:{step_time * 1000:.0f} total_train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / (step + 1):.2f}ms",
                print_to_console=True,
            )

import copy
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from nanogpt import Muon, NanoGPT
from nanogpt.dataloader import distributed_data_generator
from nanogpt.helpers import get_window_size_blocks
from nanogpt.logging import DistributedLogger, collect_code_snapshot, log_system_info

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.manual_seed(42)
torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min


RUN_NAME = ""


@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    val_tokens = 10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 36 * 1024  # FlexAttention sequence length
    val_seq_len = 32 * 1024  # FlexAttention sequence length for validation
    # optimization
    num_iterations = 1000  # number of iterations to run
    cooldown_frac = 0.4  # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False


def main(logger: DistributedLogger):
    args = Hyperparameters()
    writer = SummaryWriter(log_dir=f"./muon_tensorboard_logs/{RUN_NAME}")

    # torchrun sets these env variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 1  # this code is designed for 8xH100
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()

    ########################################
    #    Construct model and optimizer     #
    ########################################

    model: nn.Module = NanoGPT(
        vocab_size=args.vocab_size,
        num_layers=12,
        num_heads=6,
        model_dim=768,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()
    for m in model.modules():
        m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    # collect the parameters to optimize
    # hidden_matrix_params = [(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    hidden_matrix_params_mlp = [(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "mlp" in n]
    hidden_matrix_params_attn = [(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "attn" in n]
    head_params = [model.lm_head.weight]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]

    # init the optimizer(s)
    adam_params = [
        dict(params=head_params, lr=0.22),
        dict(params=embed_params, lr=0.6),
        dict(params=scalar_params, lr=0.04),
    ]
    muon_params = [
        dict(params=hidden_matrix_params_mlp, lr=0.05),
        dict(params=hidden_matrix_params_attn, lr=0.10),
    ]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    adam_optimizer = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    muon_optimizer = Muon(
        params=muon_params,
        momentum=0.95,
        rank=rank,
        world_size=world_size,
    )
    optimizers = [adam_optimizer, muon_optimizer]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def get_lr(step: int):
        x = step / args.num_iterations  # progress in training
        assert 0 <= x < 1
        if x < 1 - args.cooldown_frac:
            return 1.0
        else:
            w = (1 - x) / args.cooldown_frac
            return w * 1.0 + (1 - w) * 0.1

    model: nn.Module = torch.compile(model, dynamic=False)

    ########################################
    #            Warmup kernels            #
    ########################################

    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    warmup_steps = 10
    initial_state = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers],
    )  # save the initial state
    for _ in range(warmup_steps):
        inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
        model(inputs.to(torch.int32), targets, get_window_size_blocks(0, args.num_iterations)).backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        adam_optimizer.step()
        muon_optimizer.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state

    ########################################
    #        Training and validation       #
    ########################################

    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    train_steps = args.num_iterations
    for step in range(train_steps + 1):
        last_step = step == train_steps

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, get_window_size_blocks(step, args.num_iterations))
            val_loss /= val_steps
            del val_loader
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            writer.add_scalar("val_loss", val_loss.item(), step)
            logger.log(
                f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms",
                print_to_console=True,
            )
            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            # if master_process and args.save_checkpoint:
            #     log = dict(
            #         step=step,
            #         code=code,
            #         model=model.state_dict(),
            #         optimizers=[opt.state_dict() for opt in optimizers],
            #     )
            #     os.makedirs(f"logs/{run_id}", exist_ok=True)
            #     torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        loss = model(inputs, targets, get_window_size_blocks(step, args.num_iterations))
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        # set optimization hyperparameters
        for group in adam_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
        for group in muon_optimizer.param_groups:
            frac = min(step / 300, 1)  # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers
        adam_optimizer.step()
        muon_optimizer.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        writer.add_scalar("adam_learning_rate", adam_optimizer.param_groups[0]["lr"], step)
        writer.add_scalar("muon_learning_rate", muon_optimizer.param_groups[0]["lr"], step)
        writer.add_scalar("train_loss", loss.item(), step)
        logger.log(
            f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
            print_to_console=True,
        )
    logger.log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
        print_to_console=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    logger = DistributedLogger(name=RUN_NAME)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    system_information = log_system_info()
    logger.log("\n" + system_information)

    code = collect_code_snapshot(base_dir)
    logger.log("\n" + code)
    main(logger)

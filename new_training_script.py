import os

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from nanogpt import NanoGPT
from nanogpt.dataloader import DataParams, distributed_data_generator
from nanogpt.logging import DistributedLogger, collect_code_snapshot, log_system_info
from nanogpt.nanogpt import NanoGPTParams
from nanogpt.training_utils import TrainerParams, train_stage

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.manual_seed(69)  # nice
torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min


RUN_NAME = "train_gpt_rewrite;bs=64;hyperclone=1"


def main(logger: DistributedLogger, tensorboard_writer: SummaryWriter):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()

    data_params = DataParams(
        train_seq_len=32 * 1024,
        val_seq_len=32 * 1024,
    )
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

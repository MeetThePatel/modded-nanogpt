import os
import sys
from typing import List

from rich.console import Console
from rich.table import Table
import torch
from torch import nn
import torch.distributed as dist

from nanogpt import NanoGPT, NanoGPTParams
from nanogpt.dataloader import DataParams, distributed_data_generator
from nanogpt.helpers import get_window_size_blocks
from nanogpt.logging import DistributedLogger
from nanogpt.training_utils import TrainerParams, train_stage

TRAIN_SEQUENCE_LEN = 4 * 1024


def test_expansion_dimension_full():
    """Assert that shapes of the tensors are correct after hypercloning the full model.

    Reference paper: https://arxiv.org/abs/2409.12903

    Run with `PRINT_PARAM_COUNT=1` to get table of the parameter tensor shapes.
    """
    print_param_count_table = int(os.environ.get("PRINT_PARAM_COUNT", 0)) == 1

    model_params = NanoGPTParams(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        head_dim=32,
        model_dim=192,
        max_seq_len=TRAIN_SEQUENCE_LEN,
    )

    # Stage 0: initialization
    model: nn.Module = NanoGPT(model_params).cuda()
    for m in model.modules():
        m.bfloat16()

    pre_hyperclone_shapes = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            shape = tuple(parameter.shape)
            pre_hyperclone_shapes[name] = shape

    # Stage 1: hypercloning
    model.hyperclone_(type="full")
    for m in model.modules():
        m.bfloat16()

    post_hyperclone_shapes = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            shape = tuple(parameter.shape)
            post_hyperclone_shapes[name] = shape

    assert_statuses = {}

    def hyperclone_assert_util(key: str, scalings: List[int]):
        pre_shape = pre_hyperclone_shapes[key]
        post_shape = post_hyperclone_shapes[key]

        assert len(pre_shape) == len(post_shape)
        assert len(pre_shape) == len(scalings)

        for idx, scaling in enumerate(scalings):
            assert_statuses[key] = pre_shape[idx] * scaling == post_shape[idx]

    for key in pre_hyperclone_shapes.keys():
        # Should remain equal
        if "skip_weights" in key:
            hyperclone_assert_util(key, [1])
        elif "embed" in key:
            hyperclone_assert_util(key, [1, 2])
        elif "lm_head" in key:
            hyperclone_assert_util(key, [1, 2])
        elif "lambdas" in key:
            hyperclone_assert_util(key, [1])
        elif "act.beta" in key:
            hyperclone_assert_util(key, [1])
        elif "attn.qkv_w" in key:
            hyperclone_assert_util(key, [1, 2, 2])
        elif "attn.c_proj" in key:
            hyperclone_assert_util(key, [2, 2])
        elif "mlp.c_fc" in key:
            hyperclone_assert_util(key, [2, 2])
        elif "mlp.c_proj" in key:
            hyperclone_assert_util(key, [2, 2])

    if print_param_count_table:
        table = Table(title="Model Parameters")
        table.add_column("Modules", justify="left")
        table.add_column("Pre-Hyperclone Shape", justify="right")
        table.add_column("Post-Hyperclone Shape", justify="right")
        table.add_column("Correct?", justify="center")

        for key in pre_hyperclone_shapes.keys():
            if assert_statuses[key] == 1:
                table.add_row(key, str(pre_hyperclone_shapes[key]), str(post_hyperclone_shapes[key]), "✅")
            else:
                table.add_row(key, str(pre_hyperclone_shapes[key]), str(post_hyperclone_shapes[key]), "❌")
        console = Console(record=True)
        with open(os.devnull, "w") as devnull:
            original_stdout = sys.stdout
            sys.stdout = devnull
            try:
                console.print(table)
            finally:
                sys.stdout = original_stdout
        print(console.export_text())

    for key, item in assert_statuses.items():
        assert item == 1, f"Hypercloned shape incorrect for key {key}"


def test_expansion_dimension_mlp():
    """Assert that shapes of the tensors are correct after hypercloning the MLP projection.

    Note: This does not increase the hidden dimension of the model; only the projection dimension is increased.

    Reference paper: https://arxiv.org/abs/2409.12903

    Run with `PRINT_PARAM_COUNT=1` to get table of the parameter tensor shapes.
    """
    print_param_count_table = int(os.environ.get("PRINT_PARAM_COUNT", 0)) == 1

    model_params = NanoGPTParams(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        head_dim=32,
        model_dim=192,
        max_seq_len=TRAIN_SEQUENCE_LEN,
    )

    # Stage 0: initialization
    model: nn.Module = NanoGPT(model_params).cuda()
    for m in model.modules():
        m.bfloat16()

    pre_hyperclone_shapes = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            shape = tuple(parameter.shape)
            pre_hyperclone_shapes[name] = shape

    # Stage 1: hypercloning
    model.hyperclone_(type="mlp")
    for m in model.modules():
        m.bfloat16()

    post_hyperclone_shapes = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            shape = tuple(parameter.shape)
            post_hyperclone_shapes[name] = shape

    assert_statuses = {}

    def hyperclone_assert_util(key: str, scalings: List[int]):
        pre_shape = pre_hyperclone_shapes[key]
        post_shape = post_hyperclone_shapes[key]

        assert len(pre_shape) == len(post_shape)
        assert len(pre_shape) == len(scalings)

        for idx, scaling in enumerate(scalings):
            assert_statuses[key] = pre_shape[idx] * scaling == post_shape[idx]

    for key in pre_hyperclone_shapes.keys():
        # Should remain equal
        if "skip_weights" in key:
            hyperclone_assert_util(key, [1])
        elif "embed" in key:
            hyperclone_assert_util(key, [1, 1])
        elif "lm_head" in key:
            hyperclone_assert_util(key, [1, 1])
        elif "lambdas" in key:
            hyperclone_assert_util(key, [1])
        elif "act.beta" in key:
            hyperclone_assert_util(key, [1])
        elif "attn.qkv_w" in key:
            hyperclone_assert_util(key, [1, 1, 1])
        elif "attn.c_proj" in key:
            hyperclone_assert_util(key, [1, 1])
        elif "mlp.c_fc" in key:
            hyperclone_assert_util(key, [2, 1])
        elif "mlp.c_proj" in key:
            hyperclone_assert_util(key, [1, 2])

    if print_param_count_table:
        table = Table(title="Model Parameters")
        table.add_column("Modules", justify="left")
        table.add_column("Pre-Hyperclone Shape", justify="right")
        table.add_column("Post-Hyperclone Shape", justify="right")
        table.add_column("Correct?", justify="center")

        for key in pre_hyperclone_shapes.keys():
            if assert_statuses[key] == 1:
                table.add_row(key, str(pre_hyperclone_shapes[key]), str(post_hyperclone_shapes[key]), "✅")
            else:
                table.add_row(key, str(pre_hyperclone_shapes[key]), str(post_hyperclone_shapes[key]), "❌")
        console = Console(record=True)
        with open(os.devnull, "w") as devnull:
            original_stdout = sys.stdout
            sys.stdout = devnull
            try:
                console.print(table)
            finally:
                sys.stdout = original_stdout
        print(console.export_text())

    for key, item in assert_statuses.items():
        assert item == 1, f"Hypercloned shape incorrect for key {key}"


def test_expansion_dimension_attn():
    """Assert that shapes of the tensors are correct after hypercloning the hidden dimension.

    Note: This does not increase the dimension of the MLP layer.

    Reference paper: https://arxiv.org/abs/2409.12903

    Run with `PRINT_PARAM_COUNT=1` to get table of the parameter tensor shapes.
    """
    print_param_count_table = int(os.environ.get("PRINT_PARAM_COUNT", 0)) == 1

    model_params = NanoGPTParams(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        head_dim=32,
        model_dim=192,
        max_seq_len=TRAIN_SEQUENCE_LEN,
    )

    # Stage 0: initialization
    model: nn.Module = NanoGPT(model_params).cuda()
    for m in model.modules():
        m.bfloat16()

    pre_hyperclone_shapes = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            shape = tuple(parameter.shape)
            pre_hyperclone_shapes[name] = shape

    # Stage 1: hypercloning
    model.hyperclone_(type="attn")
    for m in model.modules():
        m.bfloat16()

    post_hyperclone_shapes = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            shape = tuple(parameter.shape)
            post_hyperclone_shapes[name] = shape

    assert_statuses = {}

    def hyperclone_assert_util(key: str, scalings: List[int]):
        pre_shape = pre_hyperclone_shapes[key]
        post_shape = post_hyperclone_shapes[key]

        assert len(pre_shape) == len(post_shape)
        assert len(pre_shape) == len(scalings)

        for idx, scaling in enumerate(scalings):
            assert_statuses[key] = pre_shape[idx] * scaling == post_shape[idx]

    for key in pre_hyperclone_shapes.keys():
        # Should remain equal
        if "skip_weights" in key:
            hyperclone_assert_util(key, [1])
        elif "embed" in key:
            hyperclone_assert_util(key, [1, 2])
        elif "lm_head" in key:
            hyperclone_assert_util(key, [1, 2])
        elif "lambdas" in key:
            hyperclone_assert_util(key, [1])
        elif "act.beta" in key:
            hyperclone_assert_util(key, [1])
        elif "attn.qkv_w" in key:
            hyperclone_assert_util(key, [1, 2, 2])
        elif "attn.c_proj" in key:
            hyperclone_assert_util(key, [2, 2])
        elif "mlp.c_fc" in key:
            hyperclone_assert_util(key, [1, 2])
        elif "mlp.c_proj" in key:
            hyperclone_assert_util(key, [2, 1])

    if print_param_count_table:
        table = Table(title="Model Parameters")
        table.add_column("Modules", justify="left")
        table.add_column("Pre-Hyperclone Shape", justify="right")
        table.add_column("Post-Hyperclone Shape", justify="right")
        table.add_column("Correct?", justify="center")

        for key in pre_hyperclone_shapes.keys():
            if assert_statuses[key] == 1:
                table.add_row(key, str(pre_hyperclone_shapes[key]), str(post_hyperclone_shapes[key]), "✅")
            else:
                table.add_row(key, str(pre_hyperclone_shapes[key]), str(post_hyperclone_shapes[key]), "❌")
        console = Console(record=True)
        with open(os.devnull, "w") as devnull:
            original_stdout = sys.stdout
            sys.stdout = devnull
            try:
                console.print(table)
            finally:
                sys.stdout = original_stdout
        print(console.export_text())

    for key, item in assert_statuses.items():
        assert item == 1, f"Hypercloned shape incorrect for key {key}"


def test_function_preservation():
    """Assert that logits are unchanged after hypercloning.

    In the paper (https://arxiv.org/abs/2409.12903), this test corresponds to the "Function Preservation" requirement
    in the Methodology section:

        > After converting the smaller model to its equivalent larger model, the logits in the final layers of
        > both networks should match.
    """
    logger = DistributedLogger(name="test_function_preservation")

    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "44444"

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()

    data_params = DataParams(
        train_seq_len=4 * 1024,
        val_seq_len=4 * 1024,
    )
    train_loader = distributed_data_generator(data_params.train_files, data_params.train_seq_len, 0, 1)

    model_params = NanoGPTParams(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        head_dim=32,
        model_dim=192,
        max_seq_len=max(data_params.train_seq_len, data_params.val_seq_len),
    )

    # Stage 0
    model: nn.Module = NanoGPT(model_params)

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
        n_steps=50,
        warmup_steps=10,
        validate_every=100,
    )
    global_step = 0
    train_stage(
        model,
        stage_0_trainer_params,
        data_params,
        global_step=global_step,
        rank=0,
        world_size=1,
        train_loader=train_loader,
        logger=logger,
        tensorboard_writer=None,
    )

    test_inputs = test_targets = torch.randint(0, model_params.vocab_size, size=(data_params.train_seq_len,), device="cuda")
    pre_hyperclone_logits, pre_hyperclone_loss = model(
        test_inputs.to(torch.int32),
        test_targets,
        get_window_size_blocks(0, stage_0_trainer_params.n_steps),
    )

    # Stage 1
    model.hyperclone_(type="full")
    model.cuda()
    for m in model.modules():
        m.bfloat16()

    # Stage 1: forward pass
    post_hyperclone_logits, post_hyperclone_loss = model(
        test_inputs.to(torch.int32),
        test_targets,
        get_window_size_blocks(0, stage_0_trainer_params.n_steps),
    )

    breakpoint()

    # Assertions
    torch.testing.assert_close(pre_hyperclone_logits, post_hyperclone_logits)
    torch.testing.assert_close(pre_hyperclone_loss, post_hyperclone_loss)
    max_diff_logits = (post_hyperclone_logits - pre_hyperclone_logits).abs().max().item()
    max_diff_loss = (post_hyperclone_loss - pre_hyperclone_loss).abs().max().item()
    print(f"Max Diff Logits: {max_diff_logits:.5f}")
    print(f"Max Diff Loss: {max_diff_loss:.5f}")

import torch
from torch import nn

from nanogpt import NanoGPT, NanoGPTParams
from nanogpt.helpers import get_window_size_blocks
from nanogpt.logging import create_parameter_count_table

TRAIN_SEQUENCE_LEN = 48 * 1024
N_STEPS = 500

if __name__ == "__main__":
    model_params = NanoGPTParams(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        head_dim=32,
        model_dim=192,
        max_seq_len=TRAIN_SEQUENCE_LEN,
    )
    model: nn.Module = NanoGPT(model_params).cuda()
    model = torch.compile(model, dynamic=False)

    print(create_parameter_count_table(model))

    test_inputs = test_targets = torch.randint(0, model.vocab_size, size=(TRAIN_SEQUENCE_LEN,), device="cuda")
    pre_hyperclone_logits, _ = model(test_inputs.to(torch.int32), test_targets, get_window_size_blocks(0, N_STEPS))

    model.hyperclone_()
    print(create_parameter_count_table(model))

    post_hyperclone_logits, _ = model(test_inputs.to(torch.int32), test_targets, get_window_size_blocks(0, N_STEPS))

    torch.testing.assert_close(pre_hyperclone_logits, post_hyperclone_logits)

from torch import nn

from nanogpt import NanoGPT
from nanogpt.logging import create_parameter_count_table

if __name__ == "__main__":
    model: nn.Module = NanoGPT(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        head_dim=32,
        model_dim=192,
        max_seq_len=max(32 * 1024, 32 * 1024),
    ).cuda()
    print(create_parameter_count_table(model))
    model.hyperclone_()
    print(create_parameter_count_table(model))
    model.hyperclone_()
    print(create_parameter_count_table(model))

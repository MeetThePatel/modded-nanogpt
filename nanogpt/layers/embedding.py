__all__ = ["Embedding"]

import torch
from torch import nn
from torch.nn import functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.current_dim = model_dim

        self.weight = nn.Parameter(torch.empty((vocab_size, self.current_dim)))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)

    def hyperclone_(self, type: str = "full"):
        """Apply HyperCloning methodology to embedding tensor.

        Call this function on the phase transition.

        This changes the embedding weight tensor from having shape (vocab_size, model_dim)
        to (vocab_size, 2 * model_dim), by applying the symmetric method described in
        Section 3.4 of https://arxiv.org/abs/2409.12903
        """
        if type in ['full', 'attn']:
            scaled = self.weight / 2
            self.weight = nn.Parameter(scaled.repeat(1, 2))
            self.current_dim *= 2

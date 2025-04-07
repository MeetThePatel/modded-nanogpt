__all__ = ["Block"]

import os
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask

from .causal_self_attention import CausalSelfAttention
from .mlp import MLP


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        layer_idx: int,
    ):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        x0: Tensor,
        block_mask: BlockMask,
    ):
        profiling = os.getenv("PROFILE") == "1"

        with torch.cuda.nvtx.range("Block") if profiling else nullcontext():
            x = self.lambdas[0] * x + self.lambdas[1] * x0

        if self.attn is not None:
            with torch.cuda.nvtx.range("Block attention") if profiling else nullcontext():
                with torch.cuda.nvtx.range("Block attention prenorm") if profiling else nullcontext():
                    attn_normed = Block.norm(x)

                with torch.cuda.nvtx.range("Block attention") if profiling else nullcontext():
                    attn_resid = self.attn(attn_normed, ve, block_mask)

                x = x + attn_resid

        with torch.cuda.nvtx.range("Block mlp") if profiling else nullcontext():
            with torch.cuda.nvtx.range("Block mlp prenorm") if profiling else nullcontext():
                mlp_normed = Block.norm(x)

            with torch.cuda.nvtx.range("Block mlp") if profiling else nullcontext():
                mlp_resid = self.mlp(mlp_normed)

            x = x + mlp_resid
        return x

    @staticmethod
    def norm(x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))

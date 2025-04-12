__all__ = ["Block"]

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
        head_dim: int,
        max_seq_len: int,
        layer_idx: int,
    ):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, head_dim) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        x0: Tensor,
        block_mask: BlockMask,
    ):
        x = self.lambdas[0] * x + self.lambdas[1] * x0

        if self.attn is not None:
            attn_normed = Block.norm(x)
            attn_resid = self.attn(attn_normed, ve, block_mask)
            x = x + attn_resid

        mlp_normed = Block.norm(x)
        mlp_resid = self.mlp(mlp_normed)
        x = x + mlp_resid
        return x

    def hyperclone_(self, type: str = "full"):
        if type == "full":
            if self.attn is not None:
                self.attn.hyperclone_(type)
            self.mlp.hyperclone_(type)
        elif type == "attn":
            if self.attn is not None:
                self.attn.hyperclone_(type)
            self.mlp.hyperclone_(type)
        elif type == "mlp":
            if self.attn is not None:
                self.attn.hyperclone_(type)
            self.mlp.hyperclone_(type)

    @staticmethod
    def norm(x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))

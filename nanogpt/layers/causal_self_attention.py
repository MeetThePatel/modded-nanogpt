__all__ = ["CausalSelfAttention"]

import os
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .casted_linear import CastedLinear
from .rotary import Rotary


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        head_dim=128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim**-0.5)
        bound = (3**0.5) * std  # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        block_mask: BlockMask,
    ):
        profiling = os.getenv("PROFILE") == "1"

        with torch.cuda.nvtx.range("CausalSelfAttention forward") if profiling else nullcontext():
            B, T = x.size(0), x.size(1)  # batch size, sequence length
            assert B == 1, "Must use batch size = 1 for FlexAttention"

            with torch.cuda.nvtx.range("CausalSelfAttention QKV projection") if profiling else nullcontext():
                q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)

            with torch.cuda.nvtx.range("CausalSelfAttention QKV norm") if profiling else nullcontext():
                q, k = CausalSelfAttention.norm(q), CausalSelfAttention.norm(k)  # QK norm @Grad62304977

            with torch.cuda.nvtx.range("CausalSelfAttention QKV rope") if profiling else nullcontext():
                q, k = self.rotary(q), self.rotary(k)

            with torch.cuda.nvtx.range("CausalSelfAttention value embedding update") if profiling else nullcontext():
                if ve is not None:
                    v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)  # @KoszarskyB & @Grad62304977
                else:  # skip mid-layers token value embeddings by @YouJiacheng
                    v = self.lambdas[0] * v
                # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
                # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283

            with torch.cuda.nvtx.range("CausalSelfAttention flex attention") if profiling else nullcontext():
                y: Tensor = flex_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    block_mask=block_mask,
                    scale=0.12,
                    return_lse=False,
                ).transpose(1, 2)

            with torch.cuda.nvtx.range("CausalSelfAttention output projection") if profiling else nullcontext():
                y = y.contiguous().view(B, T, self.num_heads * self.head_dim)  # re-assemble all head outputs side by side
                y = self.c_proj(y)
            return y

    @staticmethod
    def norm(x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))

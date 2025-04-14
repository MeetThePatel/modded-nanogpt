__all__ = ["MLP"]


import torch
from torch import Tensor, nn

from .casted_linear import CastedLinear


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.hdim = 4 * self.dim
        self.c_fc = CastedLinear(self.dim, self.hdim)
        self.c_proj = CastedLinear(self.hdim, self.dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977
        self.act = ScaledReLU2(beta_init=1.0)

    def forward(self, x: Tensor):
        x = self.c_fc(x)

        # x = F.relu(x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.act(x)

        x = self.c_proj(x)
        return x

    def hyperclone_(self, type: str):
        if type == "full":
            self.c_fc.hyperclone_()
            self.c_proj.hyperclone_()
            self.dim *= 2
            self.hdim *= 2
        elif type == "attn":
            self.c_fc.hyperclone_(dim=1)
            self.c_proj.hyperclone_(dim=0)
            self.dim *= 2
        elif type == "mlp":
            self.c_fc.hyperclone_(dim=0)
            self.c_proj.hyperclone_(dim=1)
            self.hdim *= 2


class ScaledReLU2(nn.Module):
    def __init__(self, beta_init=1.0):
        super(ScaledReLU2, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]))

    def forward(self, x):
        return self.beta * torch.relu(x).pow(2)

__all__ = ["MLP", "ScaledReLU2"]


import torch
from torch import Tensor, nn

from .casted_linear import CastedLinear


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977
        self.act = ScaledReLU2(beta_init=1.0)

    def forward(self, x: Tensor):
        # profiling = os.getenv("NANOGPT_PROFILE") == "1"

        # with torch.cuda.nvtx.range("MLP c_fc") if profiling else nullcontext():
        with torch.cuda.nvtx.range("MLP c_fc"):
            x = self.c_fc(x)

        # with torch.cuda.nvtx.range("MLP activation") if profiling else nullcontext():
        with torch.cuda.nvtx.range("MLP activation"):
            # x = F.relu(x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
            x = self.act(x)

        # with torch.cuda.nvtx.range("MLP c_proj") if profiling else nullcontext():
        with torch.cuda.nvtx.range("MLP c_proj"):
            x = self.c_proj(x)

        return x


class ScaledReLU2(nn.Module):
    def __init__(self, beta_init=1.0):
        super(ScaledReLU2, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return self.beta * torch.relu(x).pow(2)

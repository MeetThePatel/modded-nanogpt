__all__ = ["CastedLinear"]

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CastedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_fp8=False,
        x_s=1.0,
        w_s=1.0,
        grad_s=1.0,
    ):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features**-0.5)  # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3**0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

    def hyperclone_(self):
        self.weight = nn.Parameter((self.weight / 2).repeat(2, 2))

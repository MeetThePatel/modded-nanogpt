__all__ = ["Rotary"]

import torch
from torch import Tensor, nn

class Rotary(nn.Module):
    def __init__(self, initial_dim: int, max_seq_len: int, max_model_dim: int):
        super().__init__()
        self.current_dim = initial_dim
        self.max_model_dim = max_model_dim

        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=max_model_dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(max_model_dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)

        self.register_buffer("cos_full", theta.cos(), persistent=False)
        self.register_buffer("sin_full", theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        active_channels = self.current_dim // 2
        seq_len = x_BTHD.size(-3)

        cos = self.cos_full[:seq_len, :active_channels].unsqueeze(0).unsqueeze(2)
        sin = self.sin_full[:seq_len, :active_channels].unsqueeze(0).unsqueeze(2)

        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x_BTHD)

    def hyperclone_(self):
        assert self.current_dim * 2 <= self.max_model_dim, "Cannot HyperClone past maximum model dimensions."
        self.current_dim *= 2
__all__ = ["Rotary"]

import torch
from torch import Tensor, nn


class Rotary(nn.Module):
    def __init__(self, initial_dim: int, max_seq_len: int, max_model_dim: int):
        super().__init__()
        self.current_dim = initial_dim
        self.max_model_dim = max_model_dim

        self._compute_tables(self.current_dim, max_seq_len)

        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=max_model_dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(max_model_dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)

        self.register_buffer("cos_full", theta.cos(), persistent=False)
        self.register_buffer("sin_full", theta.sin(), persistent=False)

    def _compute_tables(self, model_dim: int, max_seq_len: int):
        steps = model_dim // 4

        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=steps, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(steps)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j->ij", t, angular_freq)

        self.register_buffer("cos_table", theta.cos(), persistent=False)
        self.register_buffer("sin_table", theta.sin(), persistent=False)
        self._max_seq_len = max_seq_len

    def forward(self, x_BTHD: Tensor):
        active_channels = self.current_dim // 2
        seq_len = x_BTHD.size(-3)

        cos = self.cos_table[:seq_len, :active_channels].unsqueeze(0).unsqueeze(2)
        sin = self.sin_table[:seq_len, :active_channels].unsqueeze(0).unsqueeze(2)

        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x_BTHD)

    def hyperclone_(self, alpha: float = 1.0):
        assert self.current_dim * 2 <= self.max_model_dim, "Cannot HyperClone past maximum model dimensions."

        old_dim = self.current_dim
        new_dim = old_dim * 2

        old_cos = self.cos_table.clone()
        old_sin = self.sin_table.clone()

        repeat_factor = (new_dim // 2) // (old_dim // 2)
        old_cos_stretched = old_cos.repeat(1, repeat_factor)
        old_sin_stretched = old_sin.repeat(1, repeat_factor)

        steps = new_dim // 4
        angular_freq_new = (1 / 1024) ** torch.linspace(0, 1, steps=steps, dtype=torch.float32)
        angular_freq_new = torch.cat([angular_freq_new, angular_freq_new.new_zeros(steps)])
        t = torch.arange(self._max_seq_len, dtype=torch.float32)
        theta_new = torch.einsum("i,j->ij", t, angular_freq_new)  # shape: (max_seq_len, new_dim//2)
        new_cos = theta_new.cos()
        new_sin = theta_new.sin()

        # Blend the old (stretched) and new tables.
        effective_cos = (1 - alpha) * old_cos_stretched + alpha * new_cos
        effective_sin = (1 - alpha) * old_sin_stretched + alpha * new_sin

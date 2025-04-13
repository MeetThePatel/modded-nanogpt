__all__ = ["LambdaLRScheduler"]

from typing import Callable
import torch


class LambdaLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_lambda: Callable[..., float],
        last_epoch: int = -1,
    ):
        super().__init__(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
        self.custom_lambda = lr_lambda

    def step(self, *args: int | float, **kwargs: int | float) -> None:
        if not args and not kwargs:
            super().step()

        else:
            for arg in args:
                if not isinstance(arg, (int, float)):
                    raise TypeError(f"Positional argument {arg} must be int or float.")
            for key, value in kwargs.items():
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Keyword argument {key}={value} must be int or float.")

            for i, param_group in enumerate(self.optimizer.param_groups):
                multiplier: float = self.custom_lambda(*args, **kwargs)
                param_group["lr"] = param_group["initial_lr"] * multiplier

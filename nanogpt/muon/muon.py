__all__ = ["Muon"]

from typing import Any, Dict, List, Optional, Union, cast
import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable, Optimizer

from muon import _fused_muon_cuda


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor],
        momentum: Union[float, Tensor] = 0.95,
        ns_steps: int = 5,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"lr must be >= 0. Recieved: {lr}")
        if not (0.0 < momentum <= 1.0):
            raise ValueError(f"momentum must be in (0, 1]. Recieved: {momentum}")
        if not 0 < ns_steps:
            raise ValueError(f"ns_steps must be > 0. Recieved: {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
        )

        for p in params:
            if not 2 <= len(p.shape):
                raise ValueError("Muon optimizer can only work with 2+ dimensional tensors.")

        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("ns_steps", 5)

    def __init_group(self, group, params, grads, momentum_buffer_list):
        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            self.__init_group(group, params, grads, momentum_buffer_list)

            muon(
                params=params,
                grads=grads,
                momentum_buffer_list=momentum_buffer_list,
                momentum=group["momentum"],
                lr=group["lr"],
                ns_steps=group["ns_steps"],
                # weight_decay=group['weight_decay']  # TODO: Generalize to weight decay
            )

            if group["momentum"] != 0:
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss


def muon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_buffer_list: list[Tensor],
    *,
    momentum: float,
    lr: float,
    ns_steps: int,
):
    _fused_muon(
        params,
        grads,
        momentum_buffer_list,
        momentum=momentum,
        lr=lr,
        ns_steps=ns_steps,
    )


def _fused_muon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_buffer_list: list[Tensor],
    *,
    momentum: float,
    lr: float,
    ns_steps: int,
):
    if not params:
        return

    no_momentum_buffer = momentum == 0

    is_first_step = all(t is None for t in momentum_buffer_list) and not no_momentum_buffer
    if is_first_step:
        for idx, grad in enumerate(grads):
            momentum_buffer_list[idx] = torch.empty_like(grad)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, momentum_buffer_list], with_indices=False)

    for (device, _), ((device_params_, device_grads_, device_momentum_buffer_list), _) in grouped_tensors.items():
        device_params: list[Tensor] = cast(list[Tensor], device_params_)
        device_grads: list[Tensor] = cast(list[Tensor], device_grads_)

    _fused_muon_cuda(
        device_params,
        device_grads,
        [] if no_momentum_buffer else cast(list[Tensor], device_momentum_buffer_list),
        momentum=momentum,
        lr=lr,
        ns_steps=ns_steps,
        is_first_step=is_first_step,
    )

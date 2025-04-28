__all__ = ["Muon"]

from typing import Any, Dict, List, Optional, Union, cast
import torch
from torch.nn import functional as F
from torch import Tensor
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable, Optimizer

from .newton_schulz import newton_schulz


@torch.compile
def norm(x: Tensor) -> Tensor:
    F.normalize(x, dim=(0, 1))


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
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor],
        momentum: Union[float, Tensor] = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 1.0 >= momentum:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0 < ns_steps:
            raise ValueError(f"Invalid Newton-Schulz step count: {ns_steps}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )

        super().__init__(params, defaults)

        for p in params:
            if not 2 <= len(p.shape):
                raise ValueError("Muon optimizer can only work with 2+ dimensional tensors.")

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", True)
            group.setdefault("ns_steps", 5)

    def __init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

        return has_sparse_grad

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
                nesterov=group["nesterov"],
                # weight_decay=group['weight_decay']  # TODO: Generalize to weight decay
            )

            if group["momentum"] != 0:
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss

    #     defaults = dict(momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

    #     mlp_params = params[0]["params"]
    #     mlp_lr = params[0]["lr"]
    #     attn_params = params[1]["params"]
    #     attn_lr = params[1]["lr"]

    #     param_groups = []
    #     for size in {p.numel() for (_, p) in mlp_params}:
    #         group_params = [(n, p) for (n, p) in mlp_params if p.numel() == size]
    #         b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
    #         group = dict(
    #             params=[p for (_, p) in group_params],
    #             param_names=[n for (n, _) in group_params],
    #             update_buffer=b,
    #             update_buffer_views=[b[i] for i in range(world_size)],
    #             lr=mlp_lr,
    #         )
    #         param_groups.append(group)
    #     for size in {p.numel() for (_, p) in attn_params}:
    #         group_params = [(n, p) for (n, p) in attn_params if p.numel() == size]
    #         b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
    #         group = dict(
    #             params=[p for (_, p) in group_params],
    #             param_names=[n for (n, _) in group_params],
    #             update_buffer=b,
    #             update_buffer_views=[b[i] for i in range(world_size)],
    #             lr=attn_lr,
    #         )
    #         param_groups.append(group)
    #     super().__init__(param_groups, defaults)

    # @torch.no_grad()
    # def step(self):
    #     for group in self.param_groups:
    #         update_buffer: Tensor = group["update_buffer"]
    #         update_buffer_views: list[Tensor] = group["update_buffer_views"]
    #         # generate weight updates in distributed fashion
    #         params: list[Tensor] = group["params"]
    #         handle = None
    #         params_world = None

    #         def update_prev():  # optimized Muon implementation contributed by @YouJiacheng
    #             handle.wait()
    #             for p_world, g_world in zip(params_world, update_buffer_views):
    #                 p_world.add_(g_world.view_as(p_world), alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5)

    #         for base_i in range(len(params))[:: self.world_size]:
    #             if base_i + self.rank < len(params):
    #                 p = params[base_i + self.rank]

    #                 g = p.grad
    #                 assert g is not None

    #                 state = self.state[p]
    #                 if "momentum_buffer" not in state:
    #                     state["momentum_buffer"] = torch.zeros_like(g)
    #                 buf: Tensor = state["momentum_buffer"]

    #                 buf.lerp_(g, 1 - group["momentum"])
    #                 g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
    #                 g = newton_schulz(g, steps=group["ns_steps"]).flatten()
    #             else:
    #                 g = update_buffer_views[self.rank]
    #             if base_i > 0:
    #                 update_prev()  # async all_gather instead of sync all_reduce by @YouJiacheng
    #             handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
    #             params_world = params[base_i : base_i + self.world_size]
    #         update_prev()


def muon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_buffer_list: list[Tensor],
    momentum: float,
    lr: float,
    nesterov: bool,
):
    if not params:
        return

    # grad_scale_dict:  # TODO: do this later

    no_momentum_buffer = momentum == 0
    is_first_step = all(t is None for t in momentum_buffer_list) and not no_momentum_buffer

    if is_first_step:
        for idx, g in enumerate(grads):
            momentum_buffer_list[idx] = torch.empty_like(g)

    # TODO: Will need to test on multiGPU setup, for now, following optim.SGD API
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list],
        with_indices=False,
    )

    for (device, _), (
        (device_params_, device_grads_, device_momentum_buffer_list),
        _,
    ) in grouped_tensors.items():
        device_params: List[Tensor] = cast(List[Tensor], device_params_)
        device_grads: List[Tensor] = cast(List[Tensor], device_params_)

    nanogpt._muon(
        device_params,
        device_grads,
        [] if no_momentum_buffer else cast(List[Tensor], device_momentum_buffer_list),
        momentum=momentum,
        lr=lr,
        nesterov=nesterov,
        is_first_step=is_first_step,
    )

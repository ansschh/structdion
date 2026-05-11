"""
Muon optimizer with Adam grafting.

Based on dion.muon_reference.Muon, modified to add per-parameter Adam grafting
as described in the Moonlight paper (2602.09314v1): "All matrix optimizers graft
from Adam." Grafting uses Muon's orthogonalized direction but Adam's per-parameter
step magnitude.

Grafting formula (Agarwal et al. 2020, as in DistributedShampoo):
  For each matrix parameter:
    D_muon = NewtonSchulz(momentum(grad))   # Muon's direction
    D_adam = m1_hat / (sqrt(v_hat) + eps)   # Adam's direction (no lr)
    scale = ||D_adam||_F / ||D_muon||_F     # norm ratio (no lr)
    param -= lr * scale * D_muon            # lr applied SEPARATELY

This matches DistributedShampoo which computes the grafting norm ratio
WITHOUT lr, then multiplies by -lr separately (line 1256).

Handles FSDP2 (DTensor), DDP, and single-GPU training.
"""

import math
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Optional, Tuple

from dion.muon_reference import zeropower_via_newtonschulz5


class MuonGrafted(Optimizer):
    """
    Muon with Adam grafting.

    For each matrix parameter, uses Muon's orthogonalized direction but
    Adam's per-parameter step magnitude. The grafting ratio replaces
    adjust_lr (spectral_norm/rms_norm).

    Scalar parameters use standard AdamW via algorithm="adamw" param groups.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.016,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.95, 0.95),
        weight_decay: float = 0.1,
        epsilon: float = 1e-8,
        nesterov: bool = True,
        graft_beta1: float = 0.95,
        graft_beta2: float = 0.95,
        graft_eps: float = 1e-8,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=mu,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=betas,
            epsilon=epsilon,
            graft_beta1=graft_beta1,
            graft_beta2=graft_beta2,
            graft_eps=graft_eps,
        )
        super().__init__(params, defaults)

        if isinstance(params, dict):
            params = [params]

        for param_or_param_group in params:
            if isinstance(param_or_param_group, dict):
                algo = param_or_param_group.get("algorithm", "muon")
                if algo not in ("muon", "adamw"):
                    raise ValueError(f"Unknown algorithm: {algo}")
                for p in param_or_param_group["params"]:
                    self.state[p]["algorithm"] = algo
                    if algo == "muon" and p.ndim != 2:
                        raise ValueError(
                            f"Muon requires 2D parameters, but got {p.ndim}D"
                        )
            else:
                if isinstance(param_or_param_group, torch.Tensor):
                    p = param_or_param_group
                elif (
                    isinstance(param_or_param_group, tuple)
                    and len(param_or_param_group) == 2
                ):
                    p = param_or_param_group[1]
                else:
                    raise ValueError(
                        f"Invalid parameter type: {type(param_or_param_group)}"
                    )
                self.state[p]["algorithm"] = "muon"
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon requires 2D parameters, but got {p.ndim}D"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #     Muon + Grafting      #
            ############################

            muon_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "muon"
            ]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["epsilon"]
            graft_beta1 = group["graft_beta1"]
            graft_beta2 = group["graft_beta2"]
            graft_eps = group["graft_eps"]

            for p in muon_params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]

                # Initialize state on first step
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["graft_m1"] = torch.zeros_like(g)
                    state["graft_m2"] = torch.zeros_like(g)
                    state["graft_step"] = 0

                state["graft_step"] += 1
                step_t = state["graft_step"]

                # --- Adam moments for grafting (on local shard) ---
                m1 = state["graft_m1"]
                m2 = state["graft_m2"]
                m1.lerp_(g, 1 - graft_beta1)
                m2.lerp_(g.square(), 1 - graft_beta2)

                # Bias-corrected Adam direction
                bias_correction1 = 1 - graft_beta1 ** step_t
                bias_correction2 = 1 - graft_beta2 ** step_t
                m1_hat = m1 / bias_correction1
                m2_hat = m2 / bias_correction2
                d_adam = m1_hat / (m2_hat.sqrt() + graft_eps)

                # --- Muon momentum ---
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # --- Newton-Schulz orthogonalization + grafting ---
                if isinstance(g, DTensor):
                    g_local = g.full_tensor()
                    u = zeropower_via_newtonschulz5(
                        g_local, steps=group["ns_steps"], eps=eps
                    )

                    # Grafting: norm ratio WITHOUT lr (matching DistributedShampoo).
                    # scale = ||d_adam|| / ||u||, then lr applied separately.
                    d_adam_full = d_adam.full_tensor()
                    grafting_norm = d_adam_full.norm()
                    u_norm = u.norm()
                    scale = grafting_norm / (u_norm + 1e-30)

                    u = u * scale

                    # Convert back to DTensor and redistribute
                    u = DTensor.from_local(
                        u,
                        device_mesh=g.device_mesh,
                        placements=None,
                        run_check=False,
                    ).redistribute(placements=g.placements)
                else:
                    u = zeropower_via_newtonschulz5(
                        g, steps=group["ns_steps"], eps=eps
                    )

                    grafting_norm = d_adam.norm()
                    u_norm = u.norm()
                    scale = grafting_norm / (u_norm + 1e-30)
                    u = u * scale

                # Apply weight decay and grafted update
                # Weight decay uses lr (decoupled, same as AdamW)
                p.mul_(1 - lr * weight_decay)
                # lr applied separately (NOT baked into grafting scale)
                p.add_(u, alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            adamw_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "adamw"
            ]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["epsilon"]
            weight_decay = group["weight_decay"]

            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.mul_(1 - lr * weight_decay)
                p.add_(g, alpha=-lr / scale)

        return loss

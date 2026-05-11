"""
HybridOptimizersContainer for Muon with Adam grafting.

Copied from ada_dion/hybrid_optimizer.py and modified to use MuonGrafted
instead of dion.Muon. Key difference: no adjust_lr or output head LR scaling
(grafting handles adaptive step sizing).
"""
from __future__ import annotations

import functools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable

from torchtitan.experiments.ada_dion.param_grouper import group_params_for_hybrid
from .muon_grafted_optimizer import MuonGrafted


def _extract_device_mesh(model: nn.Module) -> DeviceMesh | None:
    """Extract the DeviceMesh from the first DTensor parameter, if any."""
    for p in model.parameters():
        if isinstance(p, DTensor):
            return p.device_mesh
    return None


class MuonGraftedContainer(OptimizersContainer):
    """
    Manages a single MuonGrafted optimizer per model_part that uses
    per-param-group algorithm="adamw" for scalar parameters.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "MuonGrafted"

        # Muon momentum
        mu: float = 0.95
        nesterov: bool = True

        # Grafting (Adam moments for step size)
        graft_beta1: float = 0.95
        graft_beta2: float = 0.95
        graft_eps: float = 1e-8

        # Scalar optimizer (AdamW for embed/output/norm)
        scalar_lr: float = 0.008
        scalar_beta1: float = 0.95
        scalar_beta2: float = 0.95
        scalar_eps: float = 1e-8
        scalar_weight_decay: float = 0.1
        norm_weight_decay: float = 0.0

    def __init__(
        self, config: Config, *, model_parts: list[nn.Module]
    ) -> None:
        # DO NOT call super().__init__() — it would create standard AdamW.
        self.model_parts = model_parts
        self.optimizers: list[Optimizer] = []
        all_params: list[nn.Parameter] = []

        for model in model_parts:
            groups = group_params_for_hybrid(model)

            param_groups = []

            # Matrix params — use Muon with grafting
            if groups.matrix_params:
                param_groups.append({
                    "params": groups.matrix_params,
                })

            # Embeddings — AdamW
            if groups.embed_params:
                param_groups.append({
                    "params": groups.embed_params,
                    "algorithm": "adamw",
                    "lr": config.scalar_lr,
                    "weight_decay": config.scalar_weight_decay,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "eps": config.scalar_eps,
                })

            # Output / LM head — AdamW (same LR as other scalars, no 1/sqrt(d) scaling)
            if groups.output_params:
                param_groups.append({
                    "params": groups.output_params,
                    "algorithm": "adamw",
                    "lr": config.scalar_lr,
                    "weight_decay": config.scalar_weight_decay,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "eps": config.scalar_eps,
                })

            # Norms / biases — AdamW with no weight decay
            if groups.norm_params:
                param_groups.append({
                    "params": groups.norm_params,
                    "algorithm": "adamw",
                    "lr": config.scalar_lr,
                    "weight_decay": config.norm_weight_decay,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "eps": config.scalar_eps,
                })

            if param_groups:
                opt = MuonGrafted(
                    param_groups,
                    lr=config.lr,
                    mu=config.mu,
                    betas=(config.scalar_beta1, config.scalar_beta2),
                    weight_decay=config.weight_decay,
                    epsilon=config.graft_eps,
                    nesterov=config.nesterov,
                    graft_beta1=config.graft_beta1,
                    graft_beta2=config.graft_beta2,
                    graft_eps=config.graft_eps,
                )
                self.optimizers.append(opt)
            else:
                opt = torch.optim.AdamW(
                    [torch.nn.Parameter(torch.empty(0))],
                    lr=config.lr,
                )
                self.optimizers.append(opt)

            all_params.extend(groups.matrix_params)
            all_params.extend(groups.embed_params)
            all_params.extend(groups.output_params)
            all_params.extend(groups.norm_params)

        self._validate_length(len(self.model_parts))
        Optimizer.__init__(self, all_params, {"lr": config.lr})

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            f"Expected {expected_length} optimizers, got {len(self.optimizers)}"
        )

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for opt in self.optimizers:
            opt.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        sd = {}
        for model, opt in zip(self.model_parts, self.optimizers):
            for k, v in func(model, opt).items():
                sd[k] = v
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def init_cache_state_dict(self) -> None:
        pass

    def get_optimizers(self) -> list[Optimizer]:
        return self.optimizers

"""
BaseHybridOptimizersContainer: shared base class for all ortho_matrix optimizers.

Subclasses override:
  - Config (add optimizer-specific hyperparameters)
  - _create_optimizer (instantiate the concrete optimizer)
"""
from __future__ import annotations

import functools
import math
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

from .param_grouper import group_params_for_hybrid


def _extract_device_mesh(model: nn.Module) -> DeviceMesh | None:
    """Extract the DeviceMesh from the first DTensor parameter, if any."""
    for p in model.parameters():
        if isinstance(p, DTensor):
            return p.device_mesh
    return None


class BaseHybridOptimizersContainer(OptimizersContainer):
    """
    Base class for hybrid optimizer containers that manage a single optimizer
    per model_part using the spectral + AdamW param-group pattern.

    Subclasses must:
      1. Define a Config dataclass that inherits from this Config
      2. Override _create_optimizer() to instantiate the concrete optimizer
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        # --- Scalar optimizer (AdamW param groups) ---
        scalar_lr: float = 0.008
        scalar_weight_decay: float = 0.1
        scalar_beta1: float = 0.95
        scalar_beta2: float = 0.95
        scalar_eps: float = 1e-8

        # --- Output head LR scaling ---
        output_head_lr_scaling: bool = True

    def __init__(
        self, config: Config, *, model_parts: list[nn.Module]
    ) -> None:
        # DO NOT call super().__init__() — it would create standard AdamW.
        # We manually set up everything.

        self.model_parts = model_parts
        self.optimizers: list[Optimizer] = []
        all_params: list[nn.Parameter] = []

        for model in model_parts:
            groups = group_params_for_hybrid(model)

            # Build param groups for a single optimizer
            param_groups = []

            # Matrix params — use the spectral optimizer natively
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

            # Output / LM head — AdamW
            if groups.output_params:
                if config.output_head_lr_scaling:
                    # lr / sqrt(d_in) scaling per microsoft/dion README
                    d_in = groups.output_params[0].shape[1]
                    output_lr = config.lr / math.sqrt(d_in)
                else:
                    output_lr = config.scalar_lr
                param_groups.append({
                    "params": groups.output_params,
                    "algorithm": "adamw",
                    "lr": output_lr,
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
                    "weight_decay": 0.0,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "eps": config.scalar_eps,
                })

            if param_groups:
                mesh = _extract_device_mesh(model)
                opt = self._create_optimizer(config, param_groups, mesh)
                self.optimizers.append(opt)
            else:
                # Fallback: dummy optimizer if no params
                opt = torch.optim.AdamW(
                    [torch.nn.Parameter(torch.empty(0))],
                    lr=config.lr,
                )
                self.optimizers.append(opt)

            # Collect all params for base Optimizer.__init__
            all_params.extend(groups.matrix_params)
            all_params.extend(groups.embed_params)
            all_params.extend(groups.output_params)
            all_params.extend(groups.norm_params)

        self._validate_length(len(self.model_parts))
        # Initialize base Optimizer with all params (for grad clipping compat)
        Optimizer.__init__(self, all_params, {"lr": config.lr})

    @staticmethod
    def _create_optimizer(
        config: "BaseHybridOptimizersContainer.Config",
        param_groups: list[dict],
        mesh: DeviceMesh | None = None,
    ) -> Optimizer:
        """Create the appropriate optimizer. Must be overridden by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _create_optimizer"
        )

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            f"Expected {expected_length} optimizers, got {len(self.optimizers)}"
        )

    # ------------------------------------------------------------------
    # Iteration — LRSchedulersContainer iterates over this
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    # ------------------------------------------------------------------
    # Training loop interface
    # ------------------------------------------------------------------

    def step(self, *args, **kwargs) -> None:
        """Step all optimizers."""
        for opt in self.optimizers:
            opt.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients on all optimizers."""
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    # ------------------------------------------------------------------
    # Checkpoint state dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Save state dicts for all optimizers."""
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
        """Load state dicts for all optimizers."""
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def init_cache_state_dict(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Diagnostic interface
    # ------------------------------------------------------------------

    def get_optimizers(self) -> list[Optimizer]:
        """Return the optimizers for diagnostic access."""
        return self.optimizers

    def get_optimizer_metrics(self) -> dict[str, float]:
        """Collect diagnostic metrics from all optimizers for logging."""
        metrics: dict[str, float] = {}
        for opt in self.optimizers:
            if hasattr(opt, "get_rank"):
                for k, v in opt.get_rank().items():
                    metrics[f"z/optimizer/rank/{k}"] = v
            if hasattr(opt, "get_effective_rank"):
                for k, v in opt.get_effective_rank().items():
                    metrics[f"z/optimizer/erank/{k}"] = v
            if hasattr(opt, "get_aerr"):
                for k, v in opt.get_aerr().items():
                    metrics[f"z/optimizer/aerr/{k}"] = v
        return metrics

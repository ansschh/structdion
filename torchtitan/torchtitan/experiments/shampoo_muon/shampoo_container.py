"""
ShampooMuonContainer: Wraps DistributedShampoo configured as Muon for TorchTitan.

Paper-faithful implementation of Muon (SVD) from 2602.09314v1 Table 1:
- SpectralDescent (SVD or Newton-Schulz) for hidden weight matrices
- Adam grafting for step-size scaling (second-moment only)
- EMA momentum (not classical), no Nesterov
- AdamW for embedding/output/norm parameters
- Native FSDP2 support via FullyShardDistributedConfig
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

from distributed_shampoo import (
    AdamPreconditionerConfig,
    DistributedShampoo,
    FullyShardDistributedConfig,
    NewtonSchulzOrthogonalizationConfig,
    SingleDeviceDistributedConfig,
    SpectralDescentPreconditionerConfig,
    SVDOrthogonalizationConfig,
    WeightDecayType,
)

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.experiments.ada_dion.param_grouper import group_params_for_hybrid


def _extract_device_mesh(model: nn.Module) -> DeviceMesh | None:
    """Extract the DeviceMesh from the first DTensor parameter, if any."""
    for p in model.parameters():
        if isinstance(p, DTensor):
            return p.device_mesh
    return None


class ShampooMuonContainer(OptimizersContainer):
    """
    Wraps DistributedShampoo configured as Muon (spectral descent + Adam grafting)
    for TorchTitan's optimizer interface.

    One DistributedShampoo instance per model_part with per-param-group configs:
    - Matrix params: SpectralDescent (SVD or Newton-Schulz) + Adam grafting
    - Embed/output params: Pure AdamW (via start_preconditioning_step=inf)
    - Norm/bias params: Pure AdamW with no weight decay
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "ShampooMuon"

        # --- Orthogonalization ---
        orthogonalization: str = "svd"  # "svd" or "newton_schulz"
        ns_steps: int = 5  # Newton-Schulz iterations (only if orthogonalization="newton_schulz")

        # --- Gradient EMA ---
        beta1: float = 0.95  # Gradient filtering EMA coefficient

        # --- Adam grafting for matrix params ---
        graft_beta2: float = 0.95
        graft_eps: float = 1e-8

        # --- Scalar optimizer (embed/output/norm via AdamW) ---
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

        # Build orthogonalization config
        if config.orthogonalization == "svd":
            orth_config = SVDOrthogonalizationConfig()
        elif config.orthogonalization == "newton_schulz":
            orth_config = NewtonSchulzOrthogonalizationConfig(
                num_iterations=config.ns_steps,
                scale_by_dims_fn=lambda d_in, d_out: max(1, d_out / d_in) ** 0.5,
            )
        else:
            raise ValueError(
                f"Unknown orthogonalization: {config.orthogonalization!r}. "
                f"Supported: 'svd', 'newton_schulz'"
            )

        for model in model_parts:
            groups = group_params_for_hybrid(model)

            # Detect FSDP2 sharding
            mesh = _extract_device_mesh(model)
            if mesh is not None:
                matrix_dist_config = FullyShardDistributedConfig(
                    target_parameter_dimensionality=2,
                )
                scalar_dist_config = FullyShardDistributedConfig()
            else:
                matrix_dist_config = SingleDeviceDistributedConfig(
                    target_parameter_dimensionality=2,
                )
                scalar_dist_config = SingleDeviceDistributedConfig()

            param_groups = []

            # Matrix params — SpectralDescent + Adam grafting
            if groups.matrix_params:
                param_groups.append({
                    "params": groups.matrix_params,
                    "lr": config.lr,
                    "betas": (config.beta1, 1.0),
                    "preconditioner_config": SpectralDescentPreconditionerConfig(
                        orthogonalization_config=orth_config,
                    ),
                    "max_preconditioner_dim": math.inf,
                    "distributed_config": matrix_dist_config,
                    "grafting_config": AdamPreconditionerConfig(
                        beta2=config.graft_beta2,
                        epsilon=config.graft_eps,
                    ),
                })

            # Embeddings — AdamW
            if groups.embed_params:
                param_groups.append({
                    "params": groups.embed_params,
                    "lr": config.scalar_lr,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "start_preconditioning_step": math.inf,
                    "distributed_config": scalar_dist_config,
                    "grafting_config": AdamPreconditionerConfig(
                        beta2=config.scalar_beta2,
                        epsilon=config.scalar_eps,
                    ),
                })

            # Output / LM head — AdamW (no 1/sqrt(d) scaling per paper)
            if groups.output_params:
                param_groups.append({
                    "params": groups.output_params,
                    "lr": config.scalar_lr,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "start_preconditioning_step": math.inf,
                    "distributed_config": scalar_dist_config,
                    "grafting_config": AdamPreconditionerConfig(
                        beta2=config.scalar_beta2,
                        epsilon=config.scalar_eps,
                    ),
                })

            # Norms / biases — AdamW with no weight decay
            if groups.norm_params:
                param_groups.append({
                    "params": groups.norm_params,
                    "lr": config.scalar_lr,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "weight_decay": config.norm_weight_decay,
                    "start_preconditioning_step": math.inf,
                    "distributed_config": scalar_dist_config,
                    "grafting_config": AdamPreconditionerConfig(
                        beta2=config.scalar_beta2,
                        epsilon=config.scalar_eps,
                    ),
                })

            if param_groups:
                opt = DistributedShampoo(
                    param_groups,
                    weight_decay=config.weight_decay,
                    weight_decay_type=WeightDecayType.DECOUPLED,
                    use_bias_correction=True,
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
        for opt in self.optimizers:
            opt.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    # ------------------------------------------------------------------
    # Checkpoint state dict
    # ------------------------------------------------------------------

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

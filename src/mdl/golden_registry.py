"""Registry mapping task names to golden network implementations."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable


@dataclasses.dataclass
class GoldenSpec:
    """Specification for a golden network."""
    build_params: Callable   # (p=0.3) -> dict of JAX arrays
    forward: Callable        # (params, x) -> logits
    mdl_score: Callable      # (p=0.3) -> dict with total_bits etc.


_REGISTRY: dict[str, GoldenSpec] = {}


def register_golden(task_name: str, spec: GoldenSpec) -> None:
    _REGISTRY[task_name] = spec


def get_golden(task_name: str) -> GoldenSpec:
    if task_name not in _REGISTRY:
        raise ValueError(
            f"No golden network for task {task_name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[task_name]


def has_golden(task_name: str) -> bool:
    return task_name in _REGISTRY


# --- Register known goldens ---

def _register_anbn():
    from .golden import (
        build_golden_network_params, golden_forward, golden_mdl_score,
    )
    register_golden("anbn", GoldenSpec(
        build_params=build_golden_network_params,
        forward=golden_forward,
        mdl_score=golden_mdl_score,
    ))


def _register_dyck1():
    from .golden_dyck1 import (
        build_golden_dyck1_params, golden_dyck1_forward, golden_dyck1_mdl_score,
    )
    register_golden("dyck1", GoldenSpec(
        build_params=build_golden_dyck1_params,
        forward=golden_dyck1_forward,
        mdl_score=golden_dyck1_mdl_score,
    ))


def _register_freeform_anbn():
    from .golden_freeform_anbn import (
        build_golden_freeform_anbn_params,
        golden_freeform_anbn_forward,
        golden_freeform_anbn_mdl_score,
    )
    register_golden("anbn_freeform", GoldenSpec(
        build_params=build_golden_freeform_anbn_params,
        forward=golden_freeform_anbn_forward,
        mdl_score=golden_freeform_anbn_mdl_score,
    ))


def _register_freeform_dyck1():
    from .golden_freeform_dyck1 import (
        build_golden_freeform_dyck1_params,
        golden_freeform_dyck1_forward,
        golden_freeform_dyck1_mdl_score,
    )
    register_golden("dyck1_freeform", GoldenSpec(
        build_params=build_golden_freeform_dyck1_params,
        forward=golden_freeform_dyck1_forward,
        mdl_score=golden_freeform_dyck1_mdl_score,
    ))


def _register_freeform_anbncn():
    from .golden_freeform_anbncn import (
        build_golden_freeform_anbncn_params,
        golden_freeform_anbncn_forward,
        golden_freeform_anbncn_mdl_score,
    )
    register_golden("anbncn_freeform", GoldenSpec(
        build_params=build_golden_freeform_anbncn_params,
        forward=golden_freeform_anbncn_forward,
        mdl_score=golden_freeform_anbncn_mdl_score,
    ))


# Deferred registration to avoid circular imports
_register_anbn()
_register_dyck1()
_register_freeform_anbn()
_register_freeform_dyck1()
_register_freeform_anbncn()

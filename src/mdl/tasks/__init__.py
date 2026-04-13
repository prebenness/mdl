"""Task definitions for formal language MDL experiments.

Each task provides data generation, test sets, grammar weights, and
evaluation helpers following the conventions of Abudy et al. (2025,
"Learning Formal Languages with Small RNNs Using MDL Regularization").
"""

from .base import TaskSpec
from .anbn import AnbnTask
from .anbncn import AnbncnTask
from .dyck1 import Dyck1Task


_TASK_REGISTRY: dict[str, type[TaskSpec]] = {
    "anbn": AnbnTask,
    "anbncn": AnbncnTask,
    "dyck1": Dyck1Task,
}


def get_task(name: str, **kwargs) -> TaskSpec:
    """Instantiate a task by name.

    Args:
        name: one of the registered task names.
        **kwargs: forwarded to the task constructor (e.g. p=0.3).
    """
    if name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task {name!r}. Available: {sorted(_TASK_REGISTRY)}"
        )
    return _TASK_REGISTRY[name](**kwargs)


def register_task(name: str, cls: type[TaskSpec]) -> None:
    """Register a new task class."""
    _TASK_REGISTRY[name] = cls

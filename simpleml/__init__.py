"""SimpleML — a modular, registry-based ML training framework."""

from simpleml.registries import (
    DATASETS,
    LOSSES,
    METRICS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
)
from simpleml.registry import Registry

import simpleml.datasets  # noqa: F401, E402  # isort: skip
import simpleml.losses  # noqa: F401, E402  # isort: skip
import simpleml.models  # noqa: F401, E402  # isort: skip
import simpleml.metrics  # noqa: F401, E402  # isort: skip
import simpleml.optimizers  # noqa: F401, E402  # isort: skip

__all__ = [
    "Registry",
    "MODELS",
    "LOSSES",
    "DATASETS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "METRICS",
]

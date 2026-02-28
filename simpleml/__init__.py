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

import simpleml.configs  # noqa: F401, E402  # isort: skip
import simpleml.datasets  # noqa: F401, E402  # isort: skip
import simpleml.losses  # noqa: F401, E402  # isort: skip
import simpleml.models  # noqa: F401, E402  # isort: skip
import simpleml.metrics  # noqa: F401, E402  # isort: skip
import simpleml.optimizers  # noqa: F401, E402  # isort: skip

from simpleml.configs import Config

__all__ = [
    "Config",
    "Registry",
    "MODELS",
    "LOSSES",
    "DATASETS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "METRICS",
]

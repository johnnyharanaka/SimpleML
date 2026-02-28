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
import simpleml.trainers  # noqa: F401, E402  # isort: skip

from simpleml.configs import Config
from simpleml.trainers import Trainer

__all__ = [
    "Config",
    "Trainer",
    "Registry",
    "MODELS",
    "LOSSES",
    "DATASETS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "METRICS",
]

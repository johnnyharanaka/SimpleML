"""Global registry instances for each module category."""

from simpleml.registry import Registry

MODELS = Registry("models")
LOSSES = Registry("losses")
DATASETS = Registry("datasets")
OPTIMIZERS = Registry("optimizers")
METRICS = Registry("metrics")

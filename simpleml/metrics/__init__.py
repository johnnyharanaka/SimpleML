"""Metrics — registered for config-driven use."""

from simpleml.metrics.accuracy import Accuracy
from simpleml.metrics.auroc import AUROC
from simpleml.metrics.confusion_matrix import ConfusionMatrix
from simpleml.metrics.corloc import CorLoc
from simpleml.metrics.f1_score import F1Score
from simpleml.metrics.mean_average_precision import MeanAveragePrecision
from simpleml.metrics.precision import Precision
from simpleml.metrics.recall import Recall

__all__ = [
    "Accuracy",
    "AUROC",
    "ConfusionMatrix",
    "CorLoc",
    "F1Score",
    "MeanAveragePrecision",
    "Precision",
    "Recall",
]

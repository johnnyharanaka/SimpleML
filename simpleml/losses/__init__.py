"""Loss functions — registered for config-driven use."""

from simpleml.losses.bce_with_logits import BCEWithLogitsLoss
from simpleml.losses.cross_entropy import CrossEntropyLoss
from simpleml.losses.focal import FocalLoss
from simpleml.losses.ntxent import NTXentLoss
from simpleml.losses.supcon import SupConLoss
from simpleml.losses.triplet import TripletMarginLoss

__all__ = [
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "FocalLoss",
    "NTXentLoss",
    "SupConLoss",
    "TripletMarginLoss",
]

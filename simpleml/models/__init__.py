"""Model definitions — network architectures registered for config-driven use."""

import custom_models.dinov2.app.dino_classifier  # noqa: F401
import custom_models.dinov2.app.dino_detector  # noqa: F401

from simpleml.models.timm_model import TimmModel

__all__ = ["TimmModel"]

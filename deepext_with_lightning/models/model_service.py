from typing import List

from .base import BaseDeepextModel
from .segmentation import ShelfNet, UNet, ResUNet
from .object_detection import EfficientDetector
from .classification import MobileNetV3, CustomClassificationNetwork, AttentionBranchNetwork, EfficientNet

MODEL_DICT = {
    # Classification
    "mobilenet": MobileNetV3,
    "customnet": CustomClassificationNetwork,
    "abn": AttentionBranchNetwork,
    "efficientnet": EfficientNet,
    # Segmentation
    "unet": UNet,
    "resunet": ResUNet,
    "shelfnet": ShelfNet,
    # Object detection
    "efficientdet": EfficientDetector,
}


def resolve_model_class_from_name(model_name: str) -> BaseDeepextModel.__class__:
    model_class = MODEL_DICT.get(model_name)
    if model_class is None:
        raise ValueError(f"Invali model name. Expected {valid_model_names()}")
    return model_class


def valid_model_names() -> List[str]:
    return list(MODEL_DICT.keys())

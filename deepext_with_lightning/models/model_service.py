from typing import List

from .base import BaseDeepextModel, ClassificationModel, DetectionModel, SegmentationModel
from .segmentation import ShelfNet, UNet, ResUNet
from .object_detection import EfficientDetector
from .classification import MobileNetV3, CustomClassificationNetwork, AttentionBranchNetwork, EfficientNet

_CLS_MODEL_DICT = {
    "mobilenet": MobileNetV3,
    "customnet": CustomClassificationNetwork,
    "abn": AttentionBranchNetwork,
    "efficientnet": EfficientNet,
}

_SEGMENTATION_MODEL_DICT = {
    "unet": UNet,
    "resunet": ResUNet,
    "shelfnet": ShelfNet,
}

_DET_MODEL_DICT = {
    "efficientdet": EfficientDetector,
}


def resolve_classification_model(model_name: str) -> ClassificationModel.__class__:
    model_class = _CLS_MODEL_DICT.get(model_name)
    if model_class is None:
        raise ValueError(f"Invali model name. Expected {list(_CLS_MODEL_DICT.keys())}")
    return model_class


def resolve_segmentation_model(model_name: str) -> SegmentationModel.__class__:
    model_class = _SEGMENTATION_MODEL_DICT.get(model_name)
    if model_class is None:
        raise ValueError(f"Invali model name. Expected {list(_SEGMENTATION_MODEL_DICT.keys())}")
    return model_class


def resolve_detection_model(model_name: str) -> DetectionModel.__class__:
    model_class = _DET_MODEL_DICT.get(model_name)
    if model_class is None:
        raise ValueError(f"Invali model name. Expected {list(_DET_MODEL_DICT.keys())}")
    return model_class


def resolve_model_class_from_name(model_name: str) -> BaseDeepextModel.__class__:
    model_class = _CLS_MODEL_DICT.get(model_name)
    if model_class:
        return model_class

    model_class = _SEGMENTATION_MODEL_DICT.get(model_name)
    if model_class:
        return model_class

    model_class = _DET_MODEL_DICT.get(model_name)
    if model_class:
        return model_class
    raise ValueError(
        f"Invalid model name. Expected {valid_model_names()}")


def valid_model_names() -> List[str]:
    return list(_CLS_MODEL_DICT.keys()) + list(_SEGMENTATION_MODEL_DICT.keys()) + list(_DET_MODEL_DICT.keys())

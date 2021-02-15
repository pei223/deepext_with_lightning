from typing import Tuple
from abc import abstractmethod

import cv2
import torch
import numpy as np
from .base_deepext_model import BaseDeepextModel
from ...image_process.drawer import combine_heatmap


class ClassificationModel(BaseDeepextModel):
    """
    forward
        input:
            image: (Batch size, channels, height, width)
            target: (Batch size, classes) one-hot
        output:
            (Batch size, classes)
    """

    @abstractmethod
    def predict_label(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param img: (batch size, channels, height, width)
        :return: (Batch size(Class num value), ),  (Batch size, classes)
        """
        pass

    def onehot_to_label(self, onehot_tensor: torch.Tensor):
        if onehot_tensor.ndim == 1:
            return onehot_tensor
        return torch.argmax(onehot_tensor, dim=1)


class MultiClassificationModel(BaseDeepextModel):
    """
    forward
        input:
            image: (Batch size, channels, height, width)
            target: (Batch size, classes) one-hot (multi class)
        output:
            (Batch size, classes)
    """

    @abstractmethod
    def predict_multi_class(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param img: (batch size, channels, height, width)
        :return: (Batch size. classes),  probabilities (Batch size, classes)
        """
        pass


class AttentionClassificationModel(ClassificationModel):
    """
    forward
        input:
            image: (Batch size, channels, height, width)
            target: (Batch size, classes) one-hot
        output:
            (Batch size, classes)
    """

    @abstractmethod
    def predict_label_and_heatmap(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param img: inputs: (batch size, channels, height, width)
        :return: (Batch size(Class num value), ), (Batch size, classes), heatmap(batch size, h, w)
        """
        pass

    def generate_heatmap_image(self, origin_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """
        :param origin_image: (height, width, 3) uint8(0~255)
        :param heatmap: (height', width') uint8(0~255)
        :return: (height, width, 3) uint8(0~255)
        """
        assert heatmap.ndim == 2
        assert origin_image.ndim == 3
        heatmap = cv2.resize(heatmap, (origin_image.shape[1], origin_image.shape[0])).astype('uint8')
        return combine_heatmap(origin_image, heatmap)

    def _normalize_heatmap(self, heatmap: torch.Tensor):
        min_val = torch.min(heatmap)
        max_val = torch.max(heatmap)
        return (heatmap - min_val) / (max_val - min_val)


class AttentionMultiClassificationModel(MultiClassificationModel):
    """
    forward
        input:
            image: (Batch size, channels, height, width)
            target: (Batch size, classes) one-hot (multi class)
        output:
            (Batch size, classes)
    """

    @abstractmethod
    def predict_labels_and_heatmap(self, img: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param img: inputs: (batch size, channels, height, width)
        :return: probabilities (Batch size, classes), heatmap(batch size, h, w)
        """
        pass

    def _normalize_heatmap(self, heatmap: np.ndarray):
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        return (heatmap - min_val) / (max_val - min_val)

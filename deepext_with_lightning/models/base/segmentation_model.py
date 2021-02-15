from abc import abstractmethod
from typing import Tuple
import torch
import numpy as np
import cv2
from .base_deepext_model import BaseDeepextModel
from ...image_process.drawer import indexed_image_to_rgb


class SegmentationModel(BaseDeepextModel):
    """
    forward
        input:
            image: (Batch size, channels, height, width)
            target: (Batch size, classes, height, width) one-hot
        output:
            (Batch size, classes, height, width)
    """

    @abstractmethod
    def predict_index_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param img: (Batch size, channels, height, width)
        :return: (Batch size, height, width), (Batch size, classes, height, width)
        """
        pass

    def generate_mixed_segment_image(self, origin_image: np.ndarray, indexed_image: np.ndarray, alpha=0.7):
        """
        :param origin_image: (height, width, 3) uint8(0~255)
        :param indexed_image: Value is class index. (height', width') uint8(0~255)
        :param alpha:
        :return:
        """
        assert indexed_image.ndim == 2
        assert origin_image.ndim == 3
        index_color_image = indexed_image_to_rgb(indexed_image)  # (height, width, 3)
        index_color_image = cv2.resize(index_color_image, (origin_image.shape[1], origin_image.shape[0])).astype(
            'uint8')
        return self._blend_img(origin_image, index_color_image, result_alpha=alpha)

    def _blend_img(self, origin_img: np.ndarray, result_img: np.ndarray, origin_alpha=1.0,
                   result_alpha=0.7) -> np.ndarray:
        assert origin_img.ndim == 3
        assert result_img.ndim == 3
        return cv2.addWeighted(origin_img, origin_alpha, result_img, result_alpha, 0)

    def result_to_index_image(self, result: torch.Tensor):
        result = result.permute(0, 2, 3, 1)
        return torch.argmax(result, dim=3)

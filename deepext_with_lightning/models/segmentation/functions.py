from typing import Tuple

import cv2
import numpy as np

from ..base import SegmentationModel
from ...image_process.convert import normalize1, to_4dim, cv_to_tensor, tensor_to_cv

__all__ = ["draw_result_image_to_frame"]


def draw_result_image_to_frame(model: SegmentationModel, frame: np.ndarray, img_size_for_model: Tuple[int, int]):
    """
    :param model:
    :param frame: OpenCV image (height, width, 3)(0~255)
    :param img_size_for_model:
    :return:
    """
    origin_frame = frame.copy()
    frame = cv2.resize(frame, img_size_for_model)
    frame = normalize1(frame)
    img_tensor = to_4dim(cv_to_tensor(frame))
    labels, prob = model.predict_index_image(img_tensor)
    index_image = tensor_to_cv(labels[0]).astype('uint8')
    return model.generate_mixed_segment_image(origin_frame, index_image)

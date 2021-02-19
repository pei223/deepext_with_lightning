from typing import Tuple, List

import cv2
import numpy as np

from ..base import DetectionModel
from ...image_process.convert import normalize1, to_4dim, cv_to_tensor

__all__ = ["draw_result_bboxes_to_frame"]


def draw_result_bboxes_to_frame(model: DetectionModel, frame: np.ndarray, img_size_for_model: Tuple[int, int],
                                label_names: List[str]):
    """
    :param model:
    :param frame: OpenCV image (height, width, 3)(0~255)
    :param img_size_for_model:
    :param label_names:
    :return:
    """
    origin_frame = frame.copy()
    frame = cv2.resize(frame, img_size_for_model)
    frame = normalize1(frame)
    img_tensor = to_4dim(cv_to_tensor(frame))
    result_bboxes = model.predict_bboxes(img_tensor)[0]
    result_img = model.generate_bbox_draw_image(origin_frame, result_bboxes,
                                                model_img_size=img_size_for_model,
                                                label_names=label_names)
    return result_img

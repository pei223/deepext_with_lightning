from typing import Tuple, List

import cv2
import numpy as np

from ..base import AttentionClassificationModel, ClassificationModel
from ...image_process.convert import normalize1, to_4dim, cv_to_tensor, normalize255, tensor_to_cv
from ...image_process.drawer import draw_text_with_background

__all__ = ["draw_result_label_to_frame", "draw_result_label_and_heatmap_to_frame"]


def draw_result_label_to_frame(model: ClassificationModel, frame: np.ndarray,
                               img_size_for_model: Tuple[int, int],
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
    img_tensor = to_4dim(cv_to_tensor(normalize1(frame)))
    label, probs = model.predict_label(img_tensor)
    label = label[0]
    label_prob = probs[0][label]
    result_text = f"Inference result:    {label_names[label]} - {label_prob}%"
    offsets = (0, 40)
    background_color = (255, 255, 255)
    text_color = (0, 0, 255)
    return draw_text_with_background(origin_frame, background_color=background_color, text_color=text_color,
                                     text=result_text, offsets=offsets, font_scale=0.5)


def draw_result_label_and_heatmap_to_frame(model: AttentionClassificationModel, frame: np.ndarray,
                                           img_size_for_model: Tuple[int, int],
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
    img_tensor = to_4dim(cv_to_tensor(normalize1(frame)))
    label, prob, heatmap = model.predict_label_and_heatmap(img_tensor)
    label = label[0]
    label_prob = prob[0][label]
    heatmap = normalize255(tensor_to_cv(heatmap[0]))
    result_img = model.generate_heatmap_image(origin_frame, heatmap)
    result_text = f"Inference result:    {label_names[label]} - {label_prob}%"
    offsets = (0, 80)
    background_color = (255, 255, 255)
    text_color = (0, 0, 255)
    return draw_text_with_background(result_img, background_color=background_color, text_color=text_color,
                                     text=result_text, offsets=offsets, font_scale=0.5)

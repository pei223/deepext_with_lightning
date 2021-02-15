from abc import abstractmethod
from typing import List, Union, Tuple
import torch
import numpy as np
from .base_deepext_model import BaseDeepextModel
from ...image_process.drawer import draw_bounding_boxes_with_name_tag


class DetectionModel(BaseDeepextModel):
    @abstractmethod
    def predict_bboxes(self, imgs: torch.Tensor) -> List[np.ndarray]:
        """
        :param imgs: (Batch size, channels, height, width)
        :return: (Batch size, bounding boxes by batch, 6(xmin, ymin, xmax, ymax, label, score))
        """
        pass

    def generate_bbox_draw_image(self, origin_image: np.ndarray, bboxes: np.ndarray, model_img_size: Tuple[int, int],
                                 label_names: List[str], pred_bbox_color=(0, 0, 255)):
        """
        :param origin_image:
        :param bboxes:
        :param model_img_size:
        :param label_names:
        :param pred_bbox_color:
        :return:
        """
        assert origin_image.ndim == 3
        assert bboxes.ndim == 2
        assert bboxes.shape[-1] == 6, "Bounding box contains (xmin, ymin, xmax, ymax, label, score)"

        bboxes = self._scale_bboxes(bboxes, model_img_size, (origin_image.shape[0], origin_image.shape[1]))
        return self._draw_result_bboxes(origin_image, bboxes, label_names=label_names, pred_color=pred_bbox_color)

    def _scale_bboxes(self, bboxes: Union[np.ndarray, List[List[float or int]]], origin_size: Tuple[int, int],
                      to_size: Tuple[int, int]) -> Union[np.ndarray, List[List[float or int]]]:
        height_rate = to_size[0] / origin_size[0]
        width_rate = to_size[1] / origin_size[1]
        if bboxes is None:
            return bboxes
        for i in range(len(bboxes)):
            if bboxes[i] is None or len(bboxes[i]) == 0:
                continue
            bboxes[i][0], bboxes[i][2] = bboxes[i][0] * width_rate, bboxes[i][2] * width_rate
            bboxes[i][1], bboxes[i][3] = bboxes[i][1] * height_rate, bboxes[i][3] * height_rate
        return bboxes

    def _draw_result_bboxes(self, image: np.ndarray, bboxes: Union[np.ndarray, List[List[float or int]]],
                            label_names: List[str], pred_color=(0, 0, 255)):
        if bboxes is None:
            return image
        for bbox in bboxes:
            if bbox is None or len(bbox) == 0:
                continue
            score = int(bbox[5] * 100.)
            label_name = label_names[int(bbox[4])]
            bbox_text = f"{label_name} {score}%"
            image = draw_bounding_boxes_with_name_tag(image, bboxes, color=pred_color,
                                                      text=bbox_text)
        return image

from typing import Tuple, List
import numpy as np
from .realtime_prediction import RealtimePrediction
from ..models.base import DetectionModel
from ..models.object_detection.functions import draw_result_bboxes_to_frame


class RealtimeDetection(RealtimePrediction):
    def __init__(self, model: DetectionModel, img_size_for_model: Tuple[int, int], label_names: List[str]):
        assert isinstance(model, DetectionModel)
        super().__init__(model, img_size_for_model)
        self.label_names = label_names

    def calc_result(self, frame: np.ndarray):
        result_frame = draw_result_bboxes_to_frame(self.model, frame, self.img_size_for_model, self.label_names)
        return result_frame

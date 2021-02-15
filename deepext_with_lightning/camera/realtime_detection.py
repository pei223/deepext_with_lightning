from typing import Tuple, List
import cv2
import numpy as np
from .realtime_prediction import RealtimePrediction
from ..models.base import DetectionModel


class RealtimeDetection(RealtimePrediction):
    def __init__(self, model: DetectionModel, img_size_for_model: Tuple[int, int], label_names: List[str]):
        assert isinstance(model, DetectionModel)
        super().__init__(model, img_size_for_model)
        self.label_names = label_names

    def calc_result(self, frame: np.ndarray):
        origin_frame_size = frame.shape[:2]
        frame = cv2.resize(frame, self.img_size_for_model)
        return self.model.calc_detection_image(frame, origin_img_size=origin_frame_size, label_names=self.label_names,
                                               require_normalize=True)[1]

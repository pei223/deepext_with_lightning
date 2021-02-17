from typing import Tuple, List
import cv2
import numpy as np
from .realtime_prediction import RealtimePrediction
from ..models.base import DetectionModel
from ..image_process.convert import cv_to_tensor, to_4dim, tensor_to_cv, normalize1


class RealtimeDetection(RealtimePrediction):
    def __init__(self, model: DetectionModel, img_size_for_model: Tuple[int, int], label_names: List[str]):
        assert isinstance(model, DetectionModel)
        super().__init__(model, img_size_for_model)
        self.label_names = label_names

    def calc_result(self, frame: np.ndarray):
        origin_frame = frame.copy()
        frame = cv2.resize(frame, self.img_size_for_model)
        frame = normalize1(frame)
        img_tensor = to_4dim(cv_to_tensor(frame))
        result_bboxes = self.model.predict_bboxes(img_tensor)[0]
        result_img = self.model.generate_bbox_draw_image(origin_frame, result_bboxes,
                                                         model_img_size=self.img_size_for_model,
                                                         label_names=self.label_names)
        return result_img

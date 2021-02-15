from typing import Tuple
import cv2
import numpy as np
from ..models.base import SegmentationModel
from .realtime_prediction import RealtimePrediction
from ..image_process.convert import cv_to_tensor, to_4dim, tensor_to_cv, normalize1


class RealtimeSegmentation(RealtimePrediction):
    def __init__(self, model: SegmentationModel, img_size_for_model: Tuple[int, int]):
        assert isinstance(model, SegmentationModel)
        super().__init__(model, img_size_for_model)

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        origin_frame = frame.copy()
        frame = cv2.resize(frame, self.img_size_for_model)
        frame = normalize1(frame)
        img_tensor = to_4dim(cv_to_tensor(frame))
        labels, prob = self.model.predict_index_image(img_tensor)
        index_image = tensor_to_cv(labels[0]).astype('uint8')
        return self.model.generate_mixed_segment_image(origin_frame, index_image)

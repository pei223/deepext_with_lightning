from typing import Tuple
import numpy as np
from ..models.base import SegmentationModel
from .realtime_prediction import RealtimePrediction
from ..models.segmentation.functions import draw_result_image_to_frame


class RealtimeSegmentation(RealtimePrediction):
    def __init__(self, model: SegmentationModel, img_size_for_model: Tuple[int, int]):
        assert isinstance(model, SegmentationModel)
        super().__init__(model, img_size_for_model)

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        return draw_result_image_to_frame(self.model, frame, self.img_size_for_model)

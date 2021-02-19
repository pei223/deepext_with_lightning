from typing import List, Tuple
import numpy as np
from ..models.base import ClassificationModel, AttentionClassificationModel
from .realtime_prediction import RealtimePrediction
from ..models.classification.functions import draw_result_label_and_heatmap_to_frame, draw_result_label_to_frame


class RealtimeClassification(RealtimePrediction):
    def __init__(self, model: ClassificationModel, img_size_for_model: Tuple[int, int], label_names: List[str]):
        assert isinstance(model, ClassificationModel)
        super().__init__(model, img_size_for_model)
        self._label_names = label_names

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        return draw_result_label_to_frame(self.model, frame, self.img_size_for_model, self._label_names)


class RealtimeAttentionClassification(RealtimePrediction):
    def __init__(self, model: AttentionClassificationModel, img_size_for_model: Tuple[int, int],
                 label_names: List[str]):
        assert isinstance(model, AttentionClassificationModel)
        super().__init__(model, img_size_for_model)
        self._label_names = label_names

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        return draw_result_label_and_heatmap_to_frame(self.model, frame, self.img_size_for_model, self._label_names)

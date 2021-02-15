from typing import List, Tuple
import numpy as np
import cv2
from ..image_process.convert import cv_to_tensor, to_4dim, tensor_to_cv, normalize255
from ..models.base import ClassificationModel, AttentionClassificationModel
from .realtime_prediction import RealtimePrediction
from ..image_process.drawer import draw_text_with_background


class RealtimeClassification(RealtimePrediction):
    def __init__(self, model: ClassificationModel, img_size_for_model: Tuple[int, int], label_names: List[str]):
        assert isinstance(model, ClassificationModel)
        super().__init__(model, img_size_for_model)
        self._label_names = label_names

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, self.img_size_for_model)
        img_tensor = to_4dim(cv_to_tensor(frame))
        label, probs = self.model.predict_label(img_tensor, self.img_size_for_model)
        label = label[0]
        label_prob = probs[0][label]
        result_text = f"Inference result:    {self._label_names[label]} - {label_prob}%"
        offsets = (0, 40)
        background_color = (255, 255, 255)
        text_color = (0, 0, 255)
        return draw_text_with_background(frame, background_color=background_color, text_color=text_color,
                                         text=result_text, offsets=offsets, font_scale=0.5)


class RealtimeAttentionClassification(RealtimePrediction):
    def __init__(self, model: AttentionClassificationModel, img_size_for_model: Tuple[int, int],
                 label_names: List[str]):
        assert isinstance(model, AttentionClassificationModel)
        super().__init__(model, img_size_for_model)
        self._label_names = label_names

    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        origin_frame = frame
        frame = cv2.resize(frame, self.img_size_for_model)
        img_tensor = to_4dim(cv_to_tensor(frame))
        label, prob, heatmap = self.model.predict_label_and_heatmap(img_tensor)
        heatmap = normalize255(tensor_to_cv(heatmap[0]))
        result_img = self.model.generate_heatmap_image(origin_frame, heatmap)
        result_text = f"Inference result:    {self._label_names[label]}"
        offsets = (0, 80)
        background_color = (255, 255, 255)
        text_color = (0, 0, 255)
        # TODO calc_result終わった後に文章を載せないとぼやける
        return draw_text_with_background(result_img, background_color=background_color, text_color=text_color,
                                         text=result_text, offsets=offsets, font_scale=0.5)

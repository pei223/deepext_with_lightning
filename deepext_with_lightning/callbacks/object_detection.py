from typing import Tuple, List

import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from pytorch_lightning.callbacks import Callback

from ..image_process.drawer import draw_bounding_boxes_with_name_tag
from ..models.base import DetectionModel
from ..image_process.convert import cv_to_pil, to_4dim, tensor_to_cv, normalize255


class GenerateDetectionImageCallback(Callback):
    def __init__(self, model: DetectionModel, img_size: Tuple[int, int], dataset: Dataset, out_dir: str,
                 label_names: List[str], per_epoch: int = 10, pred_color=(0, 0, 255), teacher_color=(0, 255, 0),
                 apply_all_images=False):
        """
        :param model:
        :param img_size: (H, W)
        :param dataset:
        :param out_dir:
        :param per_epoch:
        :param pred_color:
        :param teacher_color:
        """
        self._model = model
        self._dataset = dataset
        self._pred_color = pred_color
        self._teacher_color = teacher_color
        self._per_epoch = per_epoch
        self._out_dir = out_dir
        self._img_size = img_size
        self._label_names = label_names
        self._apply_all_images = apply_all_images
        if not Path(self._out_dir).exists():
            Path(self._out_dir).mkdir()

    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self._per_epoch != 0:
            return
        if self._apply_all_images:
            for i, (img_tensor, teacher_bboxes) in enumerate(self._dataset):
                origin_image = normalize255(tensor_to_cv(img_tensor))
                result_bboxes = self._model.predict_bboxes(to_4dim(img_tensor))[0]
                result_img = self._model.generate_bbox_draw_image(origin_image, result_bboxes,
                                                                  model_img_size=self._img_size,
                                                                  label_names=self._label_names)
                result_img = self._draw_teacher_bboxes(result_img, teacher_bboxes)
                cv_to_pil(result_img).save(f"{self._out_dir}/data{i}_image{epoch + 1}.png")
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, teacher_bboxes = self._dataset[random_image_index]

        origin_image = normalize255(tensor_to_cv(img_tensor))
        result_bboxes = self._model.predict_bboxes(to_4dim(img_tensor))[0]
        result_img = self._model.generate_bbox_draw_image(origin_image, result_bboxes,
                                                          model_img_size=self._img_size,
                                                          label_names=self._label_names)
        result_img = self._draw_teacher_bboxes(result_img, teacher_bboxes)
        cv_to_pil(result_img).save(f"{self._out_dir}/image{epoch + 1}.png")

    def _draw_teacher_bboxes(self, image: np.ndarray, teacher_bboxes: List[Tuple[float, float, float, float, int]]):
        """
        :param image:
        :param teacher_bboxes: List of [x_min, y_min, x_max, y_max, label]
        :return:
        """
        if teacher_bboxes is None or len(teacher_bboxes) == 0:
            return image
        for bbox in teacher_bboxes:
            image = draw_bounding_boxes_with_name_tag(image, [bbox], color=self._teacher_color, text=None)
        return image

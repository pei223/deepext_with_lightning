import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from pytorch_lightning.callbacks import Callback
from deepext_with_lightning.models.base import SegmentationModel
from ..image_process.convert import cv_to_pil, to_4dim, tensor_to_cv, normalize255


class GenerateSegmentationImageCallback(Callback):
    def __init__(self, model: SegmentationModel, output_dir: str, per_epoch: int, dataset: Dataset, alpha=0.6,
                 apply_all_images=False):
        self._model: SegmentationModel = model
        self._output_dir = output_dir
        self._per_epoch = per_epoch
        self._dataset = dataset
        self._alpha = alpha
        self._apply_all_images = apply_all_images
        if not Path(self._output_dir).exists():
            Path(self._output_dir).mkdir(parents=True)

    def on_epoch_end(self, trainer, _):
        epoch = trainer.current_epoch
        if (epoch + 1) % self._per_epoch != 0:
            return
        if self._apply_all_images:
            for i, (img_tensor, label) in enumerate(self._dataset):
                origin_image = normalize255(tensor_to_cv(img_tensor))
                prob, pred_label = self._model.predict_index_image(to_4dim(img_tensor))
                index_image = tensor_to_cv(pred_label[0]).astype('uint8')
                mixed_img = self._model.generate_mixed_segment_image(origin_image, index_image)
                cv_to_pil(mixed_img).save(f"{self._output_dir}/data{i}_image{epoch + 1}.png")
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, _ = self._dataset[random_image_index]
        origin_image = normalize255(tensor_to_cv(img_tensor))
        pred_label, prob = self._model.predict_index_image(to_4dim(img_tensor))
        index_image = tensor_to_cv(pred_label[0])
        mixed_img = self._model.generate_mixed_segment_image(origin_image, index_image, self._alpha)
        cv_to_pil(mixed_img).save(f"{self._output_dir}/result_image{epoch + 1}.png")

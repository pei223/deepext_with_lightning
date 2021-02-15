import numpy as np
from typing import List

from torch.utils.data import Dataset
import tqdm
from pathlib import Path
from pytorch_lightning.callbacks import Callback

from ..image_process.convert import img_to_cv, cv_to_pil, to_4dim, normalize255, tensor_to_cv
from ..models.base import AttentionClassificationModel, ClassificationModel


class GenerateAttentionMap(Callback):
    def __init__(self, output_dir: str, period: int, dataset: Dataset, model: AttentionClassificationModel,
                 label_names: List[str], apply_all_images=False):
        self._out_dir, self._period, self._dataset = output_dir, period, dataset
        self._model = model
        self._label_names = label_names
        self._apply_all_images = apply_all_images
        if not Path(self._out_dir).exists():
            Path(self._out_dir).mkdir(parents=True)

    def on_epoch_end(self, trainer, _):
        epoch = trainer.current_epoch
        if (epoch + 1) % self._period != 0:
            return
        if self._apply_all_images:
            for i, (img_tensor, label) in enumerate(self._dataset):
                origin_image = normalize255(tensor_to_cv(img_tensor))
                pred_label, pred_prob, attention_map = self._model.predict_label_and_heatmap(to_4dim(img_tensor))
                attention_map = normalize255(tensor_to_cv(attention_map[0]))
                blend_img = self._model.generate_heatmap_image(origin_image, attention_map)
                cv_to_pil(blend_img).save(self._image_path(epoch, pred_label[0], label, f"data{i}_"))
            return
        data_len = len(self._dataset)
        random_image_index = np.random.randint(0, data_len)
        img_tensor, label = self._dataset[random_image_index]
        origin_image = normalize255(tensor_to_cv(img_tensor))
        pred_label, pred_prob, attention_map = self._model.predict_label_and_heatmap(to_4dim(img_tensor))
        attention_map = normalize255(tensor_to_cv(attention_map[0]))
        blend_img = self._model.generate_heatmap_image(origin_image, attention_map)
        cv_to_pil(blend_img).save(self._image_path(epoch, pred_label[0], label))

    def _image_path(self, epoch: int, pred_label: int, label: int, prefix=""):
        return f"{self._out_dir}/{prefix}epoch{epoch + 1}_T_{self._label_names[label]}_P_{self._label_names[pred_label]}.png"


class CSVClassificationResult(Callback):
    def __init__(self, out_filepath: str, period: int, dataset: Dataset, model: ClassificationModel,
                 label_names: List[str]):
        self._model = model
        self._out_filepath = out_filepath
        self._period = period
        self._dataset = dataset
        self._label_names = label_names

    def on_epoch_end(self, trainer, _):
        epoch = trainer.current_epoch
        if (epoch + 1) % self._period != 0:
            return
        with open(self._out_filepath, "w") as file:
            file.write("number,predict class,teacher class,predict name,teacher name\n")
            for i, (img_tensor, label) in enumerate(tqdm.tqdm(self._dataset)):
                img_tensor = to_4dim(img_tensor)
                pred_label, pred_prob = self._model.predict_label(img_tensor)
                pred_label_num = pred_label[0]
                file.write(
                    f"{i},{pred_label_num},{label},{self._label_names[pred_label_num]},{self._label_names[label]}\n")

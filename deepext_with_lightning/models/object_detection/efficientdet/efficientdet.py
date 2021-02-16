from typing import List

import torch
import numpy as np
import pytorch_lightning as pl

from ....models.base.detection_model import DetectionModel
from .efficientdet_lib.models.efficientdet import EfficientDet
from ....metrics.object_detection import DetectionIoU, MeanAveragePrecision
from .efficientdet_lib.utils import EFFICIENTDET
from ....image_process.convert import try_cuda

__all__ = ['EfficientDetector']


class EfficientDetector(DetectionModel):
    def __init__(self, n_classes, network='efficientdet-d0', lr=1e-4, score_threshold=0.5, max_detections=50,
                 backbone_path: str = None, backbone_pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        self._model = try_cuda(EfficientDet(num_classes=n_classes,
                                            network=network,
                                            W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                                            D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                                            D_class=EFFICIENTDET[network]['D_class'], backbone_path=backbone_path,
                                            backbone_pretrained=backbone_pretrained,
                                            threshold=score_threshold))
        self._n_classes = n_classes
        self._network = network
        self._max_detections = max_detections
        self._lr = lr
        # TODO Metric
        self._train_map: pl.metrics.Metric = try_cuda(MeanAveragePrecision(n_classes))
        self._val_map: pl.metrics.Metric = try_cuda(MeanAveragePrecision(n_classes))
        self._train_iou: pl.metrics.Metric = try_cuda(DetectionIoU(n_classes))
        self._val_iou: pl.metrics.Metric = try_cuda(DetectionIoU(n_classes))

    def forward(self, x: torch.Tensor):
        return self._model(x)

    def predict_bboxes(self, imgs: torch.Tensor) -> List[np.ndarray]:
        self._model.eval()
        self._model.is_training = False
        assert imgs.ndim == 4

        result = []
        for i in range(imgs.shape[0]):
            # TODO ここtensorにしたいけど公式がnumpy
            image = imgs[i].float().unsqueeze(0)
            scores, labels, boxes = self._model(try_cuda(image))
            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            boxes = boxes.detach().cpu().numpy()
            scores_sort = np.argsort(-scores)[:self._max_detections]
            # select detections
            image_boxes = boxes[scores_sort, :]
            image_scores = scores[scores_sort]
            image_labels = labels[scores_sort]
            image_detections = np.concatenate([
                image_boxes,
                np.expand_dims(image_labels, axis=1),
                np.expand_dims(image_scores, axis=1),
            ], axis=1)
            result.append(image_detections)
        return result

    def training_step(self, batch, batch_idx):
        self._model.train()
        self._model.is_training = True
        self._model.freeze_bn()

        inputs, targets = batch
        annotations = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets
        annotations = try_cuda(annotations.float())
        images = try_cuda(inputs)
        classification_loss, regression_loss = self._model([images, annotations])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self._model.eval()
        self._model.is_training = False

        inputs, targets = batch
        targets = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets
        inputs, targets = try_cuda(inputs).float(), try_cuda(targets).long()
        result = self.predict_bboxes(inputs)
        self._val_iou(result, targets)
        self._val_map(result, targets)

    def on_validation_epoch_end(self) -> None:
        val_iou_value = self._val_iou.compute()
        val_map_value = self._val_map.compute()
        self.log("val_iou", val_iou_value, on_step=False, on_epoch=True)
        self.log("val_map", val_map_value, on_step=False, on_epoch=True)
        self._val_iou.reset()
        self._val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self._lr, params=self._model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, mode="min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
            'interval': 'epoch',
            'frequency': 1,
        }

    def generate_model_name(self, suffix: str = "") -> str:
        return f"{self._network}{suffix}"

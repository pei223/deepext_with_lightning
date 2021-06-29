from typing import Tuple, List, Any
from warnings import warn

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ....image_process.convert import try_cuda
from ...base.segmentation_model import SegmentationModel

from .modules import ShelfNetModel, ShelfNetModelWithEfficientNet
from ...layers.backbone_key import BackBoneKey
from ...layers.loss import AdaptiveCrossEntropyLoss
from ....metrics.segmentation import SegmentationIoU


class ShelfNet(SegmentationModel):
    def __init__(self, n_classes: int, out_size: Tuple[int, int], lr=1e-3, loss_func: nn.Module = None,
                 backbone: BackBoneKey = BackBoneKey.RESNET_18, backbone_pretrained=True, aux_weight=0.5):
        """
        :param n_classes:
        :param out_size: (height, width)
        :param lr:
        :param loss_func:
        :param backbone:
        :param backbone_pretrained:
        """
        super().__init__()
        self.save_hyperparameters()
        self._n_classes = n_classes
        if BackBoneKey.is_efficentnet(backbone):
            self._model: nn.Module = try_cuda(
                ShelfNetModelWithEfficientNet(n_classes=n_classes, out_size=out_size, backbone=backbone,
                                              pretrained=backbone_pretrained))
        else:
            self._model: nn.Module = try_cuda(
                ShelfNetModel(n_classes=n_classes, out_size=out_size, backbone=backbone,
                              pretrained=backbone_pretrained))
        self._lr = lr
        self._loss_func = loss_func if loss_func else AdaptiveCrossEntropyLoss()
        self._backbone = backbone
        self._train_iou: pl.metrics.Metric = try_cuda(SegmentationIoU(n_classes))
        self._val_iou: pl.metrics.Metric = try_cuda(SegmentationIoU(n_classes))
        self._aux_weight = aux_weight

    def forward(self, x):
        return self._model(x)

    def predict_index_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self._model.eval()
            img = try_cuda(img)
            pred = self(img)[0]
            return self.result_to_index_image(pred), pred

    def training_step(self, batch, batch_idx):
        self._model.train()
        inputs, targets = batch
        inputs, targets = try_cuda(inputs).float(), try_cuda(targets).float()
        pred, pred_b, pred_c = self(inputs)[:3]
        loss_a = self._loss_func(pred, targets)
        loss_b = self._loss_func(pred_b, targets)
        loss_c = self._loss_func(pred_c, targets)
        loss = loss_a + loss_b * self._aux_weight + loss_c * self._aux_weight
        self.log('train_loss', loss)
        pred_labels = self.result_to_index_image(pred)
        targets = self.result_to_index_image(targets)
        return {'loss': loss, 'preds': pred_labels, 'target': targets}

    def training_step_end(self, outputs) -> None:
        preds, targets = outputs["preds"], outputs["target"]
        self._train_iou(preds, targets)
        return outputs["loss"]

    def training_epoch_end(self, outputs: List[Any]) -> None:
        acc_val = self._train_iou.compute()
        self.log('train_iou', acc_val, on_step=False, on_epoch=True)
        self._train_iou.reset()

    def validation_step(self, batch, batch_idx):
        self._model.eval()
        inputs, targets = batch
        inputs, targets = try_cuda(inputs).float(), try_cuda(targets).float()
        pred_prob = self(inputs)[0]
        pred_labels = self.result_to_index_image(pred_prob)
        targets = self.result_to_index_image(targets)
        self._val_iou(pred_labels, targets)

    def on_validation_epoch_end(self) -> None:
        acc_val = self._val_iou.compute()
        self.log("val_iou", acc_val, on_step=False, on_epoch=True)
        self._val_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self._lr, params=self._model.parameters(), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, mode="min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
            'interval': 'epoch',
            'frequency': 1,
        }

    def generate_model_name(self, suffix: str = "") -> str:
        return super().generate_model_name(f'_{self._backbone.value}{suffix}')

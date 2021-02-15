from typing import List, Any, Tuple

import torch
from torch.nn import functional as F
from ...base import ClassificationModel
from .mobilenetv3_lib.model import MobileNetV3 as MobileNetV3lib
from ....image_process.convert import try_cuda
import pytorch_lightning as pl

__all__ = ['MobileNetV3']


class MobileNetV3(ClassificationModel):
    def __init__(self, num_classes, lr=1e-4, mode='small', pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        self._num_classes = num_classes
        self._mode = mode
        self._model = try_cuda(MobileNetV3lib(num_classes=num_classes, mode=mode))
        self._lr = lr
        self._train_acc: pl.metrics.Metric = try_cuda(pl.metrics.Accuracy(compute_on_step=False))
        self._val_acc: pl.metrics.Metric = try_cuda(pl.metrics.Accuracy(compute_on_step=False))

    def forward(self, x):
        result = self._model(x)
        return F.softmax(result, dim=1)

    def predict_label(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._model.eval()
        img = try_cuda(img)
        pred_prob = self._model(img)
        return torch.argmax(pred_prob, dim=1), pred_prob

    def training_step(self, batch, batch_idx):
        self._model.train()
        inputs, targets = batch
        targets = self.onehot_to_label(targets)
        inputs, targets = try_cuda(inputs).float(), try_cuda(targets).long()
        pred_prob = self._model(inputs)
        pred_labels = torch.argmax(pred_prob, dim=1)
        loss = F.cross_entropy(pred_prob, targets, reduction="mean")
        self.log('train_loss', loss)
        return {'loss': loss, 'preds': pred_labels, 'target': targets}

    def training_step_end(self, outputs) -> None:
        preds, targets = outputs["preds"], outputs["target"]
        self._train_acc(preds, targets)
        return outputs["loss"]

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self._train_acc.compute()
        self.log('train_acc', self._train_acc, on_step=False, on_epoch=True)
        self.logger.log_hyperparams(self.hparams)
        self._train_acc.reset()

    def validation_step(self, batch, batch_idx):
        self._model.eval()
        inputs, targets = batch
        targets = self.onehot_to_label(targets)
        inputs, targets = try_cuda(inputs).float(), try_cuda(targets).long()
        pred_prob = self._model(inputs)
        pred_labels = torch.argmax(pred_prob, dim=1)
        self._val_acc(pred_labels, targets)

    def on_validation_epoch_end(self) -> None:
        self._val_acc.compute()
        self.log("val_acc", self._val_acc, on_step=False, on_epoch=True)
        self._val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self._lr, params=self._model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer, ], [scheduler, ]

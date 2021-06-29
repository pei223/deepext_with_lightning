from typing import Tuple, List, Any
from warnings import warn

import torch
from torch import nn
import pytorch_lightning as pl

from ....image_process.convert import try_cuda
from ....models.base import SegmentationModel

from ...layers.loss import AdaptiveCrossEntropyLoss
from ....metrics.segmentation import SegmentationIoU

from .modules import UNetModel, ResUNetModel


class UNet(SegmentationModel):
    def __init__(self, n_classes: int, n_input_channels: int = 3, first_layer_channels: int = 32, lr=1e-3,
                 loss_func: nn.Module = None):
        super().__init__()
        self.save_hyperparameters()
        self.n_channels, self.n_classes = n_input_channels, n_classes
        self._model = try_cuda(UNetModel(n_input_channels, n_classes, first_layer_channels))

        self._loss_func = loss_func if loss_func else AdaptiveCrossEntropyLoss()

        self._train_iou: pl.metrics.Metric = try_cuda(SegmentationIoU(n_classes))
        self._val_iou: pl.metrics.Metric = try_cuda(SegmentationIoU(n_classes))
        self._lr = lr

    def forward(self, x: torch.Tensor):
        return self._model(x)

    def predict_index_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self._model.eval()
            img = try_cuda(img)
            pred = self(img)
            return self.result_to_index_image(pred), pred

    def training_step(self, batch, batch_idx):
        self._model.train()
        inputs, targets = batch
        inputs, targets = try_cuda(inputs).float(), try_cuda(targets).float()
        pred = self(inputs)
        loss = self._loss_func(pred, targets)
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
        pred_prob = self(inputs)
        pred_labels = self.result_to_index_image(pred_prob)
        targets = self.result_to_index_image(targets)
        self._val_iou(pred_labels, targets)

    def on_validation_epoch_end(self) -> None:
        acc_val = self._val_iou.compute()
        self.log("val_iou", acc_val, on_step=False, on_epoch=True)
        self._val_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self._lr, params=self._model.parameters())
        if self.trainer:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        else:
            warn("UNet#configure_optimizers: Trainer pointer not found.")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)

        return [optimizer, ], [scheduler, ]


class ResUNet(SegmentationModel):
    def __init__(self, n_classes: int, n_input_channels: int = 3, first_layer_channels: int = 32, lr=1e-3,
                 loss_func: nn.Module = None):
        super().__init__()
        self.save_hyperparameters()
        self.n_channels, self.n_classes = n_input_channels, n_classes
        self._model = try_cuda(ResUNetModel(n_input_channels, n_classes, first_layer_channels))

        self._loss_func = loss_func if loss_func else AdaptiveCrossEntropyLoss()

        self._train_iou: pl.metrics.Metric = try_cuda(SegmentationIoU(n_classes))
        self._val_iou: pl.metrics.Metric = try_cuda(SegmentationIoU(n_classes))
        self._lr = lr

    def forward(self, x: torch.Tensor):
        return self._model(x)

    def predict_index_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._model.eval()
        img = try_cuda(img)
        pred = self(img)
        return self.result_to_index_image(pred), pred

    def training_step(self, batch, batch_idx):
        self._model.train()
        inputs, targets = batch
        inputs, targets = try_cuda(inputs).float(), try_cuda(targets).float()
        pred = self(inputs)
        loss = self._loss_func(pred, targets)
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
        pred_prob = self(inputs)
        pred_labels = self.result_to_index_image(pred_prob)
        targets = self.result_to_index_image(targets)
        self._val_iou(pred_labels, targets)

    def on_validation_epoch_end(self) -> None:
        acc_val = self._val_iou.compute()
        self.log("val_iou", acc_val, on_step=False, on_epoch=True)
        self._val_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self._lr, params=self._model.parameters())
        if self.trainer:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        else:
            warn("UNet#configure_optimizers: Trainer pointer not found.")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)

        return [optimizer, ], [scheduler, ]

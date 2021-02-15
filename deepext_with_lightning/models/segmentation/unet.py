from typing import Tuple, List, Any
from warnings import warn

import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl

from ...image_process.convert import try_cuda
from ..layers.basic import BottleNeck
from ..layers.loss import AdaptiveCrossEntropyLoss
from ..layers.block import DownBlock, UpBlock
from ..base import SegmentationModel
from ...metrics.segmentation import SegmentationIoU

__all__ = ['UNet', 'ResUNet']


class UNet(SegmentationModel):
    def __init__(self, n_classes: int, n_input_channels: int = 3, first_layer_channels: int = 64, lr=1e-3,
                 loss_func: nn.Module = None):
        super().__init__()
        self.save_hyperparameters()
        self.n_channels, self.n_classes = n_input_channels, n_classes
        self._model = UNetModel(n_input_channels, n_classes, first_layer_channels)

        self._optimizer = torch.optim.Adam(lr=lr, params=self._model.parameters())
        self._loss_func = loss_func if loss_func else AdaptiveCrossEntropyLoss()

        self._train_iou: pl.metrics.Metric = try_cuda(SegmentationIoU(n_classes))
        self._val_iou: pl.metrics.Metric = try_cuda(SegmentationIoU(n_classes))

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

    def generate_model_name(self, suffix: str = "") -> str:
        return super().generate_model_name(f'_{self._backbone.value}{suffix}')


class ResUNet(UNet):
    def __init__(self, n_classes: int, n_input_channels: int = 3, lr=1e-3, loss_func: nn.Module = None):
        super().__init__(n_classes, n_input_channels, lr=lr, loss_func=loss_func)

    def down_sampling_layer(self, n_input_channels: int, n_out_channels: int):
        return BottleNeck(n_input_channels, mid_channels=n_input_channels, out_channels=n_out_channels, stride=2)


class UNetModel(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, first_layer_channels: int = 64):
        super().__init__()
        self._first_layer_channels = first_layer_channels
        self._encoder_layer1 = self.down_sampling_layer(n_input_channels, first_layer_channels)
        self._encoder_layer2 = self.down_sampling_layer(first_layer_channels, first_layer_channels * 2)
        self._encoder_layer3 = self.down_sampling_layer(first_layer_channels * 2, first_layer_channels * 4)
        self._encoder_layer4 = self.down_sampling_layer(first_layer_channels * 4, first_layer_channels * 8)
        self._encoder_layer5 = self.down_sampling_layer(first_layer_channels * 8, first_layer_channels * 16)

        self._decoder_layer1 = self.up_sampling_layer(first_layer_channels * 16, first_layer_channels * 8)
        self._decoder_layer2 = self.up_sampling_layer(first_layer_channels * 16, first_layer_channels * 4)
        self._decoder_layer3 = self.up_sampling_layer(first_layer_channels * 8, first_layer_channels * 2)
        self._decoder_layer4 = self.up_sampling_layer(first_layer_channels * 4, first_layer_channels)
        self._decoder_layer5 = self.up_sampling_layer(first_layer_channels * 2, n_output_channels, is_output_layer=True)

        self.apply(init_weights_func)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        :param x: (batch size, channels, height, width)
        :return:  (batch size, class, height, width)
        """
        x = try_cuda(x)
        enc1 = self._encoder_layer1(x)
        enc2 = self._encoder_layer2(enc1)
        enc3 = self._encoder_layer3(enc2)
        enc4 = self._encoder_layer4(enc3)
        encoded_feature = self._encoder_layer5(enc4)

        x = self._decoder_layer1(encoded_feature)
        x = torch.cat([x, enc4], dim=1)
        x = self._decoder_layer2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self._decoder_layer3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self._decoder_layer4(x)
        x = torch.cat([x, enc1], dim=1)
        output = self._decoder_layer5(x)
        return output

    def down_sampling_layer(self, n_input_channels: int, n_out_channels: int):
        # 継承することでエンコーダーにResnetBlocなど適用可能
        return DownBlock(n_input_channels, n_out_channels)

    def up_sampling_layer(self, n_input_channels: int, n_out_channels: int, is_output_layer=False):
        # 継承することでエンコーダーにResnetBlocなど適用可能
        return UpBlock(n_input_channels, n_out_channels, is_output_layer)


def init_weights_func(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)

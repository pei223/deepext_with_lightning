from typing import Tuple, List, Any

import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

from ..base.classification_model import ClassificationModel
from deepext_with_lightning.models.layers.backbone_key import BackBoneKey, BACKBONE_CHANNEL_COUNT_DICT
from deepext_with_lightning.models.layers.subnetwork import create_backbone, ClassifierHead
from ...image_process.convert import try_cuda


class CustomClassificationNetwork(ClassificationModel):
    def __init__(self, n_classes: int, pretrained=True, backbone: BackBoneKey = BackBoneKey.RESNET_50, n_blocks=3,
                 lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self._backbone = backbone
        self._model = try_cuda(
            CustomClassificationModel(n_classes=n_classes, pretrained=pretrained, backbone=backbone, n_blocks=n_blocks))
        self._n_classes = n_classes
        self._n_blocks = n_blocks
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

    def generate_model_name(self, suffix: str = "") -> str:
        return super().generate_model_name(f'_{self._backbone.value}{suffix}')


class CustomClassificationModel(nn.Module):
    def __init__(self, n_classes: int, pretrained=True,
                 backbone: BackBoneKey = BackBoneKey.RESNET_50, n_blocks=3):
        super().__init__()
        self.feature_extractor = create_backbone(backbone_key=backbone, pretrained=pretrained)
        feature_channel_num = BACKBONE_CHANNEL_COUNT_DICT[backbone][-1]
        self.perception_branch = ClassifierHead(in_channels=feature_channel_num, n_blocks=n_blocks,
                                                n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (batch size, channels, height, width)
        :return: (batch size, class), (batch size, class), heatmap (batch size, 1, height, width)
        """
        origin_feature = self.feature_extractor(x)[-1]
        return self.perception_branch(origin_feature)

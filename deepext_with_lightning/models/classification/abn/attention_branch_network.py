from typing import Tuple, List, Any
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

from ...base import AttentionClassificationModel
from ...classification.abn.modules import ABNModel
from ...layers.backbone_key import BackBoneKey
from ....image_process.convert import try_cuda


class AttentionBranchNetwork(AttentionClassificationModel):
    def __init__(self, n_classes: int, pretrained=True, backbone: BackBoneKey = BackBoneKey.RESNET_50, n_blocks=3,
                 lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self._backbone = backbone
        self._n_classes = n_classes
        self._n_blocks = n_blocks
        self._lr = lr
        self._model = try_cuda(
            ABNModel(n_classes=n_classes, pretrained=pretrained, backbone=backbone, n_blocks=n_blocks))
        self._train_acc: pl.metrics.Metric = try_cuda(pl.metrics.Accuracy(compute_on_step=False))
        self.val_acc: pl.metrics.Metric = try_cuda(pl.metrics.Accuracy(compute_on_step=False))

    def forward(self, x):
        labels, attention_labels, attention_map = self._model(x)
        labels = F.softmax(labels, dim=1)
        attention_labels = F.softmax(attention_labels, dim=1)
        return labels, attention_labels, attention_map

    def training_step(self, batch, batch_idx):
        self._model.train()
        inputs, targets = batch
        targets = self.onehot_to_label(targets)
        inputs, targets = try_cuda(inputs).float(), try_cuda(targets).long()
        out = self._model(inputs)
        loss = self._calc_loss(out, targets)
        pred_prob, _, attention_map = out
        pred_labels = torch.argmax(pred_prob, dim=1)
        self.log('train_loss', loss)
        return {'loss': loss, 'preds': pred_labels, 'target': targets}

    def training_step_end(self, outputs):
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
        pred_prob, _, attention_map = self._model(inputs)
        pred_labels = torch.argmax(pred_prob, dim=1)
        self.val_acc(pred_labels, targets)

    def on_validation_epoch_end(self) -> None:
        self.val_acc.compute()
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self._lr, params=self._model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer, ], [scheduler, ]

    def _calc_loss(self, output, teacher):
        teacher = teacher.long()
        perception_pred, attention_pred, attention_map = output
        perception_pred = F.softmax(perception_pred, dim=1)
        attention_pred = F.softmax(attention_pred, dim=1)
        return F.cross_entropy(perception_pred, teacher, reduction="mean") + \
               F.cross_entropy(attention_pred, teacher, reduction="mean")
        # perception_pred = F.sigmoid(perception_pred)
        # attention_pred = F.sigmoid(attention_pred)
        # return F.binary_cross_entropy(perception_pred, teacher, reduction="mean") + F.binary_cross_entropy(
        #     attention_pred, teacher, reduction="mean")

    def predict_label_and_heatmap(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._model.eval()
        x = try_cuda(x).float()
        pred, _, heatmap = self._model(x)
        heatmap = heatmap[:, 0]
        heatmap = self._normalize_heatmap(heatmap)
        return torch.argmax(pred, dim=1), pred, heatmap

    def predict_label(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._model.eval()
        img = try_cuda(img).float()
        pred_prob, _, heatmap = self._model(img)
        return torch.argmax(pred_prob, dim=1), pred_prob

    def generate_model_name(self, suffix: str = "") -> str:
        return f'{self.__class__.__name__}_{self._backbone.value}{suffix}'

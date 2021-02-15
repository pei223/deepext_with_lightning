from typing import Tuple, List, Any
from warnings import warn

import torch
import torch.nn as nn
import pytorch_lightning as pl
from ....image_process.convert import try_cuda
from ...base.segmentation_model import SegmentationModel

from .modules import SegmentationShelf, OutLayer
from ...layers.backbone_key import BackBoneKey, BACKBONE_CHANNEL_COUNT_DICT
from ...layers.basic import Conv2DBatchNormRelu
from ...layers.subnetwork import create_backbone
from ...layers.loss import AdaptiveCrossEntropyLoss
from ....metrics.segmentation import SegmentationIoU

__all__ = ['ShelfNet']


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
        # optimizer = torch.optim.SGD(lr=self._lr, params=self._model.parameters(), weight_decay=1e-4)
        if self.trainer:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs,
                                                                   verbose=True)
        else:
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer)
            warn("Shelfnet#configure_optimizers: Trainer pointer not found.")
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, mode="min")

        # return optimizer
        return [optimizer, ], [scheduler, ]

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     # "monitor": "train_loss",
        #     'interval': 'epoch',
        #     'frequency': 1,
        #     # 'strict': True,
        #     # "mode": "min"
        # }

    def generate_model_name(self, suffix: str = "") -> str:
        return super().generate_model_name(f'_{self._backbone.value}{suffix}')


class ShelfNetModel(nn.Module):
    def __init__(self, n_classes: int, out_size: Tuple[int, int], backbone: BackBoneKey = BackBoneKey.RESNET_18,
                 pretrained=True):
        super().__init__()

        backbone_out_channel_ls = BACKBONE_CHANNEL_COUNT_DICT[backbone]
        assert backbone_out_channel_ls is not None, "Invalid backbone type."
        if backbone in [BackBoneKey.RESNET_18]:
            mid_channel_ls = [64, 128, 256, 512]
        elif backbone in [BackBoneKey.RESNET_34, BackBoneKey.RESNET_50, BackBoneKey.RESNEST_50,
                          BackBoneKey.RESNEXT_50, ]:
            mid_channel_ls = [128, 256, 512, 1024]
        elif backbone in [BackBoneKey.RESNET_101, BackBoneKey.RESNET_152,
                          BackBoneKey.RESNEXT_101, BackBoneKey.RESNEST_101,
                          BackBoneKey.RESNEST_200, BackBoneKey.RESNEST_269]:
            mid_channel_ls = [256, 512, 1024, 2048]
        else:
            assert False, "Invalid backbone type."

        self._feature_and_layer_diff = len(backbone_out_channel_ls) - len(mid_channel_ls)
        assert self._feature_and_layer_diff >= 0, "Shelfのレイヤー数はBackBoneのレイヤー数以下の必要があります."
        reducers = []
        for i in range(len(mid_channel_ls)):
            reducers.append(
                Conv2DBatchNormRelu(kernel_size=1,
                                    in_channels=backbone_out_channel_ls[self._feature_and_layer_diff + i],
                                    out_channels=mid_channel_ls[i], padding=0, bias=False))
        self._reducers = nn.ModuleList(reducers)

        self._multi_scale_backbone = create_backbone(backbone_key=backbone, pretrained=pretrained)
        self._segmentation_shelf = SegmentationShelf(in_channels_ls=mid_channel_ls)

        out_convs = []
        for i in range(len(mid_channel_ls)):
            out_convs.append(OutLayer(in_channels=mid_channel_ls[i], mid_channels=mid_channel_ls[0],
                                      n_classes=n_classes, out_size=out_size))
        self._out_convs = nn.ModuleList(out_convs)

    def forward(self, x: torch.Tensor, aux: bool = True) -> List[torch.Tensor]:
        x_list = self._multi_scale_backbone(x)
        reduced_x_list = []

        for i, reducer_layer in enumerate(self._reducers):
            reduced_x_list.append(reducer_layer(x_list[i + self._feature_and_layer_diff]))
        x_list = self._segmentation_shelf(reduced_x_list)
        outs = []
        for i, out_conv_layer in enumerate(self._out_convs):
            outs.append(out_conv_layer(x_list[i]))
        return outs


class ShelfNetModelWithEfficientNet(nn.Module):
    def __init__(self, n_classes: int, out_size: Tuple[int, int], backbone: BackBoneKey = BackBoneKey.EFFICIENTNET_B0,
                 pretrained=True):
        super().__init__()

        if backbone in [BackBoneKey.EFFICIENTNET_B0, BackBoneKey.EFFICIENTNET_B1, BackBoneKey.EFFICIENTNET_B2]:
            mid_channel_ls = [64, 128, 256, 512]
        elif backbone in [BackBoneKey.EFFICIENTNET_B3, BackBoneKey.EFFICIENTNET_B4, BackBoneKey.EFFICIENTNET_B5]:
            mid_channel_ls = [128, 256, 512, 1024]
        elif backbone in [BackBoneKey.EFFICIENTNET_B6, BackBoneKey.EFFICIENTNET_B7]:
            mid_channel_ls = [128, 256, 512, 1024]
        else:
            assert False, "Invalid backbone type."
        backbone_out_channel_ls = BACKBONE_CHANNEL_COUNT_DICT[backbone][1:5]

        self._feature_and_layer_diff = len(backbone_out_channel_ls) - len(mid_channel_ls)
        assert self._feature_and_layer_diff >= 0, "Shelfのレイヤー数はBackBoneのレイヤー数以下の必要があります."
        reducers = []
        for i in range(len(mid_channel_ls)):
            reducers.append(
                Conv2DBatchNormRelu(kernel_size=1,
                                    in_channels=backbone_out_channel_ls[self._feature_and_layer_diff + i],
                                    out_channels=mid_channel_ls[i], padding=0))
        self._reducers = nn.ModuleList(reducers)

        self._multi_scale_backbone = create_backbone(backbone_key=backbone, pretrained=pretrained)
        self._segmentation_shelf = SegmentationShelf(in_channels_ls=mid_channel_ls)

        out_convs = []
        for i in range(len(mid_channel_ls)):
            out_convs.append(OutLayer(in_channels=mid_channel_ls[i], mid_channels=mid_channel_ls[0],
                                      n_classes=n_classes, out_size=out_size))
        self._out_convs = nn.ModuleList(out_convs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_list = self._multi_scale_backbone(x, start_idx=1, end_idx=5)  # BackBoneの後ろからレイヤー数分取得する.
        reduced_x_list = []

        for i, reducer_layer in enumerate(self._reducers):
            reduced_x_list.append(reducer_layer(x_list[i]))

        x_list = self._segmentation_shelf(reduced_x_list)

        outs = []
        for i, out_conv_layer in enumerate(self._out_convs):
            outs.append(out_conv_layer(x_list[i]))
        return outs

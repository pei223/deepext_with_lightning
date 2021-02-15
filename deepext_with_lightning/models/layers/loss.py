from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from deepext_with_lightning.image_process.convert import try_cuda


class JaccardLoss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, teacher: torch.Tensor, smooth=1.0):
        teacher = teacher.float()
        intersection = (pred * teacher).sum((-1, -2))
        sum_ = (pred.abs() + pred.abs()).sum((-1, -2))
        jaccard = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jaccard).mean(1).mean(0)


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor, smooth=1.0):
        """
        :param pred:
        :param teacher:
        :param smooth:
        :return:
        """
        pred = F.normalize(pred - pred.min(), 1)
        pred, teacher = teacher.float(), pred.float()

        # batch size, classes, width and height
        intersection = (pred * teacher).sum((-1, -2))
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        teacher = teacher.contiguous().view(teacher.shape[0], teacher.shape[1], -1)
        pred_sum = pred.sum((-1,))
        teacher_sum = teacher.sum((-1,))
        dice_by_classes = (2. * intersection + smooth) / (pred_sum + teacher_sum + smooth)
        return (1. - dice_by_classes).mean((-1,)).mean((-1,))


class DiceLossWithAreaPenalty(torch.nn.Module):
    def init(self):
        super(DiceLossWithAreaPenalty, self).init()

    def forward(self, pred, teacher, smooth=1.0):
        """
        :param pred: 推論結果 (Batch size * class num * height * width)
        :param teacher: 教師データ (Batch size * class num * height * weigh )
        :param smooth:
        :return:
        """
        pred = F.normalize(pred - pred.min(), 1)
        # print("pred", pred.sum(1, ))

        intersection = (pred * teacher.float()).sum((-1, -2))
        # batch size, classes, width and height
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        teacher = teacher.contiguous().view(teacher.shape[0], teacher.shape[1], -1)

        total_area = float(teacher.shape[-1])
        area_by_classes = teacher.long().sum(-1, )
        # batch size, area by classes
        weight_by_classes = area_by_classes.float() / total_area
        # print("weight by class", weight_by_classes)

        # Batch * Classes
        pred_sum_by_classes = pred.sum((-1,))
        teacher_sum_by_classes = teacher.long().sum((-1,))
        dice_by_classes = (2. * intersection + smooth) / (pred_sum_by_classes + teacher_sum_by_classes.float() + smooth)
        dice_loss_by_classes = 1. - dice_by_classes
        dice_loss_by_classes[dice_loss_by_classes != dice_loss_by_classes] = 1.  # nanを省く
        # print("dice loss", dice_loss_by_classes)

        weighted_dice_loss = (dice_loss_by_classes * weight_by_classes).sum(-1, )  # Batch,
        # print("weighted_dice", weighted_dice_loss)

        loss = weighted_dice_loss.mean(0, )
        # print(loss)
        return loss if not torch.isnan(loss) else 1.0


class AdaptiveCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor):
        """
        :param pred: (B * C * H * W) or (B * C)
        :param teacher: (B * C * H * W) or (B * H * W) or (B * C) or (B)
        """
        if pred.ndim == 4 and pred.ndim == teacher.ndim:  # For One-Hot
            # TODO Label smoothingやりたいが今のところ無理っぽい
            return F.cross_entropy(pred, teacher.argmax(1), reduction="mean")
            # return nn.BCEWithLogitsLoss()(pred, teacher)
            # return F.binary_cross_entropy(nn.LogSoftmax(dim=1)(pred), teacher, reduction="mean")
        if pred.ndim == 4 and teacher.ndim == 3:
            return F.cross_entropy(pred, teacher, reduction="mean")
        if pred.ndim == 2 and teacher.ndim == 1:
            return F.cross_entropy(pred, teacher, reduction="mean")
        assert False, f"Invalid pred or teacher type,  {pred.shape} and {teacher.shape}"


class SegmentationFocalLoss(nn.Module):
    def __init__(self, gamma=2, weights: List[float] = None, logits=True):
        """
        :param gamma: 簡単なサンプルの重み. 大きいほど簡単なサンプルを重視しない.
        :param weights: weights by classes,
        :param logits:
        """
        super().__init__()
        self.gamma = gamma
        self.class_weight_tensor = try_cuda(torch.tensor(weights).view(-1, 1, 1)) if weights else None
        self.logits = logits
        if not logits and weights is not None:
            RuntimeWarning("重みを適用するにはlogitsをTrueにしてください.")

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        :param pred: batch_size, n_classes, height, width
        :param teacher: batch_size, n_classes, height, width
        :return:
        """
        if self.logits:
            ce_loss = F.binary_cross_entropy_with_logits(pred, teacher, reduce=False)
            pt = torch.exp(-ce_loss)
            if self.class_weight_tensor is not None:
                class_weight_tensor = self.class_weight_tensor.expand(pred.shape[0],
                                                                      self.class_weight_tensor.shape[0],
                                                                      self.class_weight_tensor.shape[1],
                                                                      self.class_weight_tensor.shape[2])
                focal_loss = (1. - pt) ** self.gamma * (ce_loss * class_weight_tensor)
            else:
                focal_loss = (1. - pt) ** self.gamma * ce_loss
        else:
            ce_loss = F.cross_entropy(pred, teacher.argmax(1), reduce=False)
            pt = torch.exp(-ce_loss)
            focal_loss = (1. - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)


class ClassificationFocalLoss(nn.Module):
    def __init__(self, n_classes: int, gamma=2, weights: List[float] = None, logits=True):
        """
        :param n_classes:
        :param gamma: 簡単なサンプルの重み. 大きいほど簡単なサンプルを重視しない.
        :param weights: weights by classes,
        :param logits:
        """
        super().__init__()
        self._n_classes = n_classes
        self.gamma = gamma
        self.class_weight_tensor = try_cuda(torch.tensor(weights).view(-1, )) if weights else None
        self.logits = logits
        if not logits and weights is not None:
            RuntimeWarning("重みを適用するにはlogitsをTrueにしてください.")

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        :param pred: batch_size, n_classes
        :param teacher: batch_size,
        :return:
        """
        if self.logits:
            if teacher.ndim == 1:  # 1次元ならonehotの2次元tensorにする
                teacher = torch.eye(self._n_classes)[teacher]
            ce_loss = F.binary_cross_entropy_with_logits(pred, teacher, reduce=False)
            pt = torch.exp(-ce_loss)

            if self.class_weight_tensor is not None:
                class_weight_tensor = self.class_weight_tensor.expand(pred.shape[0],
                                                                      self.class_weight_tensor.shape[0], )
                focal_loss = (1. - pt) ** self.gamma * (ce_loss * class_weight_tensor)
            else:
                focal_loss = (1. - pt) ** self.gamma * ce_loss
        else:
            if teacher.ndim == 2:  # onehotの2次元tensorなら1次元にする
                teacher = teacher.argmax(1)
            ce_loss = F.cross_entropy(pred, teacher, reduce=False)
            pt = torch.exp(-ce_loss)
            focal_loss = (1. - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)


class ClassificationFocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_classes: int, alpha=0.3, gamma=2, weights: List[float] = None):
        """
        :param alpha: parameter of Label Smoothing.
        :param n_classes:
        :param gamma: 簡単なサンプルの重み. 大きいほど簡単なサンプルを重視しない.
        :param weights: weights by classes,
        :param logits:
        """
        super().__init__()
        self._alpha = alpha
        self._noise_val = alpha / n_classes
        self._n_classes = n_classes
        self.gamma = gamma
        self.class_weight_tensor = try_cuda(torch.tensor(weights).view(-1, )) if weights else None

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        :param pred: batch_size, n_classes
        :param teacher: batch_size,
        :return:
        """
        if teacher.ndim == 1:  # 1次元ならonehotの2次元tensorにする
            teacher = torch.eye(self._n_classes)[teacher]
        # Label smoothing.
        teacher = teacher * (1 - self._alpha) + self._noise_val

        ce_loss = F.binary_cross_entropy_with_logits(pred, teacher, reduce=False)
        pt = torch.exp(-ce_loss)

        if self.class_weight_tensor is not None:
            class_weight_tensor = self.class_weight_tensor.expand(pred.shape[0],
                                                                  self.class_weight_tensor.shape[0], )
            focal_loss = (1. - pt) ** self.gamma * (ce_loss * class_weight_tensor)
        else:
            focal_loss = (1. - pt) ** self.gamma * ce_loss

        return torch.mean(focal_loss)


class SegmentationFocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_classes: int, gamma=2, alpha=0.3, weights: List[float] = None, logits=True):
        """
        :param alpha: parameter of Label Smoothing.
        :param gamma: 簡単なサンプルの重み. 大きいほど簡単なサンプルを重視しない.
        :param weights: weights by classes,
        :param logits:
        """
        super().__init__()
        self.gamma = gamma
        self._alpha = alpha
        self._noise_val = alpha / n_classes
        self.class_weight_tensor = try_cuda(torch.tensor(weights).view(-1, 1, 1)) if weights else None

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        :param pred: batch_size, n_classes, height, width
        :param teacher: batch_size, n_classes, height, width
        :return:
        """
        teacher = teacher * (1 - self._alpha) + self._noise_val

        ce_loss = F.binary_cross_entropy_with_logits(pred, teacher, reduce=False)
        pt = torch.exp(-ce_loss)
        if self.class_weight_tensor is not None:
            class_weight_tensor = self.class_weight_tensor.expand(pred.shape[0],
                                                                  self.class_weight_tensor.shape[0],
                                                                  self.class_weight_tensor.shape[1],
                                                                  self.class_weight_tensor.shape[2])
            focal_loss = (1. - pt) ** self.gamma * (ce_loss * class_weight_tensor)
        else:
            focal_loss = (1. - pt) ** self.gamma * ce_loss

        return torch.mean(focal_loss)

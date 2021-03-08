import torch
import pytorch_lightning as pl


class SegmentationIoU(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes: bool = False, without_background_class: bool = False):
        super().__init__(compute_on_step=False)
        self._n_classes = n_classes
        self._without_background_class = without_background_class
        self._by_classes = by_classes
        self.add_state("overlap_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")
        self.add_state("union_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds, targets = preds.reshape([preds.shape[0], -1]), targets.reshape([targets.shape[0], -1])
        for label_val in range(self._n_classes):
            targets_indices, preds_indices = (targets == label_val), (preds == label_val)
            overlap_indices = targets_indices & preds_indices
            overlap = torch.count_nonzero(overlap_indices)
            union = torch.count_nonzero(targets_indices) + torch.count_nonzero(preds_indices) - overlap
            self.overlap_by_classes[label_val] += overlap
            self.union_by_classes[label_val] += union

    def compute(self):
        """
        :return: mean IoU
        """
        iou_by_classes = self.overlap_by_classes / self.union_by_classes
        if self._without_background_class:
            iou_by_classes = iou_by_classes[1:]
        if self._by_classes:
            return iou_by_classes
        return torch.mean(iou_by_classes)

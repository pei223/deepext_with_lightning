import torch
import pytorch_lightning as pl


class ClassificationAccuracy(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes=False, average=False):
        if by_classes and average:
            raise ValueError("by_classes and average must be either")
        super().__init__()
        self._n_classes = n_classes
        self._by_classes = by_classes
        self._average = average
        self.add_state("confusion_matrix",
                       default=torch.zeros((n_classes, n_classes)))

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        for i in range(preds.shape[0]):
            self.confusion_matrix[preds[i], targets[i]] += 1

    def compute(self):
        tp = torch.diag(self.confusion_matrix)
        fn_tp = torch.sum(self.confusion_matrix, dim=0)
        if self._average:
            accuracy_by_classes = tp / fn_tp
            return torch.mean(accuracy_by_classes)
        if self._by_classes:
            return tp / fn_tp
        return torch.sum(tp) / torch.sum(fn_tp)


class ClassificationRecallPrecision(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes=False, average=False):
        if by_classes and average:
            raise ValueError("by_classes and average must be either")
        super().__init__()
        self._n_classes = n_classes
        self._by_classes = by_classes
        self._average = average
        self.add_state("confusion_matrix",
                       default=torch.zeros((n_classes, n_classes)))

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        for i in range(preds.shape[0]):
            self.confusion_matrix[preds[i], targets[i]] += 1

    def compute(self):
        tp = torch.diag(self.confusion_matrix)
        fp_tp = torch.sum(self.confusion_matrix, dim=1)
        fn_tp = torch.sum(self.confusion_matrix, dim=0)
        if self._average:
            recall = tp / fn_tp
            precision = tp / fp_tp
            return torch.mean(recall), torch.mean(precision)
        if self._by_classes:
            recall = tp / fn_tp
            precision = tp / fp_tp
            return recall, precision

        total_tp = torch.sum(tp)
        total_fp_tp = torch.sum(fp_tp)
        total_fn_tp = torch.sum(fn_tp)
        recall = total_tp / total_fn_tp
        precision = total_tp / total_fp_tp
        return recall, precision

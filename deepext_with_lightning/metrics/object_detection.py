from typing import List, Tuple, Union

import numpy as np
import torch

import pytorch_lightning as pl


def calc_area(bbox: np.ndarray):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def calc_bbox_overlap_union_iou(pred: np.ndarray or None, teacher: np.ndarray) -> Tuple[float, float, float]:
    """
    :param pred: ndarray (4, )
    :param teacher: ndarray (4, )
    :return: overlap, union, iou
    """
    teacher_area = (teacher[2] - teacher[0]) * (teacher[3] - teacher[1])
    if pred is None:
        return 0.0, teacher_area, 0.0

    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])

    intersection_width = np.maximum(np.minimum(pred[2], teacher[2]) - np.maximum(pred[0], teacher[0]), 0)
    intersection_height = np.maximum(np.minimum(pred[3], teacher[3]) - np.maximum(pred[1], teacher[1]), 0)

    overlap = intersection_width * intersection_height
    union = teacher_area + pred_area - overlap
    iou = overlap / union
    return overlap, union, iou


class DetectionIoU(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes: bool = False):
        super().__init__()
        self._n_classes = n_classes
        self._by_classes = by_classes
        self.add_state("image_count_by_classes", default=torch.tensor([0. for _ in range(n_classes)]),
                       dist_reduce_fx="sum")
        self.add_state("total_iou_by_classes", default=torch.tensor([0. for _ in range(n_classes)]),
                       dist_reduce_fx="sum")

    def update(self, preds: List[np.ndarray], targets: Union[np.ndarray, torch.Tensor]) -> None:
        """
        :param preds: Sorted by score. (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        :param targets: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        targets = targets.cpu().detach().numpy() if isinstance(targets, torch.Tensor) else targets
        # 全探索だと遅いのでクラスごとにまとめておく
        preds_by_class = []
        for pred_bboxes in preds:
            pred_by_class = [[] for _ in range(self._n_classes)]
            for pred_bbox in pred_bboxes:
                pred_by_class[int(pred_bbox[4])].append(pred_bbox)
            preds_by_class.append(pred_by_class)

        for i in range(targets.shape[0]):  # Explore every batch.
            bbox_annotations = targets[i, :, :]
            # Exclude invalid label annotation.
            bbox_annotations = bbox_annotations[bbox_annotations[:, 4] >= 0]

            pred_by_class = preds_by_class[i]

            """
            1画像でラベルごとに計算.
            ラベルごとの面積合計/overlapを計算
            1画像ごとにIoU算出、最終的に画像平均を算出
            """

            total_area_by_classes = [0 for _ in range(self._n_classes)]
            total_overlap_by_classes = [0 for _ in range(self._n_classes)]
            is_label_appeared = [False for _ in range(self._n_classes)]
            for bbox_annotation in bbox_annotations:

                label = int(bbox_annotation[4])
                total_area_by_classes[label] += calc_area(bbox_annotation)
                pred_bboxes = pred_by_class[label]

                if pred_bboxes is None or len(pred_bboxes) == 0:
                    continue

                # Calculate area and overlap by class.
                for pred_bbox in pred_bboxes:
                    overlap, _, _ = calc_bbox_overlap_union_iou(pred_bbox, bbox_annotation)
                    total_overlap_by_classes[label] += overlap
                    if is_label_appeared[label]:
                        continue
                    total_area_by_classes[label] += calc_area(pred_bbox)
                is_label_appeared[label] = True

            for label in range(self._n_classes):
                # Not exist label in this data.
                if total_area_by_classes[label] <= 0:
                    continue
                self.total_iou_by_classes[label] += total_overlap_by_classes[label] / (
                        total_area_by_classes[label] - total_overlap_by_classes[label])
                self.image_count_by_classes[label] += 1

    def compute(self):
        epsilon = 1e-8
        iou_by_classes = self.total_iou_by_classes / (self.image_count_by_classes + epsilon)
        if self._by_classes:
            return iou_by_classes
        return torch.mean(iou_by_classes)


class RecallPrecision(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes: bool = False):
        super().__init__()
        self._n_classes = n_classes
        self._by_classes = by_classes
        self.add_state("tp_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")
        self.add_state("fp_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")
        self.add_state("fn_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")

    def update(self, preds: List[np.ndarray], targets: Union[np.ndarray, torch.Tensor]) -> None:
        """
        :param preds: Sorted by score. (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        :param targets: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        targets = targets.cpu().detach().numpy() if isinstance(targets, torch.Tensor) else targets
        # 全探索だと遅いのでクラスごとにまとめておく
        preds_by_class = []
        for pred_bboxes in preds:
            pred_by_class = [[] for _ in range(self._n_classes)]
            for pred_bbox in pred_bboxes:
                pred_by_class[int(pred_bbox[4])].append(pred_bbox)
            preds_by_class.append(pred_by_class)

        for i in range(targets.shape[0]):
            bbox_annotations = targets[i, :, :]
            # Exclude invalid label annotation.
            bbox_annotations = bbox_annotations[bbox_annotations[:, 4] >= 0]

            pred_by_class = preds_by_class[i]

            applied_bbox_count_by_classes = [0 for _ in range(self._n_classes)]
            for bbox_annotation in bbox_annotations:
                label = int(bbox_annotation[4])
                pred_bboxes = pred_by_class[label]

                if pred_bboxes is None or len(pred_bboxes) == 0:
                    self.fn_by_classes[label] += 1
                    continue
                # Explore max iou of bbox_annotation
                is_matched = False
                for pred_bbox in pred_bboxes:
                    overlap, union, iou = calc_bbox_overlap_union_iou(pred_bbox, bbox_annotation)
                    if iou >= 0.5:
                        applied_bbox_count_by_classes[label] += 1
                        self.tp_by_classes[label] += 1
                        is_matched = True
                        break
                if not is_matched:
                    self.fn_by_classes[label] += 1

            for label in range(self._n_classes):
                self.fp_by_classes[label] += len(pred_by_class[label]) - applied_bbox_count_by_classes[label]

    def compute(self):
        epsilon = 1e-8
        recall = self.tp_by_classes / (self.tp_by_classes + self.fn_by_classes + epsilon)
        precision = self.tp_by_classes / (self.tp_by_classes + self.fp_by_classes + epsilon)
        f_score = 2. * recall * precision / (recall + precision + epsilon)
        if self._by_classes:
            return recall, precision, f_score
        return torch.mean(recall), torch.mean(precision), torch.mean(f_score)


class MeanAveragePrecision(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes=False, pr_rate=11):
        super().__init__()
        self._n_classes = n_classes
        self.add_state("average_precision", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("image_count", default=torch.tensor(0.), dist_reduce_fx="sum")
        self._pr_rate = pr_rate
        self._by_classes = by_classes

    def update(self, preds: List[np.ndarray], targets: Union[np.ndarray, torch.Tensor]) -> None:
        """
        :param preds: Sorted by score. (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        :param targets: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        targets = targets.cpu().detach().numpy() if isinstance(targets, torch.Tensor) else targets
        for i in range(len(preds)):
            pred_bboxes = preds[i]
            target_bboxes = targets[i]
            target_bboxes = target_bboxes[target_bboxes[:, 4] >= 0]

            ap = self._calc_average_precision(pred_bboxes, target_bboxes)
            self.average_precision += ap
            self.image_count += 1

    def compute(self):
        return self.average_precision / self.image_count

    def _calc_average_precision(self, pred_bboxes: np.ndarray, target_bboxes: np.ndarray):
        recall_ls, precision_ls = [], []
        tp, fp, fn = 0, 0, target_bboxes.shape[0]
        for i in range(pred_bboxes.shape[0]):
            correct = False
            for j in range(target_bboxes.shape[0]):
                overlap, union, iou = calc_bbox_overlap_union_iou(pred_bboxes[i], target_bboxes[j])
                if iou >= 0.5:
                    fn -= 1
                    tp += 1
                    correct = True
                    break
            if not correct:
                fp += 1
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            precision_ls.append(precision)
            recall_ls.append(recall)
        recalls = np.asarray(recall_ls)
        precisions = np.asarray(precision_ls)
        precisions = self._smooth_precisions(recalls, precisions)
        return self._calc_integral(recalls, precisions)

    def _smooth_precisions(self, recalls: np.ndarray, precisions: np.ndarray) -> np.ndarray:
        new_precision_ls = []
        for i in range(recalls.shape[0]):
            new_precision_ls.append(np.max(precisions[i - 1:]))
        return np.asarray(new_precision_ls)

    def _calc_integral(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        total_precision = 0.
        for recall_threshold in np.linspace(0, 1, self._pr_rate).tolist():
            precision = precisions[recalls >= recall_threshold]
            if precision.shape[0] == 0:
                continue
            total_precision += np.max(precision)
        return total_precision / self._pr_rate

import argparse
import torchvision
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext.models.base import DetectionModel
from deepext.data.transforms import AlbumentationsDetectionWrapperTransform
from deepext.models.object_detection import EfficientDetector
from deepext.trainer import Trainer, LearningCurveVisualizer, CosineDecayScheduler
from deepext.trainer.callbacks import ModelCheckout, GenerateDetectionImageCallback
from deepext.metrics.object_detection import *
from deepext.metrics import DetailMetricKey
from deepext.data.dataset import VOCAnnotationTransform, AdjustDetectionTensorCollator
from deepext.utils import *

from common import DETECTION_DATASET_INFO

VALID_MODEL_KEYS = ["efficientdet"]


# NOTE モデル・データセットはここを追加
def build_model(args, n_classes) -> DetectionModel:
    if args._model == "efficientdet":
        return EfficientDetector(num_classes=n_classes, lr=args._lr,
                                 network=f"efficientdet-d{args.efficientdet_scale}", score_threshold=0.5)
    raise RuntimeError(f"Invalid model name: {args._model}")


def build_transforms(args, class_index_dict: Dict[str, int]) -> Tuple[any, any]:
    train_transforms = AlbumentationsDetectionWrapperTransform([
        A.HorizontalFlip(),
        A.RandomResizedCrop(width=args.image_size, height=args.image_size, scale=(0.8, 1.), p=1.),
        A.OneOf([
            A.RandomGamma(),
            A.RandomBrightnessContrast(),
            A.Blur(blur_limit=5),
        ], p=0.5),
        ToTensorV2(),
    ], annotation_transform=VOCAnnotationTransform(class_index_dict))
    test_transforms = AlbumentationsDetectionWrapperTransform([
        A.Resize(width=args.image_size, height=args.image_size),
        ToTensorV2(),
    ], annotation_transform=VOCAnnotationTransform(class_index_dict))
    return train_transforms, test_transforms


def build_dataset(args, train_transforms, test_transforms) -> Tuple[Dataset, Dataset]:
    if args.dataset == "voc2012":
        train_dataset = torchvision.datasets.VOCDetection(root=args.dataset_root, download=True, year="2012",
                                                          transforms=train_transforms, image_set='trainval')
        test_dataset = torchvision.datasets.VOCDetection(root=args.dataset_root, download=True, year="2012",
                                                         transforms=test_transforms, image_set='trainval')
        return train_dataset, test_dataset
    elif args.dataset == "voc2007":
        train_dataset = torchvision.datasets.VOCDetection(root=args.dataset_root, download=True, year="2007",
                                                          transforms=train_transforms, image_set='trainval')
        test_dataset = torchvision.datasets.VOCDetection(root=args.dataset_root, download=True, year="2007",
                                                         transforms=test_transforms, image_set='test')
        return train_dataset, test_dataset
    raise RuntimeError(f"Invalid dataset name: {args.dataset_root}")


def build_data_loader(args, train_dataset: Dataset, test_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                      collate_fn=AdjustDetectionTensorCollator(), pin_memory=True, num_workers=4), \
           DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                      collate_fn=AdjustDetectionTensorCollator(), pin_memory=True, num_workers=4)


parser = argparse.ArgumentParser(description='Pytorch Image detection training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default="voc2012",
                    help=f'Dataset type in {list(DETECTION_DATASET_INFO.keys())}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default="efficientdet", help=f"Model type in {VALID_MODEL_KEYS}")
parser.add_argument('--load_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--save_weight_path', type=str, default=None, help="Saved weight path")
parser.add_argument('--efficientdet_scale', type=int, default=0, help="Number of scale of EfficientDet.")
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")

if __name__ == "__main__":
    args = parser.parse_args()

    dataset_info = DETECTION_DATASET_INFO.get(args.dataset)
    if dataset_info is None:
        raise ValueError(f"Invalid dataset name - {args.dataset}.  Required [{list(DETECTION_DATASET_INFO.keys())}]")

    label_names = dataset_info["label_names"]
    class_index_dict = {}
    for i, label_name in enumerate(label_names):
        class_index_dict[label_name] = i

    # Fetch dataset.
    train_transforms, test_transforms = build_transforms(args, class_index_dict)
    train_dataset, test_dataset = build_dataset(args, train_transforms, test_transforms)
    train_data_loader, test_data_loader = build_data_loader(args, train_dataset, test_dataset)

    # Fetch model and load weight.
    model = try_cuda(build_model(args, dataset_info["n_classes"]))
    if args.load_weight_path:
        model.load_weight(args.load_weight_path)

    # Training setting.
    # epoch_lr_scheduler = CosineDecayScheduler(max_lr=args.lr, max_epochs=args.epoch, warmup_epochs=0)
    # loss_lr_scheduler = None
    loss_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.get_optimizer(), patience=5, verbose=True)
    epoch_lr_scheduler = None

    callbacks = [ModelCheckout(per_epoch=int(100), model=model, our_dir="saved_weights")]
    if args.progress_dir:
        callbacks.append(GenerateDetectionImageCallback(model, args.image_size, test_dataset, per_epoch=5,
                                                        out_dir=args.progress_dir,
                                                        label_names=label_names))
    metric_ls = [DetectionIoUByClasses(label_names), RecallAndPrecision(label_names)]
    metric_for_graph = DetectionIoUByClasses(label_names, val_key=DetailMetricKey.KEY_AVERAGE)
    learning_curve_visualizer = LearningCurveVisualizer(metric_name="IoU average", ignore_epoch=10,
                                                        save_filepath="detection_learning_curve.png")
    # Training.
    Trainer(model, learning_curve_visualizer=learning_curve_visualizer).fit(train_data_loader=train_data_loader,
                                                                            test_data_loader=test_data_loader,
                                                                            epochs=args.epoch,
                                                                            metric_for_graph=metric_for_graph,
                                                                            callbacks=callbacks, metric_ls=metric_ls,
                                                                            loss_lr_scheduler=loss_lr_scheduler,
                                                                            epoch_lr_scheduler_func=epoch_lr_scheduler,
                                                                            calc_metrics_per_epoch=10)

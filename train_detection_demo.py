import argparse
import torchvision
from typing import Tuple, Dict
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from deepext_with_lightning.image_process.convert import try_cuda
from deepext_with_lightning.models.base import DetectionModel
from deepext_with_lightning.models.object_detection import EfficientDetector

from deepext_with_lightning.transforms import AlbumentationsDetectionWrapperTransform
from deepext_with_lightning.callbacks.object_detection import GenerateDetectionImageCallback
from deepext_with_lightning.dataset import VOCAnnotationTransform, AdjustDetectionTensorCollator
from deepext_with_lightning.dataset.functions import label_names_to_dict

from common import DETECTION_DATASET_INFO, get_logger, build_data_loader

VALID_MODEL_KEYS = ["efficientdet"]


def build_model(args, n_classes) -> DetectionModel:
    if args.model == "efficientdet":
        return EfficientDetector(n_classes=n_classes, lr=args.lr,
                                 network=f"efficientdet-d{args.efficientdet_scale}", score_threshold=0.5)
    raise RuntimeError(f"Invalid model name: {args.model}")


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
                                                         transforms=test_transforms, image_set='val')
        return train_dataset, test_dataset
    elif args.dataset == "voc2007":
        train_dataset = torchvision.datasets.VOCDetection(root=args.dataset_root, download=True, year="2007",
                                                          transforms=train_transforms, image_set='train')
        test_dataset = torchvision.datasets.VOCDetection(root=args.dataset_root, download=True, year="2007",
                                                         transforms=test_transforms, image_set='val')
        return train_dataset, test_dataset
    raise RuntimeError(f"Invalid dataset name: {args.dataset_root}")


parser = argparse.ArgumentParser(description='Pytorch Image detection training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default="voc2012",
                    help=f'Dataset type in {list(DETECTION_DATASET_INFO.keys())}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default="efficientdet", help=f"Model type in {VALID_MODEL_KEYS}")
parser.add_argument('--load_checkpoint_path', type=str, default=None, help="Saved checkpoint path")
parser.add_argument('--save_checkpoint_path', type=str, default="checkpoints", help="Saving checkpoint directory")
parser.add_argument('--efficientdet_scale', type=int, default=0, help="Number of scale of EfficientDet.")
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")
parser.add_argument('--val_every_n_epoch', type=int, default=5, help="Validate every n epoch.")
parser.add_argument('--log_type', type=str, default="mlflow", help="")

if __name__ == "__main__":
    args = parser.parse_args()

    dataset_info = DETECTION_DATASET_INFO.get(args.dataset)
    if dataset_info is None:
        raise ValueError(f"Invalid dataset name - {args.dataset}.  Required [{list(DETECTION_DATASET_INFO.keys())}]")

    label_names = dataset_info["label_names"]
    class_index_dict = label_names_to_dict(label_names)

    # Fetch dataset.
    train_transforms, test_transforms = build_transforms(args, class_index_dict)
    train_dataset, test_dataset = build_dataset(args, train_transforms, test_transforms)
    train_data_loader, test_data_loader = build_data_loader(args, train_dataset, test_dataset,
                                                            AdjustDetectionTensorCollator(),
                                                            AdjustDetectionTensorCollator())

    # Fetch model and load weight.
    model = try_cuda(build_model(args, dataset_info["n_classes"]))
    if args.load_checkpoint_path:
        model.load_from_checkpoint(args.load_checkpoint_path)

    # Training setting.
    logger = get_logger("detection_demo", args, model)
    callbacks = [ModelCheckpoint(period=args.val_every_n_epoch, filename=f"{model.generate_model_name()}",
                                 dirpath=args.save_checkpoint_path, monitor='val_map', verbose=True, mode="max")]
    if args.progress_dir:
        callbacks.append(GenerateDetectionImageCallback(model=model, out_dir=args.progress_dir, dataset=test_dataset,
                                                        per_epoch=2, label_names=label_names,
                                                        img_size=(args.image_size, args.image_size)))
    # Training.
    Trainer(max_epochs=args.epoch, callbacks=callbacks, gpus=-1,
            check_val_every_n_epoch=args.val_every_n_epoch, logger=logger) \
        .fit(model, train_dataloader=train_data_loader, val_dataloaders=test_data_loader)

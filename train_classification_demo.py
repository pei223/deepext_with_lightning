import argparse
from typing import Tuple

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
from torch.utils.data import Dataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from deepext_with_lightning.callbacks import GenerateAttentionMap, CSVClassificationResult
from deepext_with_lightning.models.layers.backbone_key import BackBoneKey
from deepext_with_lightning.models.base import ClassificationModel
from deepext_with_lightning.models.classification import *
from deepext_with_lightning.transforms import AlbumentationsOnlyImageWrapperTransform
from common import CLASSIFICATION_DATASET_INFO, build_data_loader, get_logger, label_names_to_dict

VALID_MODEL_KEYS = ["efficientnet", "mobilenet", "abn", "custommodel"]


# NOTE モデル・データセットはここを追加
def build_model(args, n_classes) -> ClassificationModel:
    if args.model == "efficientnet":
        return EfficientNet(num_classes=n_classes, lr=args.lr, network=f"efficientnet-b{args.efficientnet_scale}")
    if args.model == "mobilenet":
        return MobileNetV3(num_classes=n_classes, lr=args.lr, pretrained=False)
    if args.model == "abn":
        return AttentionBranchNetwork(n_classes=n_classes, lr=args.lr, backbone=BackBoneKey.from_val(args.submodel))
    if args.model == "customnet":
        return CustomClassificationNetwork(n_classes=n_classes, lr=args.lr,
                                           backbone=BackBoneKey.from_val(args.submodel))
    raise RuntimeError(f"Invalid model name: {args.model}")


def build_transforms(args) -> Tuple[any, any]:
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.RandomResizedCrop(width=args.image_size, height=args.image_size, scale=(0.7, 1.2)),
        A.Rotate((-30, 30), p=0.3),
        A.CoarseDropout(max_width=int(args.image_size / 8), max_height=int(args.image_size / 8), max_holes=3, p=0.3),
        ToTensorV2(),
    ])
    train_transforms = AlbumentationsOnlyImageWrapperTransform(train_transforms)

    test_transforms = A.Compose([
        A.Resize(width=args.image_size, height=args.image_size),
        ToTensorV2(),
    ])
    test_transforms = AlbumentationsOnlyImageWrapperTransform(test_transforms)
    return train_transforms, test_transforms


def build_dataset(args, train_transforms, test_transforms) -> Tuple[Dataset, Dataset]:
    if args.dataset == "stl10":
        train_dataset = torchvision.datasets.STL10(root=args.dataset_root, download=True, split="train",
                                                   transform=train_transforms)
        test_dataset = torchvision.datasets.STL10(root=args.dataset_root, download=True, split="test",
                                                  transform=test_transforms)
        return train_dataset, test_dataset
    if args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, download=True, train=True,
                                                     transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, download=True, train=False,
                                                    transform=test_transforms)
        return train_dataset, test_dataset
    raise RuntimeError(f"Invalid dataset name: {args.dataset_root}")


parser = argparse.ArgumentParser(description='Pytorch Image classification training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default="stl10",
                    help=f'Dataset type in {list(CLASSIFICATION_DATASET_INFO.keys())}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default="mobilenet", help=f"Model type in {VALID_MODEL_KEYS}")
parser.add_argument('--load_checkpoint_path', type=str, default=None, help="Saved checkpoint path")
parser.add_argument('--save_checkpoint_path', type=str, default="checkpoints", help="Saving checkpoint directory")
parser.add_argument('--efficientnet_scale', type=int, default=0, help="Number of scale of EfficientNet.")
parser.add_argument('--image_size', type=int, default=96, help="Image size.")
parser.add_argument('--submodel', type=str, default=None, help=f'Type of submodel(resnet18, resnet34...).')
parser.add_argument('--val_every_n_epoch', type=int, default=5, help="Validate every n epoch.")
parser.add_argument('--log_type', type=str, default="mlflow", help="")

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch dataset.
    dataset_info = CLASSIFICATION_DATASET_INFO.get(args.dataset)
    if dataset_info is None:
        raise ValueError(
            f"Invalid dataset name - {args.dataset}.  Required [{list(CLASSIFICATION_DATASET_INFO.keys())}]")

    label_names = dataset_info["label_names"]
    class_index_dict = label_names_to_dict(label_names)

    # Fetch dataset.
    train_transforms, test_transforms = build_transforms(args)
    train_dataset, test_dataset = build_dataset(args, train_transforms, test_transforms)
    train_data_loader, test_data_loader = build_data_loader(args, train_dataset, test_dataset)

    # Fetch model and load weight.
    model = build_model(args, dataset_info["n_classes"])
    if args.load_checkpoint_path:
        model.load_from_checkpoint(args.load_checkpoint_path)

    # Training setting.
    logger = get_logger("classification_demo", args, model)
    callbacks = [ModelCheckpoint(period=args.val_every_n_epoch, filename=f"{model.generate_model_name()}",
                                 dirpath=args.save_checkpoint_path, monitor='val_acc', verbose=True, mode="max"),
                 CSVClassificationResult(period=args.epoch, model=model, dataset=test_dataset,
                                         label_names=label_names, out_filepath=f"{args.progress_dir}/result.csv"), ]
    if args.progress_dir:
        if isinstance(model, AttentionBranchNetwork):
            callbacks.append(GenerateAttentionMap(model=model, output_dir=args.progress_dir, period=5,
                                                  dataset=test_dataset, label_names=label_names))
    # Training.
    Trainer(max_epochs=args.epoch, callbacks=callbacks, gpus=-1,
            check_val_every_n_epoch=args.val_every_n_epoch, logger=logger) \
        .fit(model, train_dataloader=train_data_loader, val_dataloaders=test_data_loader)

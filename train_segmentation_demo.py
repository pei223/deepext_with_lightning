import argparse
from typing import Tuple

import torchvision
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from deepext_with_lightning.image_process.convert import try_cuda
from deepext_with_lightning.models.base import SegmentationModel
from deepext_with_lightning.models.layers.backbone_key import BackBoneKey
from deepext_with_lightning.models.segmentation import *
from deepext_with_lightning.transforms import AlbumentationsSegmentationWrapperTransform
from deepext_with_lightning.callbacks import GenerateSegmentationImageCallback

from deepext_with_lightning.models.layers.loss import SegmentationFocalLoss
from deepext_with_lightning.dataset.functions import label_names_to_dict

from common import SEGMENTATION_DATASET_INFO, build_data_loader, get_logger

loss_func = SegmentationFocalLoss()
loss_func = None
VALID_MODEL_KEYS = ["unet", "resunet", "shelfnet"]


def build_model(args, n_classes: int) -> SegmentationModel:
    if args.model == "unet":
        return UNet(n_input_channels=3, n_classes=n_classes, lr=args.lr, loss_func=loss_func)
    if args.model == "resnet":
        return ResUNet(n_input_channels=3, n_classes=n_classes, lr=args.lr, loss_func=loss_func)
    if args.model == "shelfnet":
        return ShelfNet(n_classes=n_classes, lr=args.lr, out_size=(args.image_size, args.image_size),
                        loss_func=loss_func, backbone=BackBoneKey.from_val(args.submodel), backbone_pretrained=True)
    raise RuntimeError(f"Invalid model name: {args.model}")


def build_transforms(args, n_classes):
    train_transforms = A.Compose([
        A.HorizontalFlip(),
        A.RandomResizedCrop(width=args.image_size, height=args.image_size, scale=(0.5, 2.)),
        A.CoarseDropout(max_height=int(args.image_size / 5), max_width=int(args.image_size / 5), max_holes=5),
        A.Rotate(limit=(-30, 30)),
        A.ColorJitter(),
        A.OneOf([
            A.RandomGamma(),
            A.RandomBrightnessContrast(),
            A.Blur(blur_limit=5),
        ]),
        ToTensorV2(),
    ])
    train_transforms = AlbumentationsSegmentationWrapperTransform(train_transforms, class_num=n_classes,
                                                                  ignore_indices=[255, ])
    test_transforms = A.Compose([
        A.Resize(width=args.image_size, height=args.image_size),
        ToTensorV2(),
    ])
    test_transforms = AlbumentationsSegmentationWrapperTransform(test_transforms, class_num=n_classes,
                                                                 ignore_indices=[255, ])
    return train_transforms, test_transforms


def build_dataset(args, train_transforms, test_transforms) -> Tuple[Dataset, Dataset]:
    if args.dataset == "voc2012":
        train_dataset = torchvision.datasets.VOCSegmentation(root=args.dataset_root, download=True, year="2012",
                                                             image_set='trainval', transforms=train_transforms)
        test_dataset = torchvision.datasets.VOCSegmentation(root=args.dataset_root, download=True, year="2012",
                                                            image_set='val', transforms=test_transforms)
        return train_dataset, test_dataset
    if args.dataset == "voc2007":
        train_dataset = torchvision.datasets.VOCSegmentation(root=args.dataset_root, download=True, year="2007",
                                                             image_set='trainval', transforms=train_transforms)
        test_dataset = torchvision.datasets.VOCSegmentation(root=args.dataset_root, download=True, year="2007",
                                                            image_set='val', transforms=test_transforms)
        return train_dataset, test_dataset
    if args.dataset == "cityscape":
        train_dataset = torchvision.datasets.Cityscapes(root=args.dataset_root, split="train", target_type='semantic',
                                                        transforms=train_transforms)
        test_dataset = torchvision.datasets.Cityscapes(root=args.dataset_root, split="test", target_type='semantic',
                                                       transforms=test_transforms)
        return train_dataset, test_dataset

    raise RuntimeError(f"Invalid dataset name: {args.dataset_root}")


parser = argparse.ArgumentParser(description='Pytorch Image segmentation training.')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default="voc2012",
                    help=f'Dataset type in {list(SEGMENTATION_DATASET_INFO.keys())}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--dataset_root', type=str, required=True, help='Dataset folder path')
parser.add_argument('--progress_dir', type=str, default=None, help='Directory for saving progress')
parser.add_argument('--model', type=str, default="shelfnet", help=f"Model type in {VALID_MODEL_KEYS}")
parser.add_argument('--load_checkpoint_path', type=str, default=None, help="Saved checkpoint path")
parser.add_argument('--save_checkpoint_path', type=str, default="checkpoints", help="Saving checkpoint directory")
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")
parser.add_argument('--submodel', type=str, default=None, help=f'Type of sub model(resnet, resnet18, resnet34...).')
parser.add_argument('--val_every_n_epoch', type=int, default=5, help="Validate every n epoch.")
parser.add_argument('--log_type', type=str, default="mlflow", help="")

if __name__ == "__main__":
    args = parser.parse_args()

    dataset_info = SEGMENTATION_DATASET_INFO.get(args.dataset)
    if dataset_info is None:
        raise ValueError(f"Invalid dataset name - {args.dataset}.  Required [{list(SEGMENTATION_DATASET_INFO.keys())}]")

    label_names = dataset_info["label_names"]
    class_index_dict = label_names_to_dict(label_names)

    # Fetch dataset.
    train_transforms, test_transforms = build_transforms(args, dataset_info["n_classes"] + 1)
    train_dataset, test_dataset = build_dataset(args, train_transforms, test_transforms)
    train_data_loader, test_data_loader = build_data_loader(args, train_dataset, test_dataset)

    # Fetch model and load weight.
    model = try_cuda(build_model(args, dataset_info["n_classes"] + 1))  # include background class
    if args.load_checkpoint_path:
        model = model.load_from_checkpoint(args.load_checkpoint_path)

    # Training setting.
    logger = get_logger("segmentation_demo", args, model)
    callbacks = [ModelCheckpoint(period=args.val_every_n_epoch, filename=f"{model.generate_model_name()}",
                                 dirpath=args.save_checkpoint_path, monitor='val_iou', verbose=True, mode="max")]
    if args.progress_dir:
        callbacks.append(GenerateSegmentationImageCallback(output_dir=args.progress_dir, per_epoch=2, model=model,
                                                           dataset=test_dataset))
    # Training.
    Trainer(max_epochs=args.epoch, callbacks=callbacks, gpus=-1,
            check_val_every_n_epoch=args.val_every_n_epoch, logger=logger) \
        .fit(model, train_dataloader=train_data_loader, val_dataloaders=test_data_loader)

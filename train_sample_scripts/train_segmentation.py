import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dotenv import load_dotenv

from deepext_with_lightning.callbacks import GenerateSegmentationImageCallback
from deepext_with_lightning.dataset import IndexImageDataset, DatasetSplitter
from deepext_with_lightning.models.layers.backbone_key import BackBoneKey
from deepext_with_lightning.models.base import SegmentationModel
from deepext_with_lightning.models.segmentation import UNet, ResUNet, ShelfNet
from deepext_with_lightning.transforms import AlbumentationsSegmentationWrapperTransform
from deepext_with_lightning.dataset.functions import create_label_list_and_dict
from deepext_with_lightning.image_process.convert import try_cuda

load_dotenv("envs/segmentation.env")

# File/Directory path
train_images_dir = os.environ.get("TRAIN_IMAGES_PATH")
train_annotations_dir = os.environ.get("TRAIN_ANNOTATIONS_PATH")
test_images_dir = os.environ.get("TEST_IMAGES_PATH")
test_annotations_dir = os.environ.get("TEST_ANNOTATIONS_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")

load_checkpoint_path = os.environ.get("LOAD_CHECKPOINT_PATH")
save_checkpoint_dir_path = os.environ.get("SAVE_CHECKPOINT_DIR_PATH")
progress_dir = os.environ.get("PROGRESS_DIR_PATH")
# Model params
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
# Learning params
batch_size = int(os.environ.get("BATCH_SIZE"))
lr = float(os.environ.get("LR"))
epoch = int(os.environ.get("EPOCH"))

label_names, label_dict = create_label_list_and_dict(label_file_path)
n_classes = len(label_names) + 1  # Including background

ignore_indices = [255, ]

# TODO Data augmentation
train_transforms = AlbumentationsSegmentationWrapperTransform(A.Compose([
    A.HorizontalFlip(),
    A.RandomResizedCrop(width=width, height=height, scale=(0.5, 2.0)),
    A.CoarseDropout(max_height=int(height / 5), max_width=int(width / 5)),
    A.RandomBrightnessContrast(),
    ToTensorV2(),
]), class_num=n_classes, ignore_indices=ignore_indices)

test_transforms = AlbumentationsSegmentationWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]), class_num=n_classes, ignore_indices=ignore_indices)

# dataset/dataloader
if test_images_dir == "":
    test_ratio = float(os.environ.get("TEST_RATIO"))
    root_dataset = IndexImageDataset.create(train_images_dir, train_annotations_dir,
                                            transforms=None)
    train_dataset, test_dataset = DatasetSplitter().split_train_test(test_ratio,
                                                                     root_dataset,
                                                                     train_transforms=train_transforms,
                                                                     test_transforms=test_transforms)
else:
    train_dataset = IndexImageDataset.create(image_dir_path=train_images_dir,
                                             index_image_dir_path=train_annotations_dir,
                                             transforms=train_transforms)
    test_dataset = IndexImageDataset.create(image_dir_path=test_images_dir, index_image_dir_path=test_annotations_dir,
                                            transforms=test_transforms)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# TODO Model detail params
model: SegmentationModel = try_cuda(
    ShelfNet(n_classes=n_classes, out_size=(height, width)))

if load_checkpoint_path and load_checkpoint_path != "":
    model = model.load_from_checkpoint(load_checkpoint_path)

# TODO Train detail params
# Metrics/Callbacks
val_every_n_epoch = 5
callbacks = [ModelCheckpoint(period=val_every_n_epoch, filename=f"{model.generate_model_name()}",
                             dirpath=save_checkpoint_dir_path, monitor='val_iou', verbose=True, mode="max"),
             GenerateSegmentationImageCallback(output_dir=progress_dir, per_epoch=5, model=model,
                                               dataset=test_dataset)]
logger = MLFlowLogger(experiment_name=f"segmentation_{model.generate_model_name()}")

# Training.
Trainer(max_epochs=epoch, callbacks=callbacks, gpus=-1,
        check_val_every_n_epoch=val_every_n_epoch, logger=logger) \
    .fit(model, train_dataloader=train_data_loader, val_dataloaders=test_data_loader)

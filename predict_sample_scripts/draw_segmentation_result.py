from pathlib import Path
import cv2
import tqdm
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dotenv import load_dotenv

from deepext.data.transforms import AlbumentationsClsWrapperTransform
from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import SegmentationModel
from deepext.models.segmentation import ShelfNet
from deepext.utils import try_cuda
from deepext.data.dataset import ImageOnlyDataset

load_dotenv("envs/segmentation.env")

images_dir_path = os.environ.get("IMAGES_DIR_PATH")
result_dir_path = os.environ.get("RESULT_DIR_PATH")
weight_path = os.environ.get("MODEL_WEIGHT_PATH")
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
n_classes = int(os.environ.get("N_CLASSES"))

if not Path(result_dir_path).exists():
    Path(result_dir_path).mkdir()

transforms = AlbumentationsClsWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]))

dataset = ImageOnlyDataset(image_dir=images_dir_path, image_transform=transforms)

# TODO Choose model, parameters.
print("Loading model...")
model: SegmentationModel = try_cuda(
    ShelfNet(n_classes=n_classes, out_size=(height, width), backbone=BackBoneKey.RESNET_18))
model.load_weight(weight_path)
print("Model loaded")

for i, image in enumerate(tqdm.tqdm(dataset)):
    result_idx_array, result_image = model.calc_segmentation_image(image)
    result_image = cv2.resize(result_image, dataset.current_image_size())
    cv2.imwrite(f"{result_dir_path}/result_{i}.jpg", result_image)

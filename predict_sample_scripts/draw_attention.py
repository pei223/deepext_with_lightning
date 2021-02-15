from pathlib import Path
import cv2
from dotenv import load_dotenv
import os, tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import AttentionClassificationModel
from deepext.models.classification import AttentionBranchNetwork
from deepext.data.dataset import ImageOnlyDataset
from deepext.data.transforms import AlbumentationsClsWrapperTransform
from deepext.utils import try_cuda
from deepext.utils.dataset_util import create_label_list_and_dict

load_dotenv("envs/attention_classification.env")

images_dir_path = os.environ.get("IMAGES_DIR_PATH")
result_dir_path = os.environ.get("RESULT_DIR_PATH")
weight_path = os.environ.get("MODEL_WEIGHT_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
n_classes = int(os.environ.get("N_CLASSES"))

if not Path(result_dir_path).exists():
    Path(result_dir_path).mkdir()

label_names, label_dict = create_label_list_and_dict(label_file_path)

transforms = AlbumentationsClsWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]))

dataset = ImageOnlyDataset(image_dir=images_dir_path, image_transform=transforms)

# TODO Choose model, parameters.
print("Loading model...")
model: AttentionClassificationModel = try_cuda(
    AttentionBranchNetwork(n_classes=n_classes, backbone=BackBoneKey.RESNET_18))
model.load_weight(weight_path)
print("Model loaded")

for i, image in enumerate(tqdm.tqdm(dataset)):
    label, result_image = model.predict_label_and_heatmap(image)
    result_image = cv2.resize(result_image, dataset.current_image_size())
    cv2.imwrite(f"{result_dir_path}/{label_names[label]}_{i}.jpg", result_image)

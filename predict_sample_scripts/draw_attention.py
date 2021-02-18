from pathlib import Path
import cv2
from dotenv import load_dotenv
import os, tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext_with_lightning.models.layers.backbone_key import BackBoneKey
from deepext_with_lightning.models.base import AttentionClassificationModel
from deepext_with_lightning.models.classification import AttentionBranchNetwork
from deepext_with_lightning.dataset import ImageOnlyDataset
from deepext_with_lightning.transforms import AlbumentationsClsWrapperTransform
from deepext_with_lightning.image_process.convert import try_cuda, normalize255, tensor_to_cv, to_4dim
from deepext_with_lightning.dataset.functions import create_label_list_and_dict

load_dotenv("envs/attention_classification.env")

images_dir_path = os.environ.get("IMAGES_DIR_PATH")
result_dir_path = os.environ.get("RESULT_DIR_PATH")
checkpoint_path = os.environ.get("CHECKPOINT_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))

if not Path(result_dir_path).exists():
    Path(result_dir_path).mkdir()

label_names, label_dict = create_label_list_and_dict(label_file_path)
n_classes = len(label_names)

transforms = AlbumentationsClsWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]))

dataset = ImageOnlyDataset(image_dir=images_dir_path, image_transform=transforms)

print("Loading model...")
model: AttentionClassificationModel = try_cuda(
    AttentionBranchNetwork(n_classes=n_classes, backbone=BackBoneKey.RESNET_18).load_from_checkpoint(checkpoint_path))
print("Model loaded")

for i, img_tensor in enumerate(tqdm.tqdm(dataset)):
    origin_image = normalize255(tensor_to_cv(img_tensor))
    label, prob, attention_map = model.predict_label_and_heatmap(to_4dim(img_tensor))
    attention_map = normalize255(tensor_to_cv(attention_map[0]))
    blend_img = model.generate_heatmap_image(origin_image, attention_map)
    result_image = cv2.resize(blend_img, dataset.current_image_size())
    cv2.imwrite(f"{result_dir_path}/{label_names[label[0]]}_{i}.jpg", result_image)

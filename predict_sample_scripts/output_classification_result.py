import tqdm
from dotenv import load_dotenv
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext_with_lightning.models.base import ClassificationModel
from deepext_with_lightning.models import model_service
from deepext_with_lightning.dataset import ImageOnlyDataset
from deepext_with_lightning.image_process.convert import try_cuda, to_4dim
from deepext_with_lightning.transforms import AlbumentationsClsWrapperTransform
from deepext_with_lightning.dataset.functions import create_label_list_and_dict

load_dotenv("envs/classification.env")

model_name = os.environ.get("MODEL_NAME")
images_dir_path = os.environ.get("IMAGES_DIR_PATH")
result_file_path = os.environ.get("RESULT_FILE_PATH")
checkpoint_path = os.environ.get("CHECKPOINT_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))

label_names, label_dict = create_label_list_and_dict(label_file_path)
n_classes = len(label_names)

transforms = AlbumentationsClsWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]))

dataset = ImageOnlyDataset(image_dir=images_dir_path, image_transform=transforms)

# TODO Choose model, parameters.
print("Loading model...")
model_class = model_service.resolve_classification_model(model_name)
model: ClassificationModel = try_cuda(model_class.load_from_checkpoint(checkpoint_path))
print("Model loaded")

with open(result_file_path, "w") as file:
    file.write(f"filepath,result label\n")
    for i, img_tensor in enumerate(tqdm.tqdm(dataset)):
        label, prob = model.predict_label(to_4dim(img_tensor))
        file.write(f"{dataset.current_file_path()},{label_names[label[0]]}\n")

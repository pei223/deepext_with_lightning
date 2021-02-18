import os
from dotenv import load_dotenv

from deepext_with_lightning.dataset.functions import create_label_list_and_dict
from deepext_with_lightning.image_process.convert import try_cuda
from deepext_with_lightning.models import model_service
from deepext_with_lightning.models.base import ClassificationModel, AttentionClassificationModel, SegmentationModel, \
    DetectionModel, BaseDeepextModel
from deepext_with_lightning.camera import RealtimeClassification, RealtimeAttentionClassification, RealtimeSegmentation, \
    RealtimeDetection

load_dotenv("envs/camera_prediction.env")

model_name = os.environ.get("MODEL_NAME")
checkpoint_path = os.environ.get("CHECKPOINT_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))

label_names, label_dict = create_label_list_and_dict(label_file_path)
n_classes = len(label_names)

print("Loading model...")
model_class = model_service.resolve_model_class_from_name(model_name)
model: BaseDeepextModel = try_cuda(model_class.load_from_checkpoint(checkpoint_path))
print("Model loaded")

if isinstance(model, SegmentationModel):
    RealtimeSegmentation(model=model, img_size_for_model=(width, height)).realtime_predict(
        video_output_path="output.mp4")
elif isinstance(model, DetectionModel):
    RealtimeDetection(model=model, img_size_for_model=(width, height),
                      label_names=label_names).realtime_predict(video_output_path="output.mp4")
elif isinstance(model, AttentionClassificationModel):
    RealtimeAttentionClassification(model=model, img_size_for_model=(width, height),
                                    label_names=label_names).realtime_predict(video_output_path="output.mp4")
elif isinstance(model, ClassificationModel):
    RealtimeClassification(model=model, img_size_for_model=(width, height),
                           label_names=label_names).realtime_predict(video_output_path="output.mp4")

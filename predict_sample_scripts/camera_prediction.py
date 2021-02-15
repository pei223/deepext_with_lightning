import os
from dotenv import load_dotenv

from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import SegmentationModel, DetectionModel, ClassificationModel, AttentionClassificationModel
from deepext.models.segmentation import UNet, ResUNet, ShelfNet
from deepext.models.object_detection import EfficientDetector
from deepext.models.classification import EfficientNet, AttentionBranchNetwork, AttentionBranchNetwork, \
    MobileNetV3
from deepext.camera import RealtimeDetection, RealtimeSegmentation, RealtimeAttentionClassification, \
    RealtimeClassification
from deepext.utils import try_cuda
from deepext.utils.dataset_util import create_label_list_and_dict

load_dotenv("envs/camera_prediction.env")

weight_path = os.environ.get("MODEL_WEIGHT_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
n_classes = int(os.environ.get("N_CLASSES"))

label_names, label_dict = create_label_list_and_dict(label_file_path)

# TODO Choose model and load weight.
print("Loading model...")
# model = try_cuda(EfficientDetector(num_classes=n_classes, network='efficientdet-d0'))
# model = try_cuda(CustomShelfNet(n_classes=n_classes, backbone=BackBoneKey.RESNET_18, out_size=image_size))
# model = try_cuda(AttentionBranchNetwork(n_classes=n_classes, backbone=BackBoneKey.RESNET_18))
model = try_cuda(MobileNetV3(num_classes=n_classes, mode="small", pretrained=True))
model.load_weight(weight_path)
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

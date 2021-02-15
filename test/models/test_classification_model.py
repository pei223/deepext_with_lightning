import warnings

warnings.simplefilter('ignore')

from deepext_with_lightning.image_process.convert import try_cuda
from deepext_with_lightning.models.base import ClassificationModel, AttentionClassificationModel
from deepext_with_lightning.models.layers.backbone_key import BackBoneKey
from deepext_with_lightning.models.classification import CustomClassificationNetwork, AttentionBranchNetwork, \
    MobileNetV3, EfficientNet
from test.utils import gen_random_tensor, assert_tensor_shape

n_classes = 11
image_size = (96, 96)


def test_customnet():
    model = CustomClassificationNetwork(n_classes, backbone=BackBoneKey.RESNET_18, pretrained=False)
    _test_classification(model, 3, n_classes, image_size)
    _test_classification(model, 1, n_classes, image_size)


def test_abn():
    model = AttentionBranchNetwork(n_classes, backbone=BackBoneKey.RESNET_18, pretrained=False)
    _test_classification(model, 3, n_classes, image_size)
    _test_classification(model, 1, n_classes, image_size)


def test_abn_as_attention_model():
    model = AttentionBranchNetwork(n_classes, backbone=BackBoneKey.RESNET_18, pretrained=False)
    _test_classification(model, 3, n_classes, image_size)
    _test_classification(model, 1, n_classes, image_size)


def test_mobilenet():
    model = MobileNetV3(n_classes, pretrained=False)
    _test_classification(model, 3, n_classes, image_size)
    _test_classification(model, 1, n_classes, image_size)


def test_efficientnet():
    model = EfficientNet(n_classes)
    _test_classification(model, 3, n_classes, image_size)
    _test_classification(model, 1, n_classes, image_size)


def _test_classification(model: ClassificationModel, batch_size: int, n_classes=11, image_size=(96, 96)):
    expected_prob_tensor_shape = (batch_size, n_classes)
    expected_label_tensor_shape = (batch_size,)

    test_tensor = try_cuda(gen_random_tensor(batch_size, image_size))

    labels, probs = model.predict_label(test_tensor)
    assert_tensor_shape(labels, expected_label_tensor_shape, "output label shape")
    assert_tensor_shape(probs, expected_prob_tensor_shape, "output prob shape")


def _test_attention_classification(model: AttentionClassificationModel, batch_size: int, n_classes=11,
                                   image_size=(96, 96)):
    model = try_cuda(model)
    expected_prob_tensor_shape = (batch_size, n_classes)
    expected_label_tensor_shape = (batch_size,)
    expected_attention_shape_length = 3  # (batch size, height, width)

    test_tensor = try_cuda(gen_random_tensor(batch_size, image_size))

    labels, probs, attention_map = model.predict_label_and_heatmap(test_tensor)
    assert_tensor_shape(labels, expected_label_tensor_shape, "output label shape")
    assert_tensor_shape(probs, expected_prob_tensor_shape, "output prob shape")
    assert len(attention_map.shape) == expected_attention_shape_length

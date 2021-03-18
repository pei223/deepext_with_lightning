from deepext_with_lightning.dataset.detection import VOCDataset


def test_reading_file():
    images_path = "test/dataset/test_images"
    annotations_path = "test/dataset/test_bboxes"
    class_index_dict = {
        "aeroplane": 0,
        "dog": 1,
        "chair": 2
    }
    dataset = VOCDataset.create(images_path, annotations_path, transforms=None, class_index_dict=class_index_dict)
    data = dataset[0]
    assert len(dataset) == 2
    assert isinstance(data, tuple) and len(data) == 2


def test_reading_file_with_invalid_suffix():
    images_path = "test/dataset/test_images"
    annotations_path = "test/dataset/test_bboxes"
    class_index_dict = {
        "aeroplane": 0,
        "dog": 1,
        "chair": 2
    }
    dataset = VOCDataset.create(images_path, annotations_path, transforms=None, class_index_dict=class_index_dict,
                                valid_suffixes=["*.png"])
    assert len(dataset) == 0

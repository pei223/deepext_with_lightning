from deepext_with_lightning.dataset.segmentation import IndexImageDataset


def test_reading_file():
    images_path = "test/dataset/test_images"
    mask_images_path = "test/dataset/test_masks"
    dataset = IndexImageDataset.create(images_path, mask_images_path, transforms=None)
    data = dataset[0]
    assert len(dataset) == 2
    assert isinstance(data, tuple) and len(data) == 2


def test_reading_file_with_invalid_suffix():
    images_path = "test/dataset/test_images"
    mask_images_path = "test/dataset/test_masks"
    dataset = IndexImageDataset.create(images_path, mask_images_path, transforms=None, valid_suffixes=["*.png"])
    assert len(dataset) == 0

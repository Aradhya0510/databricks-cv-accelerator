"""Dataset construction against synthetic COCO and ImageFolder data."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from src.tasks.classification.data import ImageFolderClassificationDataset  # noqa: E402
from src.tasks.detection.data import COCODetectionDataset  # noqa: E402
from src.utils.coco import COCODataSource, detect_annotation_type  # noqa: E402


# ---------------------------------------------------------------------------
# COCO source
# ---------------------------------------------------------------------------

def test_category_ids_are_mapped_to_contiguous_indices(coco_annotation_file):
    """Real COCO exports have sparse, non-zero-based category ids."""
    source = COCODataSource(str(coco_annotation_file))
    assert source.num_classes == 2
    assert sorted(source.cat_to_idx.keys()) == [7, 12]
    assert sorted(source.cat_to_idx.values()) == [0, 1]
    assert source.class_names == ["cat", "dog"]


def test_boxes_are_converted_to_absolute_xyxy(coco_annotation_file):
    source = COCODataSource(str(coco_annotation_file))
    target = source.get_detection_target(1)

    # COCO bbox [8, 8, 16, 16] is xywh -> xyxy is [8, 8, 24, 24].
    assert target["boxes"][0].tolist() == [8.0, 8.0, 24.0, 24.0]
    assert target["labels"].tolist() == [0, 1]


def test_crowd_annotations_are_skipped(coco_annotation_file):
    """Image 2 has a single iscrowd=1 annotation, so it has no targets."""
    source = COCODataSource(str(coco_annotation_file))
    target = source.get_detection_target(2)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_annotation_type_is_detected(coco_annotation_file):
    assert detect_annotation_type(str(coco_annotation_file)).value == "instances"


# ---------------------------------------------------------------------------
# Detection dataset + augmentation integration
# ---------------------------------------------------------------------------

def test_detection_dataset_yields_image_and_target(coco_annotation_file, image_dir):
    ds = COCODetectionDataset(str(image_dir), str(coco_annotation_file))
    assert len(ds) == 2

    image, target = ds[0]
    assert image.size == (64, 48)
    assert target["boxes"].shape == (2, 4)


def test_augmentation_reaches_the_detection_dataset(coco_annotation_file, image_dir):
    """A deterministic flip must change the boxes the dataset hands back."""
    pytest.importorskip("albumentations")
    from src.tasks.augmentation import build_augmentations

    plain = COCODetectionDataset(str(image_dir), str(coco_annotation_file))
    flipped = COCODetectionDataset(
        str(image_dir), str(coco_annotation_file),
        augmentations=build_augmentations(
            True, {"horizontal_flip": 1.0}, task_type="detection",
        ),
    )

    _, plain_target = plain[0]
    _, flip_target = flipped[0]

    # Image is 64 wide; box [8, 8, 24, 24] mirrors to [40, 8, 56, 24].
    assert plain_target["boxes"][0].tolist() == [8.0, 8.0, 24.0, 24.0]
    assert flip_target["boxes"][0].tolist() == pytest.approx([40.0, 8.0, 56.0, 24.0])
    # Labels ride along unchanged.
    assert sorted(flip_target["labels"].tolist()) == sorted(plain_target["labels"].tolist())


# ---------------------------------------------------------------------------
# Classification dataset
# ---------------------------------------------------------------------------

def test_classification_discovers_classes_from_subdirectories(classification_dir):
    ds = ImageFolderClassificationDataset(str(classification_dir))
    assert ds.class_names == ["cat", "dog"]
    assert len(ds) == 4


def test_classification_rejects_class_names_with_no_directory(classification_dir):
    """A configured class the data does not have used to yield zero samples."""
    with pytest.raises(ValueError, match="no such subdirectories"):
        ImageFolderClassificationDataset(
            str(classification_dir), class_names=["cat", "dog", "bird"],
        )


def test_classification_augmentation_changes_pixels(classification_dir):
    """Augmentation must actually reach the pixels the dataset returns.

    Transforms fire probabilistically (brightness_contrast is p=0.5), so this
    draws repeatedly and asserts that at least one draw differs — checking
    that the pipeline is wired in, without depending on any single draw.
    """
    pytest.importorskip("albumentations")
    import random

    from src.tasks.augmentation import build_augmentations

    plain = ImageFolderClassificationDataset(str(classification_dir))
    augmented = ImageFolderClassificationDataset(
        str(classification_dir),
        augmentations=build_augmentations(
            True, {"brightness_contrast": 0.9}, task_type="classification",
        ),
    )

    plain_px, plain_label = plain[0]
    plain_arr = np.asarray(plain_px)

    random.seed(0)
    np.random.seed(0)
    differed = False
    for _ in range(20):
        aug_px, aug_label = augmented[0]
        assert aug_label == plain_label
        if not np.array_equal(plain_arr, np.asarray(aug_px)):
            differed = True
            break

    assert differed, "augmentation never altered the image in 20 draws"

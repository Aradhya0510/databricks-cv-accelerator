"""Augmentation config resolution and label co-transforms.

The config block was previously inert, so these check both that it is read at
all and that boxes and masks travel with the image when it is transformed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tasks.augmentation import (
    KNOWN_KEYS,
    build_augmentations,
    resolve_augmentation_spec,
    unknown_augmentation_keys,
)

albumentations = pytest.importorskip("albumentations")


# ---------------------------------------------------------------------------
# Spec resolution
# ---------------------------------------------------------------------------

def test_augment_false_disables_everything():
    assert resolve_augmentation_spec(False, {"horizontal_flip": True}) is None
    assert build_augmentations(False, {"horizontal_flip": True}) is None


def test_augment_true_with_a_block_uses_the_block():
    spec = resolve_augmentation_spec(True, {"rotation": 10})
    assert spec == {"rotation": 10}


def test_augment_true_without_a_block_gets_a_safe_default():
    assert resolve_augmentation_spec(True, None) == {"horizontal_flip": True}


def test_augment_as_a_dict_is_the_spec_itself():
    assert resolve_augmentation_spec({"vertical_flip": True}, None) == {"vertical_flip": True}


def test_unknown_keys_are_reported_not_silently_dropped():
    unknown = unknown_augmentation_keys({"horizontal_flip": True, "mixup": 0.2})
    assert unknown == {"mixup"}


def test_every_shipped_config_key_is_implemented():
    """The keys the repo's own configs use must all be honoured."""
    shipped = {"horizontal_flip", "vertical_flip", "rotation",
               "brightness_contrast", "hue_saturation"}
    assert shipped <= KNOWN_KEYS


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def test_builds_a_pipeline_for_detection():
    pipeline = build_augmentations(
        True, {"horizontal_flip": True, "rotation": 10}, task_type="detection",
    )
    assert pipeline is not None
    assert pipeline.processors.get("bboxes") is not None


def test_classification_pipeline_has_no_bbox_params():
    pipeline = build_augmentations(
        True, {"horizontal_flip": True}, task_type="classification",
    )
    assert pipeline is not None
    assert pipeline.processors.get("bboxes") is None


# ---------------------------------------------------------------------------
# Label co-transforms — the part that breaks silently if it is wrong
# ---------------------------------------------------------------------------

def test_horizontal_flip_moves_boxes_with_the_image():
    """A deterministic flip must mirror the box's x coordinates."""
    pipeline = build_augmentations(
        True, {"horizontal_flip": 1.0}, task_type="detection",
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # xyxy in absolute pixels, hugging the left edge.
    result = pipeline(image=image, bboxes=[[10.0, 20.0, 30.0, 40.0]], labels=[1])

    x1, y1, x2, y2 = result["bboxes"][0]
    # Mirrored about x=50: [10,30] -> [70,90].  y is untouched.
    assert x1 == pytest.approx(70.0, abs=1e-4)
    assert x2 == pytest.approx(90.0, abs=1e-4)
    assert y1 == pytest.approx(20.0, abs=1e-4)
    assert y2 == pytest.approx(40.0, abs=1e-4)
    assert result["labels"] == [1]


def test_flip_preserves_mask_class_values():
    """Geometric transforms must not interpolate across class ids."""
    pipeline = build_augmentations(
        True, {"horizontal_flip": 1.0}, task_type="segmentation",
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[:, :5] = 7  # left half is class 7

    result = pipeline(image=image, mask=mask)
    out = result["mask"]

    assert set(np.unique(out).tolist()) == {0, 7}
    assert (out[:, 5:] == 7).all()  # flipped to the right half
    assert (out[:, :5] == 0).all()


def test_boxes_pushed_out_of_frame_are_dropped():
    """A box left almost entirely outside the crop must not become a target."""
    pipeline = build_augmentations(
        True, {"horizontal_flip": 1.0}, task_type="detection",
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Valid box entirely inside the frame — should survive.
    result = pipeline(image=image, bboxes=[[10.0, 10.0, 90.0, 90.0]], labels=[3])
    assert len(result["bboxes"]) == 1


def test_empty_box_list_is_handled():
    """Images with no annotations are legal and must not raise."""
    pipeline = build_augmentations(True, {"horizontal_flip": 1.0}, task_type="detection")
    result = pipeline(image=np.zeros((50, 50, 3), dtype=np.uint8), bboxes=[], labels=[])
    assert result["bboxes"] == []

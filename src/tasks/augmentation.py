"""Data augmentation driven by the ``data.augmentations`` config block.

Every shipped config declared ``augment`` and an ``augmentations`` block, and
nothing read them — so users reasonably believed flips and colour jitter were
on and got worse results than they should have, with no signal that the
settings had been ignored.

Albumentations is used because it co-transforms bounding boxes and masks along
with the image, which is what detection and segmentation need.  It is an
optional dependency: when it is not installed, :func:`build_augmentations`
returns ``None`` and training proceeds unaugmented rather than failing.

Augmentation is applied to the *training* split only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

# Recognised keys in the ``augmentations`` config block.  Anything else is
# reported rather than silently dropped, which is how the original block came
# to be entirely inert without anyone noticing.
KNOWN_KEYS = {
    "horizontal_flip",
    "vertical_flip",
    "rotation",
    "brightness_contrast",
    "hue_saturation",
    "color_jitter",
    "random_crop",
    "random_resized_crop",
    "blur",
    "gaussian_noise",
}


def resolve_augmentation_spec(
    augment: Union[bool, Dict[str, Any]],
    augmentations: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalise the two config fields into one spec dict, or ``None``.

    ``augment`` may be a bool (use ``augmentations``, or a sensible default set)
    or a dict (the spec itself, for configs that inline it).
    """
    if isinstance(augment, dict):
        return augment or None

    if not augment:
        return None

    if augmentations:
        return augmentations

    # ``augment: true`` with no explicit block: a conservative default that is
    # safe for every task (no geometric distortion beyond a horizontal flip).
    return {"horizontal_flip": True}


def unknown_augmentation_keys(spec: Optional[Dict[str, Any]]) -> set:
    """Keys in *spec* this module does not implement."""
    if not spec:
        return set()
    return set(spec) - KNOWN_KEYS


def build_augmentations(
    augment: Union[bool, Dict[str, Any]],
    augmentations: Optional[Dict[str, Any]] = None,
    *,
    task_type: str = "detection",
    image_size: Optional[int] = None,
):
    """Build an albumentations pipeline, or ``None`` when nothing is configured.

    Args:
        augment: The ``data.augment`` field.
        augmentations: The ``data.augmentations`` block.
        task_type: Selects the label co-transform — bounding boxes for
            detection, masks for segmentation, neither for classification.
        image_size: Target size, needed only for ``random_resized_crop``.
    """
    spec = resolve_augmentation_spec(augment, augmentations)
    if not spec:
        return None

    unknown = unknown_augmentation_keys(spec)
    if unknown:
        print(
            f"Warning: ignoring unrecognised augmentation keys {sorted(unknown)}. "
            f"Supported keys: {sorted(KNOWN_KEYS)}"
        )

    try:
        import albumentations as A
    except ImportError:
        print(
            "Warning: augmentations are configured but albumentations is not "
            "installed, so training will run unaugmented. "
            "Add 'albumentations' to requirements_runtime.txt to enable them."
        )
        return None

    transforms = _build_transform_list(A, spec, image_size)
    if not transforms:
        return None

    kwargs: Dict[str, Any] = {}
    if task_type == "detection":
        # ``pascal_voc`` is [x_min, y_min, x_max, y_max] in absolute pixels,
        # which is what COCODataSource.get_detection_target produces.
        kwargs["bbox_params"] = A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            # Drop boxes an augmentation has pushed almost entirely out of frame,
            # which would otherwise become degenerate targets.
            min_visibility=0.2,
        )

    return A.Compose(transforms, **kwargs)


def _build_transform_list(A, spec: Dict[str, Any], image_size: Optional[int]) -> list:
    """Translate the config spec into albumentations transforms."""
    transforms = []

    if spec.get("horizontal_flip"):
        transforms.append(A.HorizontalFlip(p=_prob(spec["horizontal_flip"], 0.5)))

    if spec.get("vertical_flip"):
        transforms.append(A.VerticalFlip(p=_prob(spec["vertical_flip"], 0.5)))

    rotation = spec.get("rotation")
    if rotation:
        transforms.append(A.Rotate(limit=float(rotation), p=0.5))

    bc = spec.get("brightness_contrast")
    if bc:
        amount = float(bc) if not isinstance(bc, bool) else 0.2
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=amount, contrast_limit=amount, p=0.5,
            )
        )

    hs = spec.get("hue_saturation")
    if hs:
        amount = float(hs) if not isinstance(hs, bool) else 0.1
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=int(amount * 180),
                sat_shift_limit=int(amount * 255),
                val_shift_limit=int(amount * 255),
                p=0.5,
            )
        )

    if spec.get("blur"):
        transforms.append(A.Blur(blur_limit=3, p=0.2))

    if spec.get("gaussian_noise"):
        transforms.append(A.GaussNoise(p=0.2))

    cj = spec.get("color_jitter")
    if cj:
        amount = float(cj) if not isinstance(cj, bool) else 0.2
        transforms.append(
            A.ColorJitter(
                brightness=amount, contrast=amount,
                saturation=amount, hue=min(amount, 0.5), p=0.5,
            )
        )

    # ``random_crop`` and ``random_resized_crop`` are the same transform here:
    # a scale-and-crop back to the model's input size, which is what both
    # names mean in the shipped configs.
    if (spec.get("random_crop") or spec.get("random_resized_crop")) and image_size:
        transforms.append(
            A.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.7, 1.0), p=0.5,
            )
        )

    return transforms


def _prob(value: Any, default: float) -> float:
    """A config entry may be ``true`` or an explicit probability."""
    if isinstance(value, bool):
        return default
    return float(value)

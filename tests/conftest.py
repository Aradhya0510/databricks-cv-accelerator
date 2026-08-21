"""Shared fixtures.

Every fixture here is offline: models are constructed from configs in code
rather than downloaded, so the suite runs in CI with no network and no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def coco_annotation_file(tmp_path: Path) -> Path:
    """A minimal but valid COCO ``instances`` file: 2 images, 3 annotations.

    Category ids are deliberately non-contiguous and do not start at 0, which
    is what real COCO exports look like and what the contiguous-index mapping
    in COCODataSource exists to handle.
    """
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 64, "height": 48},
            {"id": 2, "file_name": "b.jpg", "width": 64, "height": 48},
        ],
        "annotations": [
            # image 1: one box of each category
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [8, 8, 16, 16],
             "area": 256, "iscrowd": 0, "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]]},
            {"id": 2, "image_id": 1, "category_id": 12, "bbox": [32, 8, 16, 24],
             "area": 384, "iscrowd": 0, "segmentation": [[32, 8, 48, 8, 48, 32, 32, 32]]},
            # image 2: one crowd annotation, which must be skipped
            {"id": 3, "image_id": 2, "category_id": 7, "bbox": [0, 0, 32, 32],
             "area": 1024, "iscrowd": 1, "segmentation": [[0, 0, 32, 0, 32, 32, 0, 32]]},
        ],
        "categories": [
            {"id": 7, "name": "cat", "supercategory": "animal"},
            {"id": 12, "name": "dog", "supercategory": "animal"},
        ],
    }
    path = tmp_path / "instances_test.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Directory holding the two images referenced by ``coco_annotation_file``."""
    from PIL import Image

    d = tmp_path / "images"
    d.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (64, 48), color=(120, 120, 120)).save(d / name)
    return d


@pytest.fixture
def classification_dir(tmp_path: Path) -> Path:
    """ImageFolder-style tree: two classes, two images each."""
    from PIL import Image

    root = tmp_path / "cls"
    for cls, color in (("cat", (200, 40, 40)), ("dog", (40, 40, 200))):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (32, 32), color=color).save(d / f"{i}.png")
    return root

# Databricks notebook source
# MAGIC %md
# MAGIC # 01. Data Exploration (COCO Detection)
# MAGIC
# MAGIC EDA for COCO-format detection datasets. Uses `COCODetectionDataset` and
# MAGIC `pycocotools.COCO` directly — no model or GPU required.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import sys, os
from pathlib import Path

sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref')

from src.config.schema import load_config

# --- Paths (customise for your workspace) ---
CATALOG = "your_catalog"
SCHEMA = "your_schema"
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_yolos_config.yaml"

config = load_config(CONFIG_PATH)

print(f"Task:        {config.model.task_type}")
print(f"Train data:  {config.data.train_data_path}")
print(f"Val data:    {config.data.val_data_path}")
print(f"Annotation:  {config.data.train_annotation_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Dataset & Compute Stats

# COMMAND ----------

from pycocotools.coco import COCO
from src.tasks.detection.data import COCODetectionDataset

# Load COCO annotations directly for stats
coco = COCO(config.data.train_annotation_file)

img_ids = coco.getImgIds()
ann_ids = coco.getAnnIds()
cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)
cat_names = {c["id"]: c["name"] for c in cats}

print(f"Images:      {len(img_ids)}")
print(f"Annotations: {len(ann_ids)}")
print(f"Categories:  {len(cat_ids)}")
print(f"Category names: {list(cat_names.values())[:20]}{'...' if len(cat_names) > 20 else ''}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Class Distribution

# COMMAND ----------

import matplotlib.pyplot as plt
from collections import Counter

# Count annotations per category
all_anns = coco.loadAnns(ann_ids)
class_counts = Counter(cat_names[ann["category_id"]] for ann in all_anns)
sorted_counts = class_counts.most_common()

names = [c[0] for c in sorted_counts]
counts = [c[1] for c in sorted_counts]

fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.3)))
ax.barh(names[::-1], counts[::-1])
ax.set_xlabel("Number of Annotations")
ax.set_title("Class Distribution (Training Set)")
plt.tight_layout()
plt.show()

print(f"Most common:  {sorted_counts[0]}")
print(f"Least common: {sorted_counts[-1]}")
print(f"Imbalance ratio: {sorted_counts[0][1] / max(sorted_counts[-1][1], 1):.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Bounding Box Size Distribution

# COMMAND ----------

import numpy as np

areas = [ann["area"] for ann in all_anns]

# COCO thresholds: small < 32^2, medium < 96^2, large >= 96^2
small = sum(1 for a in areas if a < 32**2)
medium = sum(1 for a in areas if 32**2 <= a < 96**2)
large = sum(1 for a in areas if a >= 96**2)

print(f"Small  (< 32x32):    {small:,} ({small / len(areas):.1%})")
print(f"Medium (32-96):       {medium:,} ({medium / len(areas):.1%})")
print(f"Large  (>= 96x96):   {large:,} ({large / len(areas):.1%})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Area histogram
axes[0].hist(np.sqrt(areas), bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("√Area (pixels)")
axes[0].set_ylabel("Count")
axes[0].set_title("Bounding Box Size Distribution")
axes[0].axvline(32, color="red", linestyle="--", label="Small/Med boundary")
axes[0].axvline(96, color="orange", linestyle="--", label="Med/Large boundary")
axes[0].legend()

# Pie chart
axes[1].pie([small, medium, large], labels=["Small", "Medium", "Large"],
            autopct="%1.1f%%", startangle=90)
axes[1].set_title("Size Categories (COCO Thresholds)")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sample Images with Bounding Boxes

# COMMAND ----------

from PIL import Image, ImageDraw, ImageFont
import random

# Create dataset (without model-specific transforms for viz)
dataset = COCODetectionDataset(
    root_dir=config.data.train_data_path,
    annotation_file=config.data.train_annotation_file,
    transform=None,
)

# Get 9 random samples
indices = random.sample(range(len(dataset)), min(9, len(dataset)))

fig, axes = plt.subplots(3, 3, figsize=(18, 18))

for ax, idx in zip(axes.flat, indices):
    image, target = dataset[idx]
    draw = ImageDraw.Draw(image)

    boxes = target["boxes"]
    labels = target["labels"]

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        cat_id = list(dataset.cat_to_idx.keys())[list(dataset.cat_to_idx.values()).index(int(label))]
        name = cat_names.get(cat_id, str(int(label)))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 12), name, fill="red")

    ax.imshow(image)
    ax.set_title(f"Image {target['image_id'].item()} — {len(boxes)} objects")
    ax.axis("off")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Annotation Quality Checks

# COMMAND ----------

# Check for common issues
issues = {
    "zero_area_boxes": 0,
    "out_of_bounds": 0,
    "images_with_no_annotations": 0,
}

imgs_with_anns = set()
for ann in all_anns:
    imgs_with_anns.add(ann["image_id"])

    # Zero-area check
    if ann["area"] <= 0:
        issues["zero_area_boxes"] += 1

    # Out-of-bounds check
    bbox = ann["bbox"]  # [x, y, w, h]
    img_info = coco.loadImgs(ann["image_id"])[0]
    img_w, img_h = img_info["width"], img_info["height"]

    if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > img_w + 1 or bbox[1] + bbox[3] > img_h + 1:
        issues["out_of_bounds"] += 1

issues["images_with_no_annotations"] = len(img_ids) - len(imgs_with_anns)

print("Annotation Quality Report")
print("=" * 40)
for k, v in issues.items():
    status = "✅" if v == 0 else "⚠️"
    print(f"  {status} {k}: {v}")

total_issues = sum(issues.values())
if total_issues == 0:
    print("\n✅ No quality issues found!")
else:
    print(f"\n⚠️ Found {total_issues} total issues — review before training.")

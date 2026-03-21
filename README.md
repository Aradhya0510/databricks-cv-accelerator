# Databricks CV Accelerator

A production-ready framework for fine-tuning computer vision models on Databricks. Drop in a config, point it at your data, and get a trained, evaluated, and deployed model — with full MLflow tracking, multi-GPU support, and a Streamlit UI.

## Why This Framework

Fine-tuning CV models on Databricks involves gluing together data loading, HuggingFace models, distributed training, experiment tracking, model packaging, and serving. This framework does it for you:

- **Config-driven.** One YAML file controls the entire pipeline — model, data, training, serving. No boilerplate code to write.
- **COCO as the standard data framework.** MS COCO (`pycocotools`) is the unified annotation layer. A single `instances_*.json` can drive both detection and segmentation. The shared `COCODataSource` class wraps `pycocotools.COCO` with task-specific accessors — bounding boxes for detection, `annToMask()` for instance masks, flattened class maps for semantic segmentation. COCO panoptic and ADE20K-style masks are supported as alternatives.
- **Model adapters as config, not class hierarchies.** Each model family's quirks — pixel mask requirements, box format, output attributes, model API type — are captured in a lightweight dataclass (`DetectionFamilyConfig`, `SegmentationFamilyConfig`). A `detect_*_family()` function selects the right config by substring matching on the model name. Adding a new architecture means adding a dict entry — no new class, no inheritance.
- **Task-agnostic.** Detection, classification, and segmentation work today. Add new tasks by implementing a single class. The engine, evaluation, serving, and monitoring layers all adapt automatically.
- **Databricks-native.** Unity Catalog Volumes for data, MLflow for tracking, Model Serving for deployment, system tables for monitoring. Everything wired together.
- **Multi-GPU out of the box.** HF Trainer handles DDP natively. Pass `--num_gpus 4` and it works. No Spark orchestration overhead for single-node.
- **Full lifecycle.** Train, evaluate (mAP/accuracy + error analysis + latency benchmarks), register to Unity Catalog, deploy to Model Serving, and monitor — all from the same framework.

## What You Can Do

| Task | Models | Data Format | Eval Metric |
|---|---|---|---|
| Object Detection | DETR, Conditional DETR, RT-DETR, DETA, YOLOS, any `AutoModelForObjectDetection` | COCO instances JSON | mAP |
| Image Classification | ViT, ResNet, any `AutoModelForImageClassification` | ImageFolder (class_name/image.jpg) | Accuracy, F1 |
| Segmentation | SegFormer, Mask2Former, OneFormer, MaskFormer, UperNet, BEiT, DPT | COCO instances, COCO panoptic, or ADE20K masks | mIoU |

To switch models, change two lines:

```yaml
model:
  model_name: "nvidia/segformer-b3-finetuned-ade-512-512"  # any HF model
  task_type: segmentation                                    # or "detection", "classification"
```

## Architecture

```
Config YAML ─→ PipelineConfig (Pydantic) ─→ TrainingEngine ─→ CVTrainer (HF Trainer)
                                                │
                                   TaskRegistry │ dispatches to:
                                                ├── DetectionTask
                                                ├── ClassificationTask
                                                └── SegmentationTask
```

The framework separates **what** (task-specific logic) from **how** (training loop, evaluation, serving):

- **Tasks** (`src/tasks/`) provide model loading, data handling, loss, and eval metrics
- **Engine** (`src/engine/`) runs the training loop and DDP — task-agnostic
- **Evaluation** (`src/evaluation/`) runs standalone eval, error analysis, benchmarks — task-aware
- **Serving** (`src/serving/`) wraps models as PyFunc for Model Serving — task-aware
- **Monitoring** (`src/monitoring/`) queries system tables for endpoint observability

## Project Structure

```
src/
├── config/schema.py              # Pydantic v2 config + YAML loader
├── registry.py                   # TaskRegistry (@register decorator)
├── engine/
│   ├── engine.py                 # TrainingEngine: config → train → metrics
│   ├── trainer.py                # CVTrainer (HF Trainer + task hooks)
│   └── callbacks.py              # Checkpoint + early stopping callbacks
├── tasks/
│   ├── detection/                # DetectionTask, COCO dataset, config-driven adapters
│   ├── classification/           # ClassificationTask, ImageFolder dataset
│   └── segmentation/             # SegmentationTask, COCO + ADE20K datasets, config-driven adapters
├── evaluation/
│   └── engine.py                 # EvaluationEngine: metrics, error analysis, benchmarks
├── serving/
│   ├── pyfunc.py                 # Detection, Classification, Segmentation PyFunc models
│   ├── registration.py           # register_model() → Unity Catalog
│   └── deployment.py             # deploy_endpoint() → Model Serving
├── monitoring/
│   └── endpoint_monitor.py       # Health, request metrics, prediction distribution
└── utils/
    ├── coco.py                   # COCODataSource: shared pycocotools wrapper for all tasks
    ├── coco_eval.py              # COCOeval wrappers for standardized COCO metrics
    └── environment.py            # Databricks environment detection, GPU helpers

jobs/
├── train.py                      # Training CLI
├── evaluate.py                   # Evaluation CLI
├── deploy.py                     # Registration + deployment CLI
└── monitor.py                    # Monitoring report CLI

notebooks/
├── 01_data_exploration.py        # EDA: stats, class distribution, quality checks
├── 02_model_training.py          # Interactive training
├── 03_model_evaluation.py        # Evaluation + error analysis
├── 04_model_deployment.py        # PyFunc test → register → deploy → smoke test
└── 05_model_monitoring.py        # Endpoint health + metrics + alerts

configs/                          # YAML configs per model
lakehouse_app/                    # Streamlit UI (9 pages, full lifecycle)
```

## Quick Start

See **[GETTING_STARTED.md](GETTING_STARTED.md)** for the full setup guide. The short version:

```bash
# 1. Clone
git clone <repo-url> && cd databricks-cv-accelerator

# 2. Pick a config, update paths
cp configs/classification_vit_config.yaml configs/my_config.yaml
# Edit data paths, MLflow experiment, checkpoint dirs

# 3. Train
python jobs/train.py --config_path configs/my_config.yaml

# 4. Evaluate
python jobs/evaluate.py --config_path configs/my_config.yaml --checkpoint_path /path/to/model

# 5. Deploy
python jobs/deploy.py --config_path configs/my_config.yaml --run_id <mlflow_run_id> \
    --model_name catalog.schema.my_model --endpoint_name my-endpoint
```

## Extending the Framework

### Add a New Model (Same Task)

For most models, just change `model_name` in your config — if it works with the corresponding HF AutoModel class, it works here:

| Task | AutoModel class |
|---|---|
| Detection | `AutoModelForObjectDetection` |
| Classification | `AutoModelForImageClassification` |
| Segmentation (semantic) | `AutoModelForSemanticSegmentation` |
| Segmentation (universal) | `AutoModelForUniversalSegmentation` |

For models with non-standard input/output formats, add a family config entry to the appropriate `adapters.py`. Detection example:

```python
_FAMILY_CONFIGS["my-arch"] = DetectionFamilyConfig(
    requires_pixel_mask=True,
    box_format="cxcywh_normalized",
    output_logits_attr="logits",
    output_boxes_attr="pred_boxes",
)
```

Segmentation example:

```python
_FAMILY_CONFIGS["my-seg-model"] = SegmentationFamilyConfig(
    model_type="semantic",
    requires_pixel_mask=False,
    reduce_labels=False,
    output_logits_attr="logits",
)
```

Both `detect_detection_family()` and `detect_segmentation_family()` match model names by substring, so `org/my-arch-resnet-50` will automatically use the `"my-arch"` config. No new class needed.

### Add a New Task

Create a new directory under `src/tasks/` and implement the task interface:

```python
from src.registry import TaskRegistry

@TaskRegistry.register("depth_estimation")
class DepthEstimationTask:
    def get_model(self, model_cfg): ...
    def get_train_dataset(self, config): ...
    def get_val_dataset(self, config): ...
    def get_collate_fn(self): ...
    def create_optimizer_and_scheduler(self, model, config, num_training_steps): ...
    def compute_loss(model, inputs, return_outputs=False): ...
    def get_eval_fn(self, model_cfg): ...
```

Then add `import src.tasks.depth_estimation` to `src/engine/engine.py` and `src/evaluation/engine.py`. The training engine, evaluation pipeline, and job scripts all work automatically.

### Data Formats: COCO-First Architecture

The framework uses **MS COCO as the standard annotation layer** across tasks. A shared `COCODataSource` class wraps `pycocotools.COCO` so that detection and segmentation share annotation parsing, category mapping, and mask decoding. The same `instances_*.json` file can drive both detection (bounding boxes) and segmentation (per-instance masks via `annToMask()`).

Three segmentation data formats are supported, auto-detected from the annotation file:

**1. COCO instances (preferred)** — set `train_annotation_file` to `instances_*.json`:

```
data/
├── train2017/                    # images
├── instances_train2017.json      # same file used for detection
├── val2017/
└── instances_val2017.json
```

This is the recommended path. `pycocotools.annToMask()` extracts per-instance binary masks and composes them into the format HF processors expect. For universal models (Mask2Former), instance-level information is preserved for Hungarian matching loss. For semantic models (SegFormer), masks are flattened to class-index maps automatically.

**2. COCO panoptic** — set `train_annotation_file` to `panoptic_*.json`:

```
data/
├── train2017/                    # images
├── panoptic_train2017/           # RGB-encoded panoptic PNGs
├── panoptic_train2017.json       # annotation file
├── val2017/
├── panoptic_val2017/
└── panoptic_val2017.json
```

Use when you specifically need panoptic segmentation (stuff + things) with COCO panoptic-format annotations.

**3. ADE20K-style masks (fallback)** — omit `annotation_file`:

```
data/train/
├── images/
│   ├── 0001.jpg
│   └── ...
└── masks/
    ├── 0001.png                  # single-channel, pixel value = class ID
    └── ...
```

Zero-annotation fallback for datasets that only provide semantic mask PNGs.

All three formats work with all segmentation models. The framework auto-detects the annotation type by inspecting the JSON structure (`"bbox"` → instances, `"segments_info"` → panoptic).

## Supported Models

### Detection

| Model | HuggingFace ID | Config |
|---|---|---|
| DETR | `facebook/detr-resnet-50` | `detection_detr_config.yaml` |
| Conditional DETR | `microsoft/conditional-detr-resnet-50` | `detection_conditional_detr_config.yaml` |
| RT-DETR | `PekingU/rtdetr_r50vd` | `detection_rtdetr_config.yaml` |
| DETA | `jozhang97/deta-swin-large` | `detection_deta_config.yaml` |
| YOLOS | `hustvl/yolos-base` | `detection_yolos_config.yaml` |

### Classification

| Model | HuggingFace ID | Config |
|---|---|---|
| ViT | `google/vit-base-patch16-224` | `classification_vit_config.yaml` |
| Any `AutoModelForImageClassification` model works — just change `model_name`. |

### Segmentation

| Model | HuggingFace ID | Type | Config |
|---|---|---|---|
| SegFormer | `nvidia/segformer-b3-finetuned-ade-512-512` | Semantic | `segmentation_segformer_config.yaml` |
| Mask2Former | `facebook/mask2former-swin-base-ade-semantic` | Universal | `segmentation_mask2former_config.yaml` |
| OneFormer | `shi-labs/oneformer_ade20k_swin_large` | Universal | — |
| MaskFormer | `facebook/maskformer-swin-large-ade` | Universal | — |
| UperNet | `openmmlab/upernet-swin-base` | Semantic | — |
| BEiT | `microsoft/beit-base-finetuned-ade-640-640` | Semantic | — |
| DPT | `Intel/dpt-large-ade` | Semantic | — |

## Multi-GPU

- **Single-node (default):** HF Trainer handles DDP natively. Pass `--num_gpus N` or auto-detect.
- **Multi-node (opt-in):** Pass `--distributed torchd` to use TorchDistributor across Spark workers.

```bash
# 4x A10G on a single node
python jobs/train.py --config_path configs/my_config.yaml --num_gpus 4

# Multi-node via TorchDistributor (only when needed)
python jobs/train.py --config_path configs/my_config.yaml --distributed torchd
```

## Lakehouse App

The `lakehouse_app/` directory contains a Streamlit UI covering the full lifecycle: config setup, data EDA, training, evaluation, model registration, deployment, inference testing, run history, and endpoint monitoring. Deploy as a Databricks App — see [`lakehouse_app/README.md`](lakehouse_app/README.md).

## Built With

[HuggingFace Transformers](https://huggingface.co/docs/transformers) | [MLflow](https://mlflow.org) | [Databricks](https://databricks.com) | [PyTorch](https://pytorch.org) | [torchmetrics](https://torchmetrics.readthedocs.io)

## License

MIT License — see [LICENSE](LICENSE).

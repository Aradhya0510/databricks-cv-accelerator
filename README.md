# Databricks CV Accelerator

A production-ready framework for fine-tuning computer vision models on Databricks. Drop in a config, point it at your data, and get a trained, evaluated, and deployed model — with full MLflow tracking, multi-GPU support, and a Streamlit UI.

## Why This Framework

Fine-tuning CV models on Databricks involves gluing together data loading, HuggingFace models, distributed training, experiment tracking, model packaging, and serving. This framework does it for you:

- **Config-driven.** One YAML file controls the entire pipeline — model, data, training, serving. No boilerplate code to write.
- **COCO as the standard data framework.** MS COCO (`pycocotools`) is the unified annotation layer. A single `instances_*.json` can drive both detection and segmentation. The shared `COCODataSource` class wraps `pycocotools.COCO` with task-specific accessors — bounding boxes for detection, `annToMask()` for instance masks, flattened class maps for semantic segmentation. COCO panoptic and ADE20K-style masks are supported as alternatives.
- **Model adapters as config, not class hierarchies.** Each model family's quirks — pixel mask requirements, box format, output attributes, model API type — are captured in a lightweight dataclass (`DetectionFamilyConfig`, `SegmentationFamilyConfig`). A `detect_*_family()` function selects the right config by substring matching on the model name. Adding a new architecture means adding a dict entry — no new class, no inheritance.
- **Task-agnostic.** Detection, classification, and segmentation work today. Add new tasks by implementing a single class. The engine, evaluation, serving, and monitoring layers all adapt automatically.
- **Built for Databricks AI Runtime.** Trains on serverless GPUs (A10, H100, 8xH100, 8xB300) with no clusters to manage. Unity Catalog Volumes hold the data, MLflow tracks runs, Model Serving deploys, and system tables feed monitoring.
- **Multi-GPU out of the box.** On an 8-GPU node training runs one process per GPU under real DDP — relaunched under `torchrun` from a job, or through `serverless_gpu`'s `@distributed` from a notebook. Multi-node works the same way.
- **Full lifecycle.** Train, evaluate (mAP/accuracy + error analysis + latency benchmarks), register to Unity Catalog, deploy to Model Serving, and monitor — all from the same framework.

## What You Can Do

| Task | Models | Data Format | Eval Metric |
|---|---|---|---|
| Object Detection | DETR, Conditional DETR, RT-DETR, YOLOS, any `AutoModelForObjectDetection` | COCO instances JSON | mAP |
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
│   ├── launch.py                 # torchrun relaunch from AI Runtime's node variables
│   ├── trainer.py                # CVTrainer (HF Trainer + task hooks)
│   └── callbacks.py              # Volume checkpoints, resume, early stopping
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
    ├── environment.py            # Process topology, GPU count, /Volumes → local staging
    └── manifest.py               # Run manifest: training → evaluate/deploy hand-off

jobs/
├── train.py                      # Training CLI (fans out one process per GPU)
├── evaluate.py                   # Evaluation CLI
├── deploy.py                     # Registration + deployment CLI
├── monitor.py                    # Monitoring report CLI
└── air/                          # Node-level command scripts for AI Runtime tasks

air/                              # `databricks air run` workload configs
databricks.yml                    # Bundle: AI Runtime train/eval + serverless deploy/monitor

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
# Edit data paths (UC Volumes), output.results_dir (a Volume), checkpoint dirs

# 3. Train on AI Runtime serverless GPUs, from your laptop
databricks air run -f air/train.yaml \
    --override env_variables.CV_CONFIG_PATH=configs/my_config.yaml --watch

# 4. Evaluate the model that run produced
databricks air run -f air/evaluate.yaml \
    --override env_variables.CV_CONFIG_PATH=configs/my_config.yaml --watch

# 5. Register and deploy it
python jobs/deploy.py --config_path configs/my_config.yaml \
    --model_name catalog.schema.my_model --endpoint_name my-endpoint
```

Or run the whole pipeline as a scheduled job with `databricks bundle deploy && databricks bundle run cv_training_pipeline`.

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
| YOLOS | `hustvl/yolos-base` | `detection_yolos_config.yaml` |

DETA is not supported: it was deprecated upstream and removed in transformers v5.

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

## Running on AI Runtime

The framework targets [Databricks AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/)
with the **Databricks AI environment v6** (`databricks_ai_v6`: Python 3.12,
torch 2.11, transformers 5, MLflow 3). The only extra packages are the three in
`requirements_runtime.txt`, which every job definition installs.

| Where | How | Multi-GPU |
|---|---|---|
| Laptop → GPU | `databricks air run -f air/train.yaml` | `jobs/train.py` relaunches under `torchrun` |
| Scheduled job | `databricks.yml` (`ai_runtime_task`) | same |
| Notebook | attach to AI Runtime, `TrainingEngine(config).train()` | `serverless_gpu` `@distributed` |

**One process per GPU.** Real DDP needs one process per GPU; a single process
that sees several gets `nn.DataParallel`, which is slower and treats
`batch_size` as the total rather than per-GPU batch. `jobs/train.py` therefore
checks how many processes the node needs (every GPU on it, times `NUM_NODES`)
and, if more than one, relaunches itself under `torchrun` — standalone on one
node, c10d rendezvous on AI Runtime's `MASTER_ADDR`/`MASTER_PORT` across nodes.
Pass `--num_gpus 1` to debug in a single process.

**Data.** `/Volumes` inputs are copied to local disk once per node, in
parallel, before training (`data.stage_to_local`, on by default). Image
datasets are many small files re-read every epoch, the pattern UC Volumes serve
worst.

**MLflow.** AI Runtime creates the MLflow run for CLI, bundle and
`@distributed` runs and hands it over as `MLFLOW_RUN_ID`; training logs into
that run, so its experiment (`experiment_name` in `air/*.yaml`, `experiment` in
the bundle) is where results land.

**Long runs and retries.** Jobs run for at most 14 days (CLI) or 2 days
(notebook), GPUs are on demand, and AI Runtime retries failed CLI runs. Set
`training.volume_checkpoint_dir` to a fresh Volume path and
`training.resume_from_checkpoint: latest`: the first attempt trains from
scratch and every retry resumes from the newest complete checkpoint.

**Pipeline hand-off.** AI Runtime tasks cannot pass job task values, so training
writes `run_manifest.json` (run ID and model URI) to `output.results_dir`.
`jobs/evaluate.py` and `jobs/deploy.py` read it when no model is given; make
`results_dir` a Volume path so every task sees it.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite is offline and CPU-only by design — models are built from configs in
code rather than downloaded — so it runs anywhere and cannot be broken by a
HuggingFace Hub outage.

## Lakehouse App

The `lakehouse_app/` directory contains a Streamlit UI covering the full lifecycle: config setup, data EDA, training, evaluation, model registration, deployment, inference testing, run history, and endpoint monitoring. Deploy as a Databricks App — see [`lakehouse_app/README.md`](lakehouse_app/README.md).

## Built With

[HuggingFace Transformers](https://huggingface.co/docs/transformers) | [MLflow](https://mlflow.org) | [Databricks](https://databricks.com) | [PyTorch](https://pytorch.org) | [torchmetrics](https://torchmetrics.readthedocs.io)

## License

MIT License — see [LICENSE](LICENSE).

# Databricks notebook source
# MAGIC %md
# MAGIC # 03. Model Evaluation
# MAGIC
# MAGIC Standalone evaluation of a trained detection model: mAP metrics, error
# MAGIC analysis, latency benchmarks, and result visualisation.
# MAGIC
# MAGIC Uses `EvaluationEngine` from `src/evaluation/`.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Environment
# MAGIC
# MAGIC Run from a Git folder clone of this repo, attached to **AI Runtime**
# MAGIC (serverless GPU) with the **AI v6** base environment, which already ships
# MAGIC torch, transformers v5 and MLflow. This installs the few packages it lacks.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements_runtime.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import sys, os
from pathlib import Path

# The notebook runs from notebooks/ in the Git folder; only the repo root goes
# on the path, so `src` imports resolve the same way the job entry points do.
REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config.schema import load_config
from src.evaluation import EvaluationEngine

# Pipeline config: a Volume path, or a path relative to the repo root.
dbutils.widgets.text("config_path", "", "Config YAML")
CONFIG_PATH = dbutils.widgets.get("config_path")
if not CONFIG_PATH:
    raise ValueError("Set the config_path widget to your pipeline config YAML.")
if not os.path.isabs(CONFIG_PATH):
    CONFIG_PATH = os.path.join(REPO_ROOT, CONFIG_PATH)

config = load_config(CONFIG_PATH)

# Model source: a local checkpoint, an MLflow run, or — by default — the last
# training run, read from the run manifest in output.results_dir.
dbutils.widgets.text("run_id", "", "MLflow run ID (default: last training run)")
dbutils.widgets.text("checkpoint_path", "", "Checkpoint directory (overrides run)")
CHECKPOINT_PATH = dbutils.widgets.get("checkpoint_path") or None
RUN_ID = dbutils.widgets.get("run_id") or None
MODEL_URI = None
if not (CHECKPOINT_PATH or RUN_ID):
    from src.utils.manifest import read_run_manifest

    manifest = read_run_manifest(config.output.results_dir)
    RUN_ID, MODEL_URI = manifest["run_id"], manifest.get("model_uri")
if CHECKPOINT_PATH:
    RUN_ID = MODEL_URI = None

engine = EvaluationEngine(config)

print(f"Model source: {CHECKPOINT_PATH or MODEL_URI or RUN_ID}")
print(f"Model:       {config.model.model_name}")
print(f"Val data:    {config.data.val_data_path}")
print(f"Results dir: {config.output.results_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. mAP Metrics

# COMMAND ----------

metrics = engine.evaluate(
    model_path=CHECKPOINT_PATH,
    run_id=RUN_ID,
    model_uri=MODEL_URI,
)

import pandas as pd

metrics_df = pd.DataFrame(
    [(k, f"{v:.4f}" if isinstance(v, float) else str(v)) for k, v in sorted(metrics.items())],
    columns=["Metric", "Value"],
)
display(metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Per-Class AP Breakdown

# COMMAND ----------

import matplotlib.pyplot as plt

per_class = {k: v for k, v in metrics.items() if "map_class_" in k}

if per_class:
    class_ids = [k.split("_")[-1] for k in sorted(per_class.keys())]
    values = [per_class[k] for k in sorted(per_class.keys())]

    fig, ax = plt.subplots(figsize=(14, max(6, len(class_ids) * 0.3)))
    ax.barh(class_ids[::-1], values[::-1])
    ax.set_xlabel("AP")
    ax.set_title("Per-Class Average Precision")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.show()
else:
    print("Per-class metrics not available (need >1 class in validation set).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Error Analysis

# COMMAND ----------

errors = engine.error_analysis(
    model_path=CHECKPOINT_PATH,
    run_id=RUN_ID,
    model_uri=MODEL_URI,
    max_batches=100,
)

summary = errors["summary"]
print("Error Analysis Summary")
print("=" * 40)
for k, v in summary.items():
    print(f"  {k}: {v}")

# Breakdown chart
labels = ["True Positives", "FP (Background)", "FP (Confusion)", "FP (Localisation)", "False Negatives"]
values = [
    summary["true_positives"],
    summary["false_positives_background"],
    summary["false_positives_confusion"],
    summary["false_positives_localisation"],
    summary["false_negatives"],
]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2ecc71", "#e74c3c", "#e67e22", "#f39c12", "#3498db"]
ax.bar(labels, values, color=colors)
ax.set_ylabel("Count")
ax.set_title("Prediction Error Breakdown")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# Precision / recall
tp = summary["true_positives"]
total_pred = summary["total_predictions"]
total_gt = summary["total_ground_truths"]
precision = tp / max(total_pred, 1)
recall = tp / max(total_gt, 1)
print(f"\nPrecision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Latency Benchmark

# COMMAND ----------

bench = engine.benchmark(
    model_path=CHECKPOINT_PATH,
    run_id=RUN_ID,
    model_uri=MODEL_URI,
    num_warmup=10,
    num_batches=100,
)

print("Benchmark Results")
print("=" * 40)
print(f"  FPS:              {bench['fps']:.1f}")
print(f"  Total images:     {bench['total_images']}")
print(f"  Total time (s):   {bench['total_time_s']:.2f}")
print(f"  Latency mean (ms):{bench['latency_per_batch_ms']['mean']:.1f}")
print(f"  Latency p50 (ms): {bench['latency_per_batch_ms']['p50']:.1f}")
print(f"  Latency p95 (ms): {bench['latency_per_batch_ms']['p95']:.1f}")
print(f"  Latency p99 (ms): {bench['latency_per_batch_ms']['p99']:.1f}")
print(f"  Device:           {bench['device']}")
if "gpu_memory_mb" in bench:
    print(f"  GPU Memory (MB):  {bench['gpu_memory_mb']:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results
# MAGIC
# MAGIC `EvaluationEngine` writes every result above as JSON to `config.output.results_dir`.
# MAGIC Make that a Volume path so results outlive the serverless session.

# COMMAND ----------

results_dir = config.output.results_dir
for fname in ["evaluation_metrics.json", "error_analysis.json", "benchmark.json"]:
    path = os.path.join(results_dir, fname)
    print(f"{'✅' if os.path.exists(path) else '⚠️ missing'}  {path}")

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
# MAGIC ## 1. Configuration

# COMMAND ----------

import sys, os
from pathlib import Path

sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref')

from src.config.schema import load_config
from src.evaluation import EvaluationEngine

# --- Paths (customise for your workspace) ---
CATALOG = "your_catalog"
SCHEMA = "your_schema"
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_yolos_config.yaml"

config = load_config(CONFIG_PATH)

# Choose ONE model source — MLflow run_id OR local checkpoint path
RUN_ID = None  # e.g., "abc123def456"
CHECKPOINT_PATH = None  # e.g., "/Volumes/.../checkpoints/best_model"

engine = EvaluationEngine(config)

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
# MAGIC ## 6. Save Results to Volume
# MAGIC
# MAGIC All results are automatically saved to `config.output.results_dir` as JSON.
# MAGIC Copy to a Volume for persistence if running on ephemeral compute.

# COMMAND ----------

import shutil

results_dir = config.output.results_dir
volume_results = f"{BASE_VOLUME_PATH}/results/evaluation"
os.makedirs(volume_results, exist_ok=True)

for fname in ["evaluation_metrics.json", "error_analysis.json", "benchmark.json"]:
    src = os.path.join(results_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(volume_results, fname))
        print(f"Copied {fname} → {volume_results}/")

print(f"\nResults saved to: {volume_results}")

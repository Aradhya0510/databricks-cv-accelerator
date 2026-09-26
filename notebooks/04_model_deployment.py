# Databricks notebook source
# MAGIC %md
# MAGIC # 04. Model Deployment
# MAGIC
# MAGIC Register a trained model as a PyFunc to Unity Catalog and deploy to
# MAGIC Databricks Model Serving.
# MAGIC
# MAGIC Steps:
# MAGIC 1. Test PyFunc locally with a sample image
# MAGIC 2. Register to Unity Catalog via `register_model()`
# MAGIC 3. Deploy endpoint via `deploy_endpoint()`
# MAGIC 4. Smoke test the live endpoint
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

import sys, os, base64
from pathlib import Path

# The notebook runs from notebooks/ in the Git folder; only the repo root goes
# on the path, so `src` imports resolve the same way the job entry points do.
REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config.schema import load_config

# Pipeline config: a Volume path, or a path relative to the repo root.
dbutils.widgets.text("config_path", "", "Config YAML")
CONFIG_PATH = dbutils.widgets.get("config_path")
if not CONFIG_PATH:
    raise ValueError("Set the config_path widget to your pipeline config YAML.")
if not os.path.isabs(CONFIG_PATH):
    CONFIG_PATH = os.path.join(REPO_ROOT, CONFIG_PATH)

config = load_config(CONFIG_PATH)

# --- Deployment settings (widgets override the config's serving section) ---
dbutils.widgets.text("run_id", "", "MLflow run ID (default: last training run)")
dbutils.widgets.text("model_name", "", "UC model (default: serving.registered_model_name)")
dbutils.widgets.text("endpoint_name", "", "Endpoint (default: serving.endpoint_name)")
dbutils.widgets.text("test_image", "", "Test image (default: first val image)")

RUN_ID = dbutils.widgets.get("run_id") or None
MODEL_URI = None
if not RUN_ID:
    from src.utils.manifest import read_run_manifest

    manifest = read_run_manifest(config.output.results_dir)
    RUN_ID, MODEL_URI = manifest["run_id"], manifest.get("model_uri")

REGISTERED_MODEL_NAME = dbutils.widgets.get("model_name") or config.serving.registered_model_name
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name") or config.serving.endpoint_name
if not (REGISTERED_MODEL_NAME and ENDPOINT_NAME):
    raise ValueError("Set serving.registered_model_name and serving.endpoint_name, or the widgets.")

TEST_IMAGE_PATH = dbutils.widgets.get("test_image") or os.path.join(
    config.data.val_data_path,
    sorted(f for f in os.listdir(config.data.val_data_path) if f.lower().endswith((".jpg", ".jpeg", ".png")))[0],
)

print(f"Model:     {config.model.model_name}")
print(f"Run ID:    {RUN_ID}")
print(f"Image:     {TEST_IMAGE_PATH}")
print(f"UC Model:  {REGISTERED_MODEL_NAME}")
print(f"Endpoint:  {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test PyFunc Locally

# COMMAND ----------

from src.serving.artifacts import resolve_model_dir
from src.serving.pyfunc import DetectionPyFuncModel
import mlflow

# The same flat model + processor directory registration will package.
model_dir = resolve_model_dir(run_id=RUN_ID, model_uri=MODEL_URI)

# Simulate PyFunc load_context
pyfunc = DetectionPyFuncModel()

class _MockContext:
    def __init__(self, model_dir):
        self.artifacts = {"model_dir": model_dir}

pyfunc.load_context(_MockContext(model_dir))

# Test with a real image
with open(TEST_IMAGE_PATH, "rb") as f:
    b64_image = base64.b64encode(f.read()).decode()

import pandas as pd
test_input = pd.DataFrame([{"image": b64_image}])
results = pyfunc.predict(None, test_input)

print(f"Status:         {results[0]['predictions']['status']}")
print(f"Num detections: {results[0]['predictions']['num_detections']}")
print(f"Sample boxes:   {results[0]['predictions']['boxes'][:3]}")
print(f"Sample scores:  {results[0]['predictions']['scores'][:3]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Register Model to Unity Catalog

# COMMAND ----------

from src.serving.registration import register_model

reg_result = register_model(
    run_id=RUN_ID,
    model_uri=MODEL_URI,
    registered_model_name=REGISTERED_MODEL_NAME,
    task_type=config.model.task_type,
    # Not "latest": Unity Catalog reserves it and registration fails after the
    # version is created.
    aliases=["champion"],
    tags={
        "framework": "hf_trainer",
        "task": "detection",
        "model_arch": config.model.model_name,
    },
    validate=True,
    test_image_path=TEST_IMAGE_PATH,
)

print(f"Registered: {reg_result['registered_model_name']} v{reg_result['model_version']}")
print(f"Model URI:  {reg_result['model_uri']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Set Aliases

# COMMAND ----------

client = mlflow.MlflowClient()

# List current aliases
model_info = client.get_registered_model(REGISTERED_MODEL_NAME)
for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"):
    print(f"  Version {mv.version}: aliases={mv.aliases}, tags={mv.tags}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy to Model Serving

# COMMAND ----------

from src.serving.deployment import deploy_endpoint, wait_for_ready

deploy_result = deploy_endpoint(
    endpoint_name=ENDPOINT_NAME,
    registered_model_name=REGISTERED_MODEL_NAME,
    model_version=str(reg_result["model_version"]),
    workload_size=config.serving.workload_size,
    scale_to_zero=config.serving.scale_to_zero,
)

print(f"Endpoint: {deploy_result['endpoint_name']} — {deploy_result['status']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Wait for Endpoint READY

# COMMAND ----------

ready_result = wait_for_ready(ENDPOINT_NAME, timeout=1800, poll_interval=30)
print(f"Endpoint state: {ready_result['state']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Endpoint

# COMMAND ----------

from src.serving.deployment import test_endpoint

# Single image test
test_result = test_endpoint(
    endpoint_name=ENDPOINT_NAME,
    test_image_path=TEST_IMAGE_PATH,
)
print(f"Endpoint test: {test_result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Endpoint Performance Benchmark

# COMMAND ----------

import time, requests, json
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Build request
with open(TEST_IMAGE_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {"dataframe_records": [{"image": b64}]}

# Warm up
for _ in range(3):
    w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=payload["dataframe_records"])

# Timed requests
latencies = []
for _ in range(20):
    t0 = time.perf_counter()
    w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=payload["dataframe_records"])
    latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
print(f"Endpoint Latency (ms):")
print(f"  Mean: {sum(latencies)/len(latencies):.0f}")
print(f"  P50:  {latencies[len(latencies)//2]:.0f}")
print(f"  P95:  {latencies[int(len(latencies)*0.95)]:.0f}")
print(f"  P99:  {latencies[int(len(latencies)*0.99)]:.0f}")

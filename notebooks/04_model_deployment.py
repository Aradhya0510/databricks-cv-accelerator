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
# MAGIC ## 1. Configuration

# COMMAND ----------

import sys, os, base64
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

# --- Deployment settings ---
RUN_ID = "YOUR_MLFLOW_RUN_ID"  # From training notebook / MLflow UI
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.yolos_detection"
ENDPOINT_NAME = "yolos-detection-endpoint"

# Path to a test image for validation
TEST_IMAGE_PATH = f"{BASE_VOLUME_PATH}/data/val2017/000000000139.jpg"  # Example

print(f"Model:     {config.model.model_name}")
print(f"Run ID:    {RUN_ID}")
print(f"UC Model:  {REGISTERED_MODEL_NAME}")
print(f"Endpoint:  {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test PyFunc Locally

# COMMAND ----------

from src.serving.pyfunc import DetectionPyFuncModel
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image
import mlflow
import tempfile

# Download model from MLflow and save locally
artifact_path = mlflow.artifacts.download_artifacts(run_id=RUN_ID, artifact_path="model")

tmpdir = tempfile.mkdtemp()
model = AutoModelForObjectDetection.from_pretrained(artifact_path)
processor = AutoImageProcessor.from_pretrained(artifact_path)
model.save_pretrained(tmpdir)
processor.save_pretrained(tmpdir)

# Simulate PyFunc load_context
pyfunc = DetectionPyFuncModel()

class _MockContext:
    def __init__(self, model_dir):
        self.artifacts = {"model_dir": model_dir}

pyfunc.load_context(_MockContext(tmpdir))

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
    registered_model_name=REGISTERED_MODEL_NAME,
    aliases=["champion", "latest"],
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

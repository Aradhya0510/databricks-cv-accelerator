# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Monitoring and Drift Detection
# MAGIC 
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Enable model monitoring for a Databricks Model Serving endpoint
# MAGIC 2. Log prediction and ground truth events for monitoring
# MAGIC 3. Analyze drift metrics and endpoint health
# MAGIC 4. Set up alerting for model performance and data drift
# MAGIC 5. Query logs and metrics using Databricks SQL
# MAGIC 6. Integrate with external alerting systems
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **Model Monitoring Goals:**
# MAGIC - Detect data and prediction drift
# MAGIC - Monitor endpoint health and latency
# MAGIC - Track model performance over time
# MAGIC - Enable alerting for anomalies and failures
# MAGIC 
# MAGIC ---

# COMMAND ----------

# (Databricks only) Install requirements and restart Python if running interactively
# %pip install -r "../requirements.txt"
# dbutils.library.restartPython()

# (Databricks only) Run setup and previous notebooks if running interactively
# %run ./00_setup_and_config
# %run ./05_model_registration_deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Enable Model Monitoring on Endpoint
# MAGIC 
# MAGIC Databricks Model Serving endpoints support built-in monitoring for:
# MAGIC - Input data drift
# MAGIC - Prediction drift
# MAGIC - Latency and error rates
# MAGIC - Custom metrics
# MAGIC 
# MAGIC Monitoring is enabled via the Databricks SDK or UI. Below is an example using the SDK.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import MonitoringConfig

# Set your endpoint name (should match the one created in deployment)
ENDPOINT_NAME = "detr-endpoint"  # Update if your endpoint name is different

client = WorkspaceClient()

# Enable monitoring (if not already enabled)
monitoring_config = MonitoringConfig(
    enabled=True,
    request_logging_enabled=True,
    response_logging_enabled=True,
    drift_threshold=0.1  # Example: set drift threshold for alerts
)

try:
    client.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        monitoring=monitoring_config
    )
    print(f"✅ Monitoring enabled for endpoint: {ENDPOINT_NAME}")
except Exception as e:
    print(f"⚠️  Could not enable monitoring (may already be enabled): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Log Prediction and Ground Truth Events
# MAGIC 
# MAGIC To enable drift and performance monitoring, log both prediction requests and ground truth labels (when available).
# MAGIC 
# MAGIC - **Prediction logging** is automatic if enabled above.
# MAGIC - **Ground truth logging** must be done via the Model Monitoring API.
# MAGIC 
# MAGIC Example: Log ground truth for a batch of predictions.

# COMMAND ----------

import requests
import os
import json
from datetime import datetime

# Get endpoint URL (from deployment notebook or UI)
ENDPOINT_URL = f"https://<your-workspace-url>/serving-endpoints/{ENDPOINT_NAME}/invocations"  # Update as needed
MONITORING_API_URL = f"https://<your-workspace-url>/api/2.0/serving-endpoints/{ENDPOINT_NAME}/monitoring/ground-truth"  # Update as needed
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "<your-databricks-token>")

# Example: Log ground truth for a batch
headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

ground_truth_payload = {
    "request_ids": ["req-123", "req-124"],  # Use actual request IDs from prediction logs
    "ground_truth": [
        {"label": 1},
        {"label": 0}
    ]
}

response = requests.post(MONITORING_API_URL, headers=headers, data=json.dumps(ground_truth_payload))
if response.status_code == 200:
    print("✅ Ground truth logged successfully")
else:
    print(f"❌ Failed to log ground truth: {response.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Analyze Drift Metrics and Endpoint Health
# MAGIC 
# MAGIC Databricks automatically computes drift and health metrics for monitored endpoints. You can view these in the UI or query them programmatically.
# MAGIC 
# MAGIC Example: Query endpoint metrics using the Databricks SDK.

# COMMAND ----------

# Get endpoint metrics
endpoint = client.serving_endpoints.get(ENDPOINT_NAME)

print(f"Endpoint state: {endpoint.state.ready}")
print(f"Health: {endpoint.state.health}")

# Example: Print drift metrics (if available)
if hasattr(endpoint.state, "monitoring_metrics"):
    print("Drift metrics:")
    print(endpoint.state.monitoring_metrics)
else:
    print("No drift metrics available yet (may require more data)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Query Logs and Metrics with Databricks SQL
# MAGIC 
# MAGIC All model serving logs and metrics are available in Unity Catalog tables. You can query them using Databricks SQL.
# MAGIC 
# MAGIC Example queries:
# MAGIC - **Request/response logs**: `serving_requests` table
# MAGIC - **Drift metrics**: `serving_drift_metrics` table
# MAGIC - **Errors and latency**: `serving_endpoint_metrics` table
# MAGIC 
# MAGIC Replace `<catalog>`, `<schema>`, and `<endpoint_name>` as appropriate.

# COMMAND ----------

# Example: Query recent requests
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM <catalog>.<schema>.serving_requests
# MAGIC WHERE endpoint_name = '<endpoint_name>'
# MAGIC   AND request_timestamp > now() - INTERVAL 1 DAY
# MAGIC ORDER BY request_timestamp DESC
# MAGIC LIMIT 100;

# Example: Query drift metrics
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM <catalog>.<schema>.serving_drift_metrics
# MAGIC WHERE endpoint_name = '<endpoint_name>'
# MAGIC   AND timestamp > now() - INTERVAL 7 DAY
# MAGIC ORDER BY timestamp DESC;

# Example: Query error rates and latency
# MAGIC %sql
# MAGIC SELECT
# MAGIC   date_trunc('hour', request_timestamp) AS hour,
# MAGIC   COUNT(*) AS total_requests,
# MAGIC   SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS errors,
# MAGIC   AVG(latency_ms) AS avg_latency
# MAGIC FROM <catalog>.<schema>.serving_requests
# MAGIC WHERE endpoint_name = '<endpoint_name>'
# MAGIC   AND request_timestamp > now() - INTERVAL 1 DAY
# MAGIC GROUP BY hour
# MAGIC ORDER BY hour DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Set Up Alerting for Drift and Failures
# MAGIC 
# MAGIC You can set up alerts in Databricks SQL or integrate with external systems (PagerDuty, Slack, email, etc.).
# MAGIC 
# MAGIC Example: Create a Databricks SQL alert for high drift or error rate.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Example: Alert if drift exceeds threshold
# MAGIC SELECT
# MAGIC   timestamp,
# MAGIC   drift_score
# MAGIC FROM <catalog>.<schema>.serving_drift_metrics
# MAGIC WHERE endpoint_name = '<endpoint_name>'
# MAGIC   AND drift_score > 0.2
# MAGIC ORDER BY timestamp DESC;

# MAGIC %sql
# MAGIC -- Example: Alert if error rate exceeds threshold
# MAGIC SELECT
# MAGIC   hour,
# MAGIC   errors,
# MAGIC   total_requests,
# MAGIC   (errors * 1.0 / total_requests) AS error_rate
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     date_trunc('hour', request_timestamp) AS hour,
# MAGIC     COUNT(*) AS total_requests,
# MAGIC     SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS errors
# MAGIC   FROM <catalog>.<schema>.serving_requests
# MAGIC   WHERE endpoint_name = '<endpoint_name>'
# MAGIC     AND request_timestamp > now() - INTERVAL 1 DAY
# MAGIC   GROUP BY hour
# MAGIC )
# MAGIC WHERE error_rate > 0.05
# MAGIC ORDER BY hour DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Integrate with External Alerting Systems
# MAGIC 
# MAGIC Databricks SQL alerts can be configured to send notifications to email, Slack, PagerDuty, or webhooks.
# MAGIC 
# MAGIC - In the Databricks SQL UI, create an alert from a query and configure the notification destination.
# MAGIC - For advanced integrations, use the Databricks REST API to trigger external systems.
# MAGIC 
# MAGIC Example: Use a webhook to trigger an external alert.

# COMMAND ----------

# Example: Send alert to a webhook (Python)
import requests

WEBHOOK_URL = "https://hooks.slack.com/services/your/webhook/url"  # Replace with your webhook
alert_message = {
    "text": "🚨 Model drift detected on endpoint detr-endpoint! Drift score exceeded threshold."
}

response = requests.post(WEBHOOK_URL, json=alert_message)
if response.status_code == 200:
    print("✅ Alert sent to webhook")
else:
    print(f"❌ Failed to send alert: {response.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Best Practices for Production Monitoring
# MAGIC 
# MAGIC - Enable monitoring and logging for all production endpoints
# MAGIC - Regularly review drift and error metrics
# MAGIC - Set up alerts for drift, latency, and failures
# MAGIC - Log ground truth labels for performance tracking
# MAGIC - Periodically retrain and redeploy models as needed
# MAGIC - Document monitoring and alerting procedures
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC For more details, see the [Databricks Model Monitoring documentation](https://docs.databricks.com/en/machine-learning/model-serving/monitor-diagnose-endpoints.html) 
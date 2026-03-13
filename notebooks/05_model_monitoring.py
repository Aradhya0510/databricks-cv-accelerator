# Databricks notebook source
# MAGIC %md
# MAGIC # 05. Model Monitoring
# MAGIC
# MAGIC Observability for a deployed Model Serving endpoint: health checks,
# MAGIC request metrics, prediction distributions, and alert guidance.
# MAGIC
# MAGIC Uses `EndpointMonitor` from `src/monitoring/`.
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

from src.monitoring import EndpointMonitor

ENDPOINT_NAME = "yolos-detection-endpoint"  # Your endpoint name
LOOKBACK_HOURS = 24

monitor = EndpointMonitor(ENDPOINT_NAME)
print(f"Monitoring endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Health Check

# COMMAND ----------

health = monitor.get_health()

print(f"Endpoint:  {health['endpoint_name']}")
print(f"Ready:     {health['ready']}")
print(f"Checked:   {health['checked_at']}")

if health.get("served_models"):
    print("\nServed Models:")
    for m in health["served_models"]:
        print(f"  - {m['entity_name']} v{m['entity_version']}")
        print(f"    Size: {m['workload_size']}, Scale-to-zero: {m['scale_to_zero']}")
else:
    print("\n⚠️ No served models found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Request Metrics

# COMMAND ----------

req = monitor.get_request_metrics(hours=LOOKBACK_HOURS)

if "error" not in req:
    print(f"Request Metrics (last {LOOKBACK_HOURS}h)")
    print("=" * 40)
    print(f"  Total requests: {req.get('total_requests', 0):,}")
    print(f"  Error count:    {req.get('error_count', 0):,}")
    print(f"  Error rate:     {req.get('error_rate', 0):.2%}")
    print(f"  Avg latency:    {req.get('avg_latency_ms', 0):.0f} ms")
    print(f"  P50 latency:    {req.get('p50_latency_ms', 0):.0f} ms")
    print(f"  P95 latency:    {req.get('p95_latency_ms', 0):.0f} ms")
    print(f"  P99 latency:    {req.get('p99_latency_ms', 0):.0f} ms")
else:
    print(f"⚠️ Could not query request metrics: {req['error']}")
    print("Ensure system.serving.served_model_requests is accessible.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prediction Distribution

# COMMAND ----------

import matplotlib.pyplot as plt

pred_dist = monitor.get_prediction_distribution(hours=LOOKBACK_HOURS)

if "error" not in pred_dist:
    print(f"Prediction Distribution (last {LOOKBACK_HOURS}h)")
    print(f"  Responses sampled: {pred_dist.get('num_responses_sampled', 0)}")

    conf = pred_dist.get("confidence_stats", {})
    print(f"  Avg confidence:    {conf.get('mean', 0):.3f}")
    print(f"  Min confidence:    {conf.get('min', 0):.3f}")
    print(f"  Max confidence:    {conf.get('max', 0):.3f}")

    class_dist = pred_dist.get("class_distribution", {})
    if class_dist:
        sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
        labels = [str(c[0]) for c in sorted_classes[:20]]
        counts = [c[1] for c in sorted_classes[:20]]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(labels, counts)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Count")
        ax.set_title(f"Predicted Class Distribution (last {LOOKBACK_HOURS}h)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("  No class distribution data available.")
else:
    print(f"⚠️ Could not query prediction distribution: {pred_dist['error']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate Monitoring Report

# COMMAND ----------

report = monitor.generate_report(
    output_path=f"/tmp/monitoring_report_{ENDPOINT_NAME}.json"
)

print("Report sections:")
for key in report:
    print(f"  - {key}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Alert Configuration Guidance
# MAGIC
# MAGIC Set up SQL alerts on Databricks system tables for proactive monitoring.
# MAGIC
# MAGIC ### Error Rate Alert
# MAGIC ```sql
# MAGIC -- Create a SQL alert that fires when error rate exceeds 5%
# MAGIC SELECT
# MAGIC     COUNT(*) AS total,
# MAGIC     SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS errors,
# MAGIC     SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS error_rate_pct
# MAGIC FROM system.serving.served_model_requests
# MAGIC WHERE served_entity_name = 'yolos-detection-endpoint'
# MAGIC   AND request_time >= CURRENT_TIMESTAMP - INTERVAL 1 HOUR
# MAGIC HAVING error_rate_pct > 5
# MAGIC ```
# MAGIC
# MAGIC ### Latency Alert
# MAGIC ```sql
# MAGIC -- Alert when P95 latency exceeds 500ms
# MAGIC SELECT
# MAGIC     PERCENTILE(execution_time_ms, 0.95) AS p95_latency_ms
# MAGIC FROM system.serving.served_model_requests
# MAGIC WHERE served_entity_name = 'yolos-detection-endpoint'
# MAGIC   AND request_time >= CURRENT_TIMESTAMP - INTERVAL 1 HOUR
# MAGIC HAVING p95_latency_ms > 500
# MAGIC ```
# MAGIC
# MAGIC ### Volume Drop Alert
# MAGIC ```sql
# MAGIC -- Alert when request volume drops below expected baseline
# MAGIC SELECT
# MAGIC     COUNT(*) AS request_count
# MAGIC FROM system.serving.served_model_requests
# MAGIC WHERE served_entity_name = 'yolos-detection-endpoint'
# MAGIC   AND request_time >= CURRENT_TIMESTAMP - INTERVAL 1 HOUR
# MAGIC HAVING request_count < 10  -- Adjust threshold based on expected traffic
# MAGIC ```

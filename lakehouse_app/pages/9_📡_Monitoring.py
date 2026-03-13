"""
Page 9: Model Monitoring
Display endpoint health, request metrics, and prediction distributions.
"""

import streamlit as st
import json
from datetime import datetime

from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from components.visualizations import VisualizationHelper

# Initialize state
StateManager.initialize()

# Page config
st.title("📡 Model Monitoring")
st.markdown("Monitor deployed model endpoints: health, metrics, and predictions")

# Initialize client
client = DatabricksJobClient()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏥 Health Check", "📊 Request Metrics", "📈 Prediction Distribution", "📋 Reports"])

with tab1:
    st.markdown("### Endpoint Health")

    endpoints = StateManager.get("endpoints", [])
    endpoint_names = [e["endpoint_name"] for e in endpoints]

    if not endpoint_names:
        st.info("No endpoints tracked. Deploy a model first.")
    else:
        selected_endpoint = st.selectbox("Select Endpoint", endpoint_names, key="health_endpoint")

        if st.button("🔍 Check Health", type="primary"):
            with st.spinner("Checking endpoint health..."):
                try:
                    status = client.get_endpoint_status(selected_endpoint)
                    state = status.get("state", "UNKNOWN")
                    ready = status.get("ready", "UNKNOWN")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if state == "NOT_UPDATING" and ready == "READY":
                            st.success("READY")
                        elif state == "UPDATING":
                            st.warning("UPDATING")
                        else:
                            st.error(state)
                    with col2:
                        st.metric("State", state)
                    with col3:
                        st.metric("Ready", ready)

                    if status.get("served_models"):
                        st.markdown("#### Served Models")
                        for m in status["served_models"]:
                            st.markdown(f"- **{m.get('entity_name', 'N/A')}** v{m.get('entity_version', '?')}")

                except Exception as e:
                    st.error(f"Error checking health: {e}")

with tab2:
    st.markdown("### Request Metrics")

    if not endpoint_names:
        st.info("No endpoints tracked.")
    else:
        selected_endpoint = st.selectbox("Select Endpoint", endpoint_names, key="metrics_endpoint")
        hours = st.slider("Lookback (hours)", 1, 168, 24)

        if st.button("📊 Load Metrics", type="primary"):
            with st.spinner("Querying system tables..."):
                st.info(f"Querying `system.serving.served_model_requests` for last {hours}h")

                st.markdown("#### Metrics Summary")
                st.markdown("""
                Metrics are queried from `system.serving.served_model_requests`.
                Run `jobs/monitor.py` to generate a full report, then view results here.
                """)

                # Check for pre-computed report
                report_path = f"/tmp/monitoring/monitoring_report_{selected_endpoint}.json"
                try:
                    with open(report_path) as f:
                        report = json.load(f)

                    req = report.get("request_metrics_24h", {})
                    if "error" not in req:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Requests", f"{req.get('total_requests', 0):,}")
                        with col2:
                            st.metric("Error Rate", f"{req.get('error_rate', 0):.2%}")
                        with col3:
                            st.metric("Avg Latency", f"{req.get('avg_latency_ms', 0):.0f} ms")
                        with col4:
                            st.metric("P95 Latency", f"{req.get('p95_latency_ms', 0):.0f} ms")
                    else:
                        st.warning(f"Report error: {req.get('error')}")

                except FileNotFoundError:
                    st.info("No pre-computed report found. Run `jobs/monitor.py` first.")
                    st.code(
                        f"python jobs/monitor.py --endpoint_name {selected_endpoint} --hours {hours}",
                        language="bash",
                    )

with tab3:
    st.markdown("### Prediction Distribution")

    if not endpoint_names:
        st.info("No endpoints tracked.")
    else:
        selected_endpoint = st.selectbox("Select Endpoint", endpoint_names, key="dist_endpoint")

        if st.button("📈 Load Distribution", type="primary"):
            report_path = f"/tmp/monitoring/monitoring_report_{selected_endpoint}.json"
            try:
                with open(report_path) as f:
                    report = json.load(f)

                pred_dist = report.get("prediction_distribution_24h", {})
                class_dist = pred_dist.get("class_distribution", {})

                if class_dist:
                    fig = VisualizationHelper.class_distribution_chart(
                        {str(k): v for k, v in class_dist.items()},
                        "Predicted Class Distribution (Last 24h)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                conf = pred_dist.get("confidence_stats", {})
                if conf:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Confidence", f"{conf.get('mean', 0):.3f}")
                    with col2:
                        st.metric("Min Confidence", f"{conf.get('min', 0):.3f}")
                    with col3:
                        st.metric("Max Confidence", f"{conf.get('max', 0):.3f}")
            except FileNotFoundError:
                st.info("No pre-computed report found. Run `jobs/monitor.py` first.")

with tab4:
    st.markdown("### Monitoring Reports")

    if not endpoint_names:
        st.info("No endpoints tracked.")
    else:
        selected_endpoint = st.selectbox("Select Endpoint", endpoint_names, key="report_endpoint")

        st.markdown("#### Generate Report via Job")
        st.code(
            f"python jobs/monitor.py --endpoint_name {selected_endpoint} --hours 24",
            language="bash",
        )

        st.markdown("#### Scheduled Monitoring")
        st.info("""
        For production monitoring, schedule `jobs/monitor.py` as a daily Databricks Job:
        1. Create a job with `jobs/monitor.py` as the task
        2. Set schedule to run daily
        3. Configure alerts on the monitoring report thresholds
        """)

        st.markdown("#### SQL Alert Examples")
        st.code(f"""
-- Error rate alert (fires when > 5%)
SELECT
    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS error_rate
FROM system.serving.served_model_requests
WHERE served_entity_name = '{selected_endpoint}'
  AND request_time >= CURRENT_TIMESTAMP - INTERVAL 1 HOUR
HAVING error_rate > 5
        """, language="sql")

# Sidebar
with st.sidebar:
    st.markdown("### 📡 Monitoring")

    endpoints = StateManager.get("endpoints", [])
    st.metric("Tracked Endpoints", len(endpoints))

    st.markdown("---")
    st.markdown("### Quick Actions")

    if st.button("🌐 Go to Deployment", use_container_width=True):
        st.switch_page("pages/6_🌐_Deployment.py")

    if st.button("🎮 Test Endpoint", use_container_width=True):
        st.switch_page("pages/7_🎮_Inference.py")

"""
Page 8: History & Management
Track and manage all pipeline activities
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd

from utils.state_manager import StateManager
from components.visualizations import VisualizationHelper
from components.theme import inject_theme, page_header, section_title, metric_card, status_badge

inject_theme()
StateManager.initialize()

page_header("History & Management", "Track and manage your ML pipeline activities")

tab1, tab2, tab3, tab4 = st.tabs(["Activity Log", "Analytics", "Settings", "Export / Import"])

with tab1:
    section_title("Recent Activity")

    col1, col2, col3 = st.columns(3)
    with col1:
        activity_type = st.selectbox("Activity Type", options=["All", "Training", "Evaluation", "Deployment", "Inference"])
    with col2:
        date_range = st.selectbox("Time Range", options=["Last 24 hours", "Last 7 days", "Last 30 days", "All time"])
    with col3:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    training_history = StateManager.get("training_history", [])
    registered_models = StateManager.get("registered_models", [])
    endpoints = StateManager.get("endpoints", [])
    recent_configs = StateManager.get("recent_configs", [])

    activities = []
    for run in training_history:
        activities.append({
            "type": "Training",
            "description": f"Training run: {run.get('job_name', 'Unnamed')}",
            "status": run.get("status", "UNKNOWN"),
            "timestamp": run.get("timestamp", ""),
            "details": run,
        })
    for model in registered_models:
        activities.append({
            "type": "Registration",
            "description": f"Registered model: {model.get('name', 'Unnamed')}",
            "status": "SUCCESS",
            "timestamp": model.get("creation_timestamp", ""),
            "details": model,
        })
    for endpoint in endpoints:
        activities.append({
            "type": "Deployment",
            "description": f"Deployed endpoint: {endpoint.get('endpoint_name', 'Unnamed')}",
            "status": "SUCCESS",
            "timestamp": endpoint.get("created_at", ""),
            "details": endpoint,
        })
    for cfg_path in recent_configs:
        activities.append({
            "type": "Configuration",
            "description": f"Config: {Path(cfg_path).name}",
            "status": "SUCCESS",
            "timestamp": datetime.now().isoformat(),
            "details": {"path": cfg_path},
        })

    activities.sort(key=lambda x: x["timestamp"], reverse=True)

    if not activities:
        st.info("No activities found. Start using the pipeline to see activities here.")
    else:
        if activity_type != "All":
            activities = [a for a in activities if a["type"] == activity_type]

        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#4E566A;margin-bottom:16px;">'
            f'{len(activities)} ACTIVITIES</div>',
            unsafe_allow_html=True,
        )

        for activity in activities:
            ts = activity.get("timestamp", "")[:16].replace("T", " ")
            badge = status_badge(activity["status"])
            with st.expander(f"{activity['type']} — {activity['description']} ({ts})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Type:** {activity['type']}")
                    st.markdown(f"{badge}", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**Time:** {ts}")

                if activity["type"] == "Training":
                    details = activity["details"]
                    st.markdown(f"**Task:** {details.get('task', 'N/A')}")
                    st.markdown(f"**Model:** {details.get('model', 'N/A')}")
                    st.markdown(f"**Run ID:** `{details.get('run_id', 'N/A')[:16]}…`")
                elif activity["type"] == "Registration":
                    details = activity["details"]
                    st.markdown(f"**Model:** `{details.get('name', 'N/A')}`")
                    st.markdown(f"**Version:** {details.get('version', 'N/A')}")
                elif activity["type"] == "Deployment":
                    details = activity["details"]
                    st.markdown(f"**Endpoint:** `{details.get('endpoint_name', 'N/A')}`")
                    st.markdown(f"**Size:** {details.get('workload_size', 'N/A')}")

with tab2:
    section_title("Pipeline Analytics")

    cols = st.columns(3)
    with cols[0]:
        metric_card("Training Runs", str(len(training_history)))
    with cols[1]:
        metric_card("Registered Models", str(len(registered_models)))
    with cols[2]:
        metric_card("Active Endpoints", str(len(endpoints)))

    if training_history:
        section_title("Training History")
        fig = VisualizationHelper.resource_usage_chart(training_history)
        st.plotly_chart(fig, use_container_width=True)

    section_title("Activity Distribution")
    activity_counts = {
        "Training": len(training_history),
        "Registration": len(registered_models),
        "Deployment": len(endpoints),
        "Configuration": len(recent_configs),
    }
    fig = VisualizationHelper.class_distribution_chart(activity_counts, "Activity by Type")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    section_title("Preferences")

    prefs = StateManager.get_user_preferences()

    col1, col2 = st.columns(2)
    with col1:
        default_catalog = st.text_input("Default Catalog", value=prefs.get("default_catalog", "main"))
    with col2:
        default_schema = st.text_input("Default Schema", value=prefs.get("default_schema", "cv_models"))

    default_volume = st.text_input("Default Volume", value=prefs.get("default_volume", "cv_data"))
    workspace_email = st.text_input(
        "Workspace Email", value=prefs.get("workspace_email", ""),
        placeholder="user@email.com", help="Your Databricks workspace email (for experiment paths)",
    )

    if st.button("Save Preferences", type="primary"):
        StateManager.set_user_preferences({
            "default_catalog": default_catalog,
            "default_schema": default_schema,
            "default_volume": default_volume,
            "workspace_email": workspace_email,
        })
        st.success("Preferences saved")

    section_title("Default Paths")
    default_paths = StateManager.get_default_paths()
    with st.expander("View Default Paths"):
        for key, path in default_paths.items():
            st.code(f"{key}: {path}")

    section_title("Danger Zone")
    with st.expander("Clear Data"):
        st.warning("These actions cannot be undone!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Training History", use_container_width=True):
                StateManager.set("training_history", [])
                st.success("Training history cleared")
                st.rerun()
        with col2:
            if st.button("Clear All Data", use_container_width=True):
                confirm = st.checkbox("I understand this will clear all data")
                if confirm:
                    StateManager.reset_state()
                    st.success("All data cleared")
                    st.rerun()

with tab4:
    section_title("Export Data")
    st.info("Export your pipeline history and configurations")

    export_options = st.multiselect(
        "Select data to export",
        options=["Training History", "Registered Models", "Endpoints", "Configurations", "User Preferences"],
        default=["Training History", "Configurations"],
    )

    if st.button("Export Data", type="primary"):
        import json

        export_data = {}
        if "Training History" in export_options:
            export_data["training_history"] = training_history
        if "Registered Models" in export_options:
            export_data["registered_models"] = registered_models
        if "Endpoints" in export_options:
            export_data["endpoints"] = endpoints
        if "Configurations" in export_options:
            export_data["recent_configs"] = recent_configs
        if "User Preferences" in export_options:
            export_data["preferences"] = StateManager.get_user_preferences()

        export_json = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="Download Export",
            data=export_json,
            file_name=f"pipeline_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    section_title("Import Data")
    st.info("Import previously exported data")

    uploaded_file = st.file_uploader("Upload export file", type=["json"], help="Upload a JSON export file")
    if uploaded_file:
        try:
            import json

            import_data = json.load(uploaded_file)
            st.success("File loaded successfully")

            with st.expander("Preview Import Data"):
                st.json(import_data)

            if st.button("Import Data", type="primary"):
                if "training_history" in import_data:
                    StateManager.set("training_history", import_data["training_history"])
                if "registered_models" in import_data:
                    StateManager.set("registered_models", import_data["registered_models"])
                if "endpoints" in import_data:
                    StateManager.set("endpoints", import_data["endpoints"])
                if "recent_configs" in import_data:
                    StateManager.set("recent_configs", import_data["recent_configs"])
                if "preferences" in import_data:
                    StateManager.set_user_preferences(import_data["preferences"])
                st.success("Data imported successfully")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Sidebar
with st.sidebar:
    st.markdown("### Quick Stats")
    total_activities = len(training_history) + len(registered_models) + len(endpoints) + len(recent_configs)
    st.metric("Total Activities", total_activities)
    if training_history:
        completed = len([r for r in training_history if r.get("status") == "SUCCESS"])
        st.metric("Completed Runs", completed)

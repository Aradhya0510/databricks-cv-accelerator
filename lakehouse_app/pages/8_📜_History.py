"""
Page 8: History & Management
Track and manage all pipeline activities
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import StateManager
from components.visualizations import VisualizationHelper

# Initialize state
StateManager.initialize()

# Page config
st.title("📜 History & Management")
st.markdown("Track and manage your ML pipeline activities")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔄 Activity Log", "📊 Analytics", "⚙️ Settings", "🗂️ Export/Import"])

with tab1:
    st.markdown("### Recent Activity")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        activity_type = st.selectbox(
            "Activity Type",
            options=["All", "Training", "Evaluation", "Deployment", "Inference"]
        )
    
    with col2:
        date_range = st.selectbox(
            "Time Range",
            options=["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
        )
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Aggregate all activities
    training_history = StateManager.get("training_history", [])
    registered_models = StateManager.get("registered_models", [])
    endpoints = StateManager.get("endpoints", [])
    recent_configs = StateManager.get("recent_configs", [])
    
    # Create activity timeline
    activities = []
    
    # Add training runs
    for run in training_history:
        activities.append({
            "type": "Training",
            "description": f"Training run: {run.get('job_name', 'Unnamed')}",
            "status": run.get("status", "UNKNOWN"),
            "timestamp": run.get("timestamp", ""),
            "details": run
        })
    
    # Add model registrations
    for model in registered_models:
        activities.append({
            "type": "Registration",
            "description": f"Registered model: {model.get('name', 'Unnamed')}",
            "status": "SUCCESS",
            "timestamp": model.get("creation_timestamp", ""),
            "details": model
        })
    
    # Add deployments
    for endpoint in endpoints:
        activities.append({
            "type": "Deployment",
            "description": f"Deployed endpoint: {endpoint.get('endpoint_name', 'Unnamed')}",
            "status": "SUCCESS",
            "timestamp": endpoint.get("created_at", ""),
            "details": endpoint
        })
    
    # Add configs
    for config in recent_configs:
        activities.append({
            "type": "Configuration",
            "description": f"Config: {Path(config).name}",
            "status": "SUCCESS",
            "timestamp": datetime.now().isoformat(),
            "details": {"path": config}
        })
    
    # Sort by timestamp
    activities.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Display activities
    if not activities:
        st.info("ℹ️ No activities found")
        st.markdown("Start using the pipeline to see activities here")
    else:
        st.success(f"✅ Found {len(activities)} activit(ies)")
        
        # Filter by type
        if activity_type != "All":
            activities = [a for a in activities if a["type"] == activity_type]
        
        # Display as timeline
        for activity in activities:
            with st.expander(
                f"{activity['type']} - {activity['description']} ({activity.get('timestamp', 'N/A')[:16]})",
                expanded=False
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Type:** {activity['type']}")
                    st.markdown(f"**Status:** {activity['status']}")
                
                with col2:
                    timestamp = activity.get('timestamp', 'N/A')
                    if timestamp != 'N/A':
                        st.markdown(f"**Time:** {timestamp[:19].replace('T', ' ')}")
                    else:
                        st.markdown("**Time:** N/A")
                
                # Show details based on type
                if activity["type"] == "Training":
                    details = activity["details"]
                    st.markdown(f"**Task:** {details.get('task', 'N/A')}")
                    st.markdown(f"**Model:** {details.get('model', 'N/A')}")
                    st.markdown(f"**Run ID:** `{details.get('run_id', 'N/A')[:16]}...`")
                
                elif activity["type"] == "Registration":
                    details = activity["details"]
                    st.markdown(f"**Model:** `{details.get('name', 'N/A')}`")
                    st.markdown(f"**Version:** {details.get('version', 'N/A')}")
                
                elif activity["type"] == "Deployment":
                    details = activity["details"]
                    st.markdown(f"**Endpoint:** `{details.get('endpoint_name', 'N/A')}`")
                    st.markdown(f"**Size:** {details.get('workload_size', 'N/A')}")

with tab2:
    st.markdown("### Pipeline Analytics")
    
    # Summary metrics
    st.markdown("#### Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Runs", len(training_history))
    with col2:
        st.metric("Registered Models", len(registered_models))
    with col3:
        st.metric("Active Endpoints", len(endpoints))
    with col4:
        st.metric("Configurations", len(recent_configs))
    
    st.markdown("---")
    
    # Training history chart
    if training_history:
        st.markdown("#### Training History")
        
        # Resource usage over time
        fig = VisualizationHelper.resource_usage_chart(training_history)
        st.plotly_chart(fig, use_container_width=True)
    
    # Activity by type
    st.markdown("#### Activity Distribution")
    
    activity_counts = {
        "Training": len(training_history),
        "Registration": len(registered_models),
        "Deployment": len(endpoints),
        "Configuration": len(recent_configs)
    }
    
    fig = VisualizationHelper.class_distribution_chart(
        activity_counts,
        "Activity by Type"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Settings & Preferences")
    
    # User preferences
    st.markdown("#### User Preferences")
    
    prefs = StateManager.get_user_preferences()
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_catalog = st.text_input(
            "Default Catalog",
            value=prefs.get("default_catalog", "main"),
            help="Default Unity Catalog name"
        )
    
    with col2:
        default_schema = st.text_input(
            "Default Schema",
            value=prefs.get("default_schema", "cv_models"),
            help="Default schema name"
        )
    
    default_volume = st.text_input(
        "Default Volume",
        value=prefs.get("default_volume", "cv_data"),
        help="Default volume name for data storage"
    )
    
    workspace_email = st.text_input(
        "Workspace Email",
        value=prefs.get("workspace_email", ""),
        placeholder="user@email.com",
        help="Your Databricks workspace email (for experiment paths)"
    )
    
    if st.button("💾 Save Preferences", type="primary"):
        StateManager.set_user_preferences({
            "default_catalog": default_catalog,
            "default_schema": default_schema,
            "default_volume": default_volume,
            "workspace_email": workspace_email
        })
        st.success("✅ Preferences saved!")
    
    st.markdown("---")
    
    # Default paths preview
    st.markdown("#### Default Paths")
    
    default_paths = StateManager.get_default_paths()
    
    with st.expander("📁 View Default Paths"):
        for key, path in default_paths.items():
            st.code(f"{key}: {path}")
    
    st.markdown("---")
    
    # Danger zone
    st.markdown("#### ⚠️ Danger Zone")
    
    with st.expander("🗑️ Clear Data", expanded=False):
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
    st.markdown("### Export & Import")
    
    # Export section
    st.markdown("#### Export Data")
    
    st.info("Export your pipeline history and configurations")
    
    export_options = st.multiselect(
        "Select data to export",
        options=[
            "Training History",
            "Registered Models",
            "Endpoints",
            "Configurations",
            "User Preferences"
        ],
        default=["Training History", "Configurations"]
    )
    
    if st.button("📥 Export Data", type="primary"):
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
            label="📥 Download Export",
            data=export_json,
            file_name=f"pipeline_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    st.markdown("---")
    
    # Import section
    st.markdown("#### Import Data")
    
    st.info("Import previously exported data")
    
    uploaded_file = st.file_uploader(
        "Upload export file",
        type=["json"],
        help="Upload a JSON export file"
    )
    
    if uploaded_file:
        try:
            import json
            
            import_data = json.load(uploaded_file)
            
            st.success("✅ File loaded successfully")
            
            # Preview
            with st.expander("📄 Preview Import Data"):
                st.json(import_data)
            
            if st.button("📤 Import Data", type="primary"):
                # Merge imported data
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
                
                st.success("✅ Data imported successfully!")
                st.balloons()
                st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    
    total_activities = len(training_history) + len(registered_models) + len(endpoints) + len(recent_configs)
    st.metric("Total Activities", total_activities)
    
    if training_history:
        completed = len([r for r in training_history if r.get("status") == "SUCCESS"])
        st.metric("Completed Runs", completed)
    
    st.markdown("---")
    st.markdown("### 🔄 Quick Actions")
    
    if st.button("📊 View Analytics", use_container_width=True):
        # Already on this page
        pass
    
    if st.button("⚙️ Settings", use_container_width=True):
        # Already on this page
        pass


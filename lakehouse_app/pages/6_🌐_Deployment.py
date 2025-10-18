"""
Page 6: Model Deployment
Deploy models to serving endpoints
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Note: lakehouse_app is self-contained, no need for parent directory imports
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient

# Initialize state
StateManager.initialize()

# Page config
st.title("🌐 Model Deployment")
st.markdown("Deploy models to Databricks Model Serving endpoints")

# Initialize client
client = DatabricksJobClient()

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Deploy Model", "📊 Active Endpoints", "⚙️ Batch Inference"])

with tab1:
    st.markdown("### Deploy to Model Serving")
    
    # Check if model was selected from registration page
    deployment_model = StateManager.get("deployment_model")
    
    # Step 1: Select model
    st.markdown("#### Step 1: Select Model")
    
    if deployment_model:
        st.success(f"✅ Selected model: {deployment_model.get('name', 'N/A')}")
        model_name = deployment_model.get("name", "")
        model_version = deployment_model.get("version", "1")
        
        if st.button("🔄 Choose Different Model"):
            StateManager.set("deployment_model", None)
            st.rerun()
    else:
        catalog = st.text_input("Catalog", value="main")
        schema = st.text_input("Schema", value="cv_models")
        model_name_input = st.text_input("Model Name", value="")
        
        if model_name_input:
            model_name = f"{catalog}.{schema}.{model_name_input}"
            model_version = st.text_input("Model Version", value="1")
        else:
            model_name = ""
            model_version = "1"
        
        st.info("💡 Or select a model from the Model Registration page")
    
    # Step 2: Endpoint configuration
    st.markdown("---")
    st.markdown("#### Step 2: Endpoint Configuration")
    
    endpoint_name = st.text_input(
        "Endpoint Name",
        value=f"{model_name.split('.')[-1] if model_name else 'model'}_endpoint",
        help="Name for the serving endpoint"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        workload_size = st.selectbox(
            "Workload Size",
            options=["Small", "Medium", "Large"],
            help="Compute size for the endpoint"
        )
    
    with col2:
        scale_to_zero = st.checkbox(
            "Enable Scale-to-Zero",
            value=True,
            help="Automatically scale down when not in use"
        )
    
    # Endpoint details
    with st.expander("ℹ️ Workload Size Details"):
        st.markdown("""
        - **Small:** For light workloads and testing (cheaper)
        - **Medium:** For moderate production workloads
        - **Large:** For high-throughput production workloads
        
        Scale-to-zero reduces costs by shutting down when idle.
        """)
    
    # Step 3: Review and deploy
    st.markdown("---")
    st.markdown("#### Step 3: Review and Deploy")
    
    with st.expander("📋 Deployment Summary", expanded=True):
        st.markdown(f"**Model:** `{model_name}`")
        st.markdown(f"**Version:** {model_version}")
        st.markdown(f"**Endpoint:** `{endpoint_name}`")
        st.markdown(f"**Workload Size:** {workload_size}")
        st.markdown(f"**Scale-to-Zero:** {'Enabled' if scale_to_zero else 'Disabled'}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Deploy to Endpoint", type="primary", use_container_width=True):
            if not model_name:
                st.error("❌ Please select or enter a model name")
            elif not endpoint_name:
                st.error("❌ Please enter an endpoint name")
            else:
                with st.spinner("Deploying model to endpoint..."):
                    try:
                        result = client.create_model_serving_endpoint(
                            endpoint_name=endpoint_name,
                            model_name=model_name,
                            model_version=model_version,
                            workload_size=workload_size,
                            scale_to_zero=scale_to_zero
                        )
                        
                        if result.get("status") in ["created", "updated"]:
                            st.success(f"✅ Model deployed successfully!")
                            st.info(f"**Endpoint:** {endpoint_name}")
                            st.info(f"**Status:** {result['status'].title()}")
                            st.balloons()
                            
                            # Update state
                            StateManager.add_endpoint({
                                "endpoint_name": endpoint_name,
                                "model_name": model_name,
                                "model_version": model_version,
                                "workload_size": workload_size,
                                "created_at": datetime.now().isoformat()
                            })
                            
                            st.info("💡 Endpoint is being provisioned. Check the 'Active Endpoints' tab to monitor status.")
                        else:
                            st.error(f"❌ Deployment failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error deploying model: {str(e)}")
    
    with col2:
        if st.button("💾 Save Config", use_container_width=True):
            st.info("Deployment config saved")

with tab2:
    st.markdown("### Active Endpoints")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Get endpoints from state
    endpoints = StateManager.get("endpoints", [])
    
    if not endpoints:
        st.info("ℹ️ No endpoints tracked")
        st.markdown("Deploy a model to create an endpoint")
    else:
        st.success(f"✅ Found {len(endpoints)} endpoint(s)")
        
        # Display endpoints
        for endpoint in endpoints:
            endpoint_name = endpoint.get("endpoint_name", "")
            
            with st.expander(f"🌐 {endpoint_name}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Endpoint:** `{endpoint_name}`")
                    st.markdown(f"**Model:** {endpoint.get('model_name', 'N/A')}")
                    st.markdown(f"**Version:** {endpoint.get('model_version', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Size:** {endpoint.get('workload_size', 'N/A')}")
                    st.markdown(f"**Created:** {endpoint.get('created_at', 'N/A')[:10]}")
                
                # Get endpoint status
                try:
                    status = client.get_endpoint_status(endpoint_name)
                    
                    st.markdown("#### Status")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        state = status.get("state", "UNKNOWN")
                        ready = status.get("ready", "UNKNOWN")
                        
                        if state == "NOT_UPDATING" and ready == "READY":
                            st.success("✅ Ready")
                        elif state == "UPDATING":
                            st.info("🔄 Updating...")
                        else:
                            st.warning(f"⚠️ {state}")
                    
                    with col2:
                        if status.get("endpoint_url"):
                            st.markdown(f"[🔗 Endpoint URL]({status['endpoint_url']})")
                
                except Exception as e:
                    st.error(f"Could not fetch status: {str(e)}")
                
                st.markdown("---")
                
                # Actions
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🎮 Test", key=f"test_{endpoint_name}", use_container_width=True):
                        StateManager.set("inference_endpoint", endpoint_name)
                        st.switch_page("pages/7_🎮_Inference.py")
                
                with col2:
                    if st.button("📊 Metrics", key=f"metrics_{endpoint_name}", use_container_width=True):
                        st.info("Endpoint metrics dashboard")
                
                with col3:
                    if st.button("⚙️ Update", key=f"update_{endpoint_name}", use_container_width=True):
                        st.info("Update endpoint configuration")
                
                with col4:
                    if st.button("🗑️ Delete", key=f"delete_{endpoint_name}", use_container_width=True):
                        st.warning("Delete confirmation would appear here")

with tab3:
    st.markdown("### Batch Inference")
    
    st.info("Configure batch inference jobs for processing large datasets")
    
    # Step 1: Select model/endpoint
    st.markdown("#### Step 1: Select Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        inference_type = st.radio(
            "Inference Type",
            options=["Use Endpoint", "Use Checkpoint"],
            help="Choose whether to use a deployed endpoint or model checkpoint"
        )
    
    with col2:
        if inference_type == "Use Endpoint":
            endpoints = StateManager.get("endpoints", [])
            endpoint_names = [e["endpoint_name"] for e in endpoints]
            
            if endpoint_names:
                selected_endpoint = st.selectbox("Select Endpoint", endpoint_names)
            else:
                st.warning("No endpoints available")
                selected_endpoint = None
        else:
            checkpoint_path = st.text_input(
                "Checkpoint Path",
                value="/Volumes/<catalog>/<schema>/<volume>/checkpoints/model.ckpt"
            )
    
    # Step 2: Input/output configuration
    st.markdown("---")
    st.markdown("#### Step 2: Data Configuration")
    
    input_path = st.text_input(
        "Input Data Path",
        value="/Volumes/<catalog>/<schema>/<volume>/inference/input",
        help="Path to images or Delta table for inference"
    )
    
    output_path = st.text_input(
        "Output Path",
        value="/Volumes/<catalog>/<schema>/<volume>/inference/output",
        help="Path to save predictions"
    )
    
    output_format = st.selectbox(
        "Output Format",
        options=["Delta Table", "Parquet", "JSON", "CSV"],
        help="Format for saving predictions"
    )
    
    # Step 3: Job configuration
    st.markdown("---")
    st.markdown("#### Step 3: Job Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=256, value=32)
    
    with col2:
        save_visualizations = st.checkbox("Save Annotated Images", value=True)
    
    # Launch batch job
    if st.button("🚀 Launch Batch Inference", type="primary"):
        st.info("""
        🔄 Batch inference job would:
        1. Load input data
        2. Run inference using model/endpoint
        3. Save predictions to output path
        4. Optionally save visualizations
        """)
        st.success("✅ Batch job configured")
        st.info("💡 In production, this would submit a Databricks job")

# Sidebar
with st.sidebar:
    st.markdown("### 🌐 Deployment Status")
    
    endpoints = StateManager.get("endpoints", [])
    st.metric("Active Endpoints", len(endpoints))
    
    if endpoints:
        st.markdown("#### Quick Access")
        for endpoint in endpoints[:3]:  # Show first 3
            if st.button(f"🌐 {endpoint['endpoint_name']}", key=f"sidebar_{endpoint['endpoint_name']}", use_container_width=True):
                StateManager.set("inference_endpoint", endpoint['endpoint_name'])
                st.switch_page("pages/7_🎮_Inference.py")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    
    st.info("""
    - Test endpoints before production use
    - Enable scale-to-zero to reduce costs
    - Monitor endpoint metrics regularly
    - Use batch inference for large datasets
    """)


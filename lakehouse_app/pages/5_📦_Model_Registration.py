"""
Page 5: Model Registration
Register trained models to Unity Catalog
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Note: lakehouse_app is self-contained, no need for parent directory imports
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from components.metrics_display import MetricsDisplay

# Initialize state
StateManager.initialize()

# Page config
st.title("📦 Model Registration")
st.markdown("Register trained models to Unity Catalog Model Registry")

# Initialize client
client = DatabricksJobClient()

# Tabs
tab1, tab2 = st.tabs(["📝 Register Model", "📋 Registered Models"])

with tab1:
    st.markdown("### Register a Trained Model")
    
    st.info("Register your trained model checkpoints to Unity Catalog for versioning and deployment")
    
    # Step 1: Select checkpoint
    st.markdown("#### Step 1: Model Checkpoint")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="/Volumes/<catalog>/<schema>/<volume>/checkpoints/best_model.ckpt",
            help="Path to the model checkpoint file"
        )
    
    with col2:
        if st.button("🔍 Browse", use_container_width=True):
            st.info("File browser not yet implemented")
    
    config_path = st.text_input(
        "Configuration Path",
        value=StateManager.get("config_path", "/Volumes/<catalog>/<schema>/<volume>/configs/config.yaml"),
        help="Path to the configuration used for training"
    )
    
    # Step 2: Model naming
    st.markdown("---")
    st.markdown("#### Step 2: Model Registration Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        catalog = st.text_input("Catalog", value="main", help="Unity Catalog name")
    
    with col2:
        schema = st.text_input("Schema", value="cv_models", help="Schema name")
    
    with col3:
        model_name = st.text_input(
            "Model Name",
            value=f"cv_model_{datetime.now().strftime('%Y%m%d')}",
            help="Name for the registered model"
        )
    
    full_model_name = f"{catalog}.{schema}.{model_name}"
    st.info(f"📝 Full model name: `{full_model_name}`")
    
    # Step 3: Model metadata
    st.markdown("---")
    st.markdown("#### Step 3: Model Metadata")
    
    description = st.text_area(
        "Description",
        value="",
        placeholder="Enter a description for this model...",
        help="Describe the model, dataset, and any important details"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        tags_input = st.text_input(
            "Tags (comma-separated)",
            value="cv,pytorch,hf_trainer",
            help="Add tags for easy searching"
        )
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
    
    with col2:
        stage = st.selectbox(
            "Initial Stage",
            options=["None", "Staging", "Production"],
            help="Set the initial stage for this model version"
        )
    
    # Step 4: Review and register
    st.markdown("---")
    st.markdown("#### Step 4: Review and Register")
    
    with st.expander("📋 Registration Summary", expanded=True):
        st.markdown(f"**Model Name:** `{full_model_name}`")
        st.markdown(f"**Checkpoint:** `{checkpoint_path}`")
        st.markdown(f"**Config:** `{config_path}`")
        st.markdown(f"**Description:** {description if description else 'None'}")
        st.markdown(f"**Tags:** {', '.join(tags)}")
        st.markdown(f"**Stage:** {stage}")
    
    # MLflow model source
    run_id = st.text_input(
        "MLflow Run ID",
        value="",
        help="Run ID from training (HF Trainer logs model to MLflow automatically)"
    )
    model_uri_input = st.text_input(
        "Model URI (optional, preferred over Run ID)",
        value="",
        help="Direct model URI from log_model, e.g. models:/abc123 — leave blank to auto-resolve from run",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("📦 Register Model", type="primary", use_container_width=True):
            if not run_id and not model_uri_input:
                st.error("❌ Please provide an MLflow Run ID or Model URI")
            elif not full_model_name or "<" in full_model_name:
                st.error("❌ Please provide a valid model name")
            else:
                with st.spinner("Registering model to Unity Catalog..."):
                    try:
                        import mlflow

                        if model_uri_input:
                            model_uri = model_uri_input
                        else:
                            # Resolve model URI from run params (MLflow 3 path)
                            _client = mlflow.MlflowClient()
                            _run = _client.get_run(run_id)
                            stored = _run.data.params.get("logged_model_uri")
                            model_uri = stored if stored else f"runs:/{run_id}/model"

                        mv = mlflow.register_model(model_uri, full_model_name)
                        version = mv.version

                        if description:
                            client.mlflow_client.update_registered_model(
                                full_model_name, description=description
                            )
                        for tag in tags:
                            client.mlflow_client.set_registered_model_tag(
                                full_model_name, tag, "true"
                            )

                        StateManager.add_registered_model({
                            "name": full_model_name,
                            "version": str(version),
                            "run_id": run_id,
                            "config_path": config_path,
                            "description": description,
                            "tags": tags,
                            "stage": stage,
                            "creation_timestamp": datetime.now().isoformat()
                        })

                        st.success(f"✅ Registered **{full_model_name}** version {version}")
                        st.info("💡 You can now deploy this model from the Deployment page")
                    except Exception as e:
                        st.error(f"❌ Error registering model: {e}")
                        st.info("Tracking locally — you can retry registration later")
                        StateManager.add_registered_model({
                            "name": full_model_name,
                            "version": "pending",
                            "run_id": run_id,
                            "config_path": config_path,
                            "description": description,
                            "tags": tags,
                            "stage": stage,
                            "creation_timestamp": datetime.now().isoformat()
                        })
    
    with col2:
        if st.button("💾 Save as Draft", use_container_width=True):
            st.info("Draft saved locally")

with tab2:
    st.markdown("### Registered Models")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Search Models",
            placeholder="Search by name, tag, or description...",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("🔄 Refresh List", use_container_width=True):
            st.rerun()
    
    # Load registered models
    try:
        # Try to get from MLflow
        mlflow_models = client.get_registered_models(max_results=50)
        
        # Merge with local state
        local_models = StateManager.get("registered_models", [])
        
        # Combine (prioritize MLflow)
        all_models = mlflow_models if mlflow_models else local_models
        
        if not all_models:
            st.info("ℹ️ No registered models found")
            st.markdown("Register a model to see it appear here")
        else:
            st.success(f"✅ Found {len(all_models)} registered model(s)")
            
            # Filter by search
            if search_query:
                all_models = [
                    m for m in all_models
                    if search_query.lower() in m.get("name", "").lower()
                    or search_query.lower() in str(m.get("description", "")).lower()
                ]
            
            # Display models
            for model in all_models:
                with st.expander(f"📦 {model.get('name', 'Unnamed Model')}"):
                    MetricsDisplay.display_model_card(model)
                    
                    st.markdown("---")
                    
                    # Actions
                    col1, col2, col3, col4 = st.columns(4)
                    
                    model_name_key = model.get("name", "").replace(".", "_").replace("-", "_")
                    
                    with col1:
                        if st.button("🚀 Deploy", key=f"deploy_{model_name_key}", use_container_width=True):
                            # Store model info for deployment page
                            StateManager.set("deployment_model", model)
                            st.switch_page("pages/6_🌐_Deployment.py")
                    
                    with col2:
                        if st.button("📊 Evaluate", key=f"eval_{model_name_key}", use_container_width=True):
                            st.switch_page("pages/4_📈_Evaluation.py")
                    
                    with col3:
                        if st.button("🎮 Test", key=f"test_{model_name_key}", use_container_width=True):
                            StateManager.set("inference_model", model)
                            st.switch_page("pages/7_🎮_Inference.py")
                    
                    with col4:
                        if st.button("📥 Download", key=f"download_{model_name_key}", use_container_width=True):
                            st.info("Download functionality would export model artifacts")
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        
        # Fall back to local state
        local_models = StateManager.get("registered_models", [])
        
        if local_models:
            st.warning("⚠️ Showing locally tracked models only")
            
            for model in local_models:
                with st.expander(f"📦 {model.get('name', 'Unnamed Model')}"):
                    st.markdown(f"**Version:** {model.get('version', 'N/A')}")
                    st.markdown(f"**Checkpoint:** `{model.get('checkpoint_path', 'N/A')}`")
                    st.markdown(f"**Created:** {model.get('creation_timestamp', 'N/A')}")

# Sidebar
with st.sidebar:
    st.markdown("### 📦 Model Registry")
    
    registered_models = StateManager.get("registered_models", [])
    st.metric("Registered Models", len(registered_models))
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("🚀 Deploy Model", use_container_width=True):
        st.switch_page("pages/6_🌐_Deployment.py")
    
    if st.button("🎮 Test Model", use_container_width=True):
        st.switch_page("pages/7_🎮_Inference.py")
    
    st.markdown("---")
    st.markdown("### 💡 Best Practices")
    
    st.info("""
    - Use descriptive model names
    - Add comprehensive descriptions
    - Tag models appropriately
    - Version models semantically
    - Test before promoting to Production
    """)


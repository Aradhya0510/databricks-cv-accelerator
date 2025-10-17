"""
Page 3: Model Training
Launch and monitor training jobs
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from utils.config_generator import ConfigGenerator
from components.visualizations import VisualizationHelper
from components.metrics_display import MetricsDisplay

# Initialize state
StateManager.initialize()

# Page config
st.title("🚀 Model Training")
st.markdown("Launch and monitor training jobs")

# Check for active config
current_config = StateManager.get_current_config()
config_path = StateManager.get("config_path")

if not current_config:
    st.warning("⚠️ No active configuration found")
    st.info("Please create or load a configuration in the Config Setup page first")
    
    if st.button("Go to Config Setup"):
        st.switch_page("pages/1_⚙️_Config_Setup.py")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Launch Training", "📊 Monitor Training", "📜 Training History"])

with tab1:
    st.markdown("### Launch Training Job")
    
    # Display current configuration summary
    st.success(f"✅ Active Configuration Loaded")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        task = current_config.get("model", {}).get("task_type", "N/A")
        st.metric("Task", task.replace("_", " ").title())
    with col2:
        model_name = current_config.get("model", {}).get("model_name", "N/A")
        st.metric("Model", model_name.split("/")[-1])
    with col3:
        epochs = current_config.get("training", {}).get("max_epochs", "N/A")
        st.metric("Epochs", epochs)
    with col4:
        batch_size = current_config.get("data", {}).get("batch_size", "N/A")
        st.metric("Batch Size", batch_size)
    
    with st.expander("📄 View Full Configuration"):
        st.code(ConfigGenerator.get_config_preview(current_config), language="yaml")
    
    st.markdown("---")
    
    # Job configuration
    st.markdown("### Job Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_name = st.text_input(
            "Job Name",
            value=f"cv_training_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name for the training job"
        )
    
    with col2:
        src_path = st.text_input(
            "Source Code Path",
            value="/Workspace/Repos/<username>/Databricks_CV_ref/src",
            help="Path to the src directory in your workspace"
        )
    
    # Cluster configuration
    st.markdown("#### Compute Configuration")
    
    use_serverless = st.checkbox(
        "Use Serverless Compute",
        value=False,
        help="Use Databricks serverless compute (recommended for simpler setup)"
    )
    
    if not use_serverless:
        col1, col2 = st.columns(2)
        
        with col1:
            node_type = st.selectbox(
                "Node Type",
                options=[
                    "g5.4xlarge (1 GPU, 16 vCPU, 64GB)",
                    "g5.8xlarge (1 GPU, 32 vCPU, 128GB)",
                    "g5.12xlarge (4 GPU, 48 vCPU, 192GB)",
                    "g5.24xlarge (4 GPU, 96 vCPU, 384GB)",
                ],
                index=0,
                help="Select GPU instance type"
            )
            node_type_id = node_type.split(" ")[0]
        
        with col2:
            num_workers = st.number_input(
                "Number of Workers",
                min_value=0,
                max_value=10,
                value=0,
                help="0 for single-node, >0 for distributed training"
            )
    
    # Email notifications
    with st.expander("📧 Email Notifications (Optional)"):
        enable_notifications = st.checkbox("Enable email notifications")
        
        if enable_notifications:
            email_list = st.text_input(
                "Email Addresses (comma-separated)",
                help="Enter email addresses to receive job notifications"
            )
            emails = [e.strip() for e in email_list.split(",") if e.strip()] if email_list else []
        else:
            emails = []
    
    st.markdown("---")
    
    # Launch button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Launch Training Job", type="primary", use_container_width=True):
            if not config_path:
                st.error("❌ Configuration path not found. Please save your configuration first.")
            elif not src_path or "<username>" in src_path:
                st.error("❌ Please provide a valid source code path")
            else:
                try:
                    with st.spinner("Creating and launching job..."):
                        # Initialize Databricks client
                        client = DatabricksJobClient()
                        
                        # Create cluster config
                        cluster_config = None if use_serverless else {
                            "spark_version": "14.3.x-gpu-ml-scala2.12",
                            "node_type_id": node_type_id,
                            "num_workers": num_workers,
                            "runtime_engine": "STANDARD",
                            "data_security_mode": "SINGLE_USER",
                        }
                        
                        # Create job
                        job_id = client.create_training_job(
                            job_name=job_name,
                            config_path=config_path,
                            src_path=src_path,
                            cluster_config=cluster_config,
                            email_notifications=emails if emails else None
                        )
                        
                        # Run job
                        run_id = client.run_job(job_id)
                        
                        # Update state
                        StateManager.set_active_training_run(run_id)
                        StateManager.add_training_run({
                            "run_id": run_id,
                            "job_id": job_id,
                            "job_name": job_name,
                            "config_path": config_path,
                            "task": task,
                            "model": model_name,
                            "timestamp": datetime.now().isoformat(),
                            "status": "RUNNING"
                        })
                        
                        st.success(f"✅ Training job launched successfully!")
                        st.info(f"**Job ID:** {job_id}")
                        st.info(f"**Run ID:** {run_id}")
                        st.balloons()
                        
                        # Switch to monitor tab
                        st.info("💡 Switch to the 'Monitor Training' tab to track progress")
                
                except Exception as e:
                    st.error(f"❌ Error launching job: {str(e)}")
                    st.exception(e)
    
    with col2:
        if st.button("💾 Save Config First", use_container_width=True):
            st.switch_page("pages/1_⚙️_Config_Setup.py")
    
    with col3:
        if st.button("📊 View Data", use_container_width=True):
            st.switch_page("pages/2_📊_Data_EDA.py")

with tab2:
    st.markdown("### Monitor Active Training")
    
    active_run_id = StateManager.get_active_training_run()
    
    if not active_run_id:
        st.info("ℹ️ No active training run")
        st.markdown("Launch a training job from the 'Launch Training' tab")
    else:
        st.success(f"📊 Monitoring Run ID: {active_run_id}")
        
        # Auto-refresh toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            auto_refresh = st.checkbox("Auto-refresh (every 30 seconds)", value=False)
        with col2:
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.rerun()
        
        try:
            # Get job status
            client = DatabricksJobClient()
            status = client.get_job_status(active_run_id)
            
            # Display status
            st.markdown("#### Job Status")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                state = status.get("life_cycle_state", "UNKNOWN")
                VisualizationHelper.display_status_badge(state)
            with col2:
                result = status.get("result_state", "UNKNOWN")
                st.metric("Result", result)
            with col3:
                if status.get("start_time"):
                    st.metric("Started", status["start_time"].strftime("%H:%M:%S"))
                else:
                    st.metric("Started", "N/A")
            with col4:
                if status.get("duration_seconds"):
                    duration_min = status["duration_seconds"] / 60
                    st.metric("Duration", f"{duration_min:.1f} min")
                elif status.get("start_time"):
                    elapsed = (datetime.now() - status["start_time"]).total_seconds() / 60
                    st.metric("Elapsed", f"{elapsed:.1f} min")
                else:
                    st.metric("Duration", "N/A")
            
            # Progress indicator
            if state == "RUNNING":
                st.info("🔄 Training in progress...")
                
                # Show progress bar (mock - in real impl would get from MLflow)
                epochs = current_config.get("training", {}).get("max_epochs", 100)
                mock_current_epoch = min(10, epochs)  # Mock current epoch
                
                MetricsDisplay.display_progress_bar(
                    mock_current_epoch,
                    epochs,
                    "Training Progress"
                )
            
            elif state == "TERMINATED":
                if result == "SUCCESS":
                    st.success("✅ Training completed successfully!")
                else:
                    st.error(f"❌ Training failed: {result}")
            
            # Link to run page
            if status.get("run_page_url"):
                st.markdown(f"[🔗 View in Databricks]({status['run_page_url']})")
            
            st.markdown("---")
            
            # Metrics display (mock - would fetch from MLflow in real implementation)
            st.markdown("#### Training Metrics")
            
            st.info("💡 Real-time metrics will be displayed here from MLflow")
            
            # Mock metrics visualization
            if state == "RUNNING" or result == "SUCCESS":
                # Mock metric data
                import random
                random.seed(42)
                
                mock_train_loss = [
                    {"step": i, "value": 2.0 * (0.9 ** i) + random.uniform(-0.1, 0.1), "timestamp": datetime.now()}
                    for i in range(20)
                ]
                
                mock_val_loss = [
                    {"step": i * 5, "value": 2.0 * (0.85 ** (i * 5 / 5)) + random.uniform(-0.1, 0.1), "timestamp": datetime.now()}
                    for i in range(4)
                ]
                
                # Display charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = VisualizationHelper.multi_metric_chart(
                        mock_train_loss,
                        mock_val_loss,
                        "Loss"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Mock accuracy/mAP
                    if task == "classification":
                        metric_name = "Accuracy"
                        mock_val_metric = [
                            {"step": i * 5, "value": 0.3 + 0.6 * (1 - 0.9 ** (i * 5 / 5)), "timestamp": datetime.now()}
                            for i in range(4)
                        ]
                    else:
                        metric_name = "mAP"
                        mock_val_metric = [
                            {"step": i * 5, "value": 0.2 + 0.5 * (1 - 0.9 ** (i * 5 / 5)), "timestamp": datetime.now()}
                            for i in range(4)
                        ]
                    
                    fig = VisualizationHelper.training_metrics_chart(
                        mock_val_metric,
                        metric_name
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Control buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if state == "RUNNING":
                    if st.button("⏸️ Cancel Training", type="secondary", use_container_width=True):
                        if client.cancel_job(active_run_id):
                            st.success("✅ Training cancelled")
                            StateManager.clear_active_training_run()
                            st.rerun()
                        else:
                            st.error("❌ Failed to cancel training")
            
            with col2:
                if state == "TERMINATED":
                    if st.button("🗑️ Clear Active Run", use_container_width=True):
                        StateManager.clear_active_training_run()
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error fetching job status: {str(e)}")
            st.info("The job may not exist or there may be a connection issue")
            
            if st.button("🗑️ Clear Active Run"):
                StateManager.clear_active_training_run()
                st.rerun()
        
        # Auto-refresh logic
        if auto_refresh and state == "RUNNING":
            time.sleep(30)
            st.rerun()

with tab3:
    st.markdown("### Training History")
    
    training_history = StateManager.get("training_history", [])
    
    if not training_history:
        st.info("ℹ️ No training history found")
        st.markdown("Launch training jobs to see them appear here")
    else:
        st.markdown(f"Found {len(training_history)} training run(s)")
        
        # Display as table
        import pandas as pd
        
        history_data = []
        for run in training_history:
            history_data.append({
                "Job Name": run.get("job_name", "N/A"),
                "Task": run.get("task", "N/A"),
                "Model": run.get("model", "N/A").split("/")[-1],
                "Status": run.get("status", "UNKNOWN"),
                "Timestamp": run.get("timestamp", "N/A"),
                "Run ID": run.get("run_id", "N/A")[:8]
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Detailed view
        st.markdown("---")
        st.markdown("### Run Details")
        
        for idx, run in enumerate(training_history):
            with st.expander(f"📊 {run.get('job_name', f'Run {idx}')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Run ID:** `{run.get('run_id', 'N/A')}`")
                    st.markdown(f"**Job ID:** `{run.get('job_id', 'N/A')}`")
                    st.markdown(f"**Task:** {run.get('task', 'N/A')}")
                    st.markdown(f"**Model:** {run.get('model', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Status:** {run.get('status', 'UNKNOWN')}")
                    st.markdown(f"**Timestamp:** {run.get('timestamp', 'N/A')}")
                    st.markdown(f"**Config:** `{Path(run.get('config_path', 'N/A')).name}`")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 View Metrics", key=f"metrics_{idx}", use_container_width=True):
                        st.switch_page("pages/4_📈_Evaluation.py")
                
                with col2:
                    if st.button("🔄 Set as Active", key=f"active_{idx}", use_container_width=True):
                        StateManager.set_active_training_run(run.get("run_id"))
                        st.success("✅ Set as active run")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{idx}", use_container_width=True):
                        training_history.remove(run)
                        StateManager.set("training_history", training_history)
                        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### 🚀 Training Status")
    
    active_run_id = StateManager.get_active_training_run()
    
    if active_run_id:
        st.success("✅ Active Training")
        st.code(f"Run ID: {active_run_id[:16]}...")
        
        if st.button("📊 Monitor", use_container_width=True):
            # Already on this page, just rerun to show monitor tab
            st.rerun()
    else:
        st.info("ℹ️ No active training")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    training_history = StateManager.get("training_history", [])
    st.metric("Total Runs", len(training_history))
    
    if training_history:
        completed = len([r for r in training_history if r.get("status") == "SUCCESS"])
        st.metric("Completed", completed)


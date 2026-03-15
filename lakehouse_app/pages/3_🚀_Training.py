"""
Page 3: Model Training
Launch and monitor training jobs
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import time
import yaml

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
        project_path = st.text_input(
            "Project Path",
            value="/Workspace/Users/<username>/Databricks_CV_ref",
            help="Workspace path to the project root (contains jobs/, src/, configs/)"
        )
    
    # Cluster configuration
    st.markdown("#### Compute Configuration")
    
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
            elif not project_path or "<username>" in project_path:
                st.error("❌ Please provide a valid project path")
            else:
                try:
                    with st.spinner("Creating and launching job..."):
                        client = DatabricksJobClient()

                        # Upload config YAML to a Volume so the cluster can read it via FUSE
                        config_filename = Path(config_path).name if config_path else f"{job_name}.yaml"
                        data_cfg = current_config.get("data", {})
                        train_path = data_cfg.get("train_data_path", "")
                        if train_path.startswith("/Volumes"):
                            parts = train_path.strip("/").split("/")
                            vol_base = "/".join(parts[:4])
                            vol_config_dir = f"/{vol_base}/configs"
                        else:
                            vol_config_dir = "/tmp/configs"
                        remote_config_path = f"{vol_config_dir}/{config_filename}"
                        config_bytes = yaml.dump(
                            current_config, default_flow_style=False, sort_keys=False
                        ).encode("utf-8")
                        import io as _io
                        if remote_config_path.startswith("/Volumes"):
                            client.workspace_client.files.upload(
                                remote_config_path, _io.BytesIO(config_bytes), overwrite=True
                            )
                        else:
                            Path(vol_config_dir).mkdir(parents=True, exist_ok=True)
                            with open(remote_config_path, "wb") as fout:
                                fout.write(config_bytes)
                        st.info(f"Config uploaded to `{remote_config_path}`")

                        cluster_config = {
                            "spark_version": "17.3.x-gpu-ml-scala2.12",
                            "node_type_id": node_type_id,
                            "num_workers": num_workers,
                            "data_security_mode": "SINGLE_USER",
                        }
                        
                        job_id = client.create_training_job(
                            job_name=job_name,
                            config_path=remote_config_path,
                            project_path=project_path,
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
                
                max_epochs = current_config.get("training", {}).get("max_epochs", 100)
                current_epoch = max_epochs
                mlflow_experiment = current_config.get("mlflow", {}).get("experiment_name", "")
                if mlflow_experiment:
                    try:
                        runs = client.get_mlflow_runs(mlflow_experiment, max_results=1)
                        if runs and "epoch" in runs[0].get("metrics", {}):
                            current_epoch = int(runs[0]["metrics"]["epoch"])
                    except Exception:
                        pass
                
                MetricsDisplay.display_progress_bar(
                    min(current_epoch, max_epochs),
                    max_epochs,
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
            
            st.markdown("#### Training Metrics")

            if state == "RUNNING" or result == "SUCCESS":
                mlflow_experiment = current_config.get("mlflow", {}).get("experiment_name", "")
                if mlflow_experiment:
                    try:
                        runs = client.get_mlflow_runs(mlflow_experiment, max_results=1)
                        if runs:
                            latest = runs[0]
                            metrics = latest.get("metrics", {})
                            loss_history = client.get_run_metrics_history(latest["run_id"], "eval_loss")
                            primary_metric = "eval_map" if task != "classification" else "eval_accuracy"
                            primary_history = client.get_run_metrics_history(latest["run_id"], primary_metric)

                            col1, col2 = st.columns(2)
                            with col1:
                                if loss_history:
                                    fig = VisualizationHelper.training_metrics_chart(loss_history, "Eval Loss")
                                    st.plotly_chart(fig, use_container_width=True)
                                elif "eval_loss" in metrics:
                                    st.metric("Eval Loss", f"{metrics['eval_loss']:.4f}")
                            with col2:
                                if primary_history:
                                    fig = VisualizationHelper.training_metrics_chart(primary_history, primary_metric.replace("_", " ").title())
                                    st.plotly_chart(fig, use_container_width=True)
                                elif primary_metric in metrics:
                                    st.metric(primary_metric.replace("_", " ").title(), f"{metrics[primary_metric]:.4f}")
                    except Exception:
                        st.info("💡 Metrics will appear once the run starts logging to MLflow")
                else:
                    st.info("💡 Configure an MLflow experiment in your config to see live metrics")
            
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


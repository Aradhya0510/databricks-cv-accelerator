"""
Page 4: Model Evaluation
Evaluate and analyze model performance
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import random

# Note: lakehouse_app is self-contained, no need for parent directory imports
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from components.visualizations import VisualizationHelper
from components.metrics_display import MetricsDisplay
from components.image_viewer import ImageViewer

# Initialize state
StateManager.initialize()

# Page config
st.title("📈 Model Evaluation")
st.markdown("Analyze model performance and compare runs")

# Initialize client
client = DatabricksJobClient()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics Dashboard", "🔍 Predictions", "⚖️ Compare Models", "📑 Reports"])

with tab1:
    st.markdown("### Model Performance Metrics")
    
    # Select experiment
    col1, col2 = st.columns([3, 1])
    
    current_config = StateManager.get_current_config()
    default_exp = (current_config or {}).get("mlflow", {}).get("experiment_name", "/Users/<email@databricks.com>/cv_experiments")

    with col1:
        experiment_name = st.text_input(
            "MLflow Experiment Name",
            value=default_exp,
            help="Enter the MLflow experiment path"
        )
    
    with col2:
        if st.button("🔍 Load Runs", type="primary", use_container_width=True):
            st.session_state["load_runs"] = True
    
    if st.session_state.get("load_runs", False):
        with st.spinner("Loading MLflow runs..."):
            try:
                runs = client.get_mlflow_runs(experiment_name, max_results=50)
                
                if not runs:
                    st.info("ℹ️ No runs found in this experiment")
                else:
                    st.success(f"✅ Loaded {len(runs)} run(s)")
                    
                    # Select run
                    run_options = {
                        f"{run['run_name']} ({run['run_id'][:8]}) - {run['status']}": run
                        for run in runs
                    }
                    
                    selected_run_name = st.selectbox(
                        "Select Run",
                        options=list(run_options.keys())
                    )
                    
                    selected_run = run_options[selected_run_name]
                    
                    # Display run summary
                    MetricsDisplay.display_training_summary(
                        selected_run,
                        show_params=True,
                        show_metrics=True
                    )
                    
                    # Metrics visualization
                    if selected_run.get("metrics"):
                        st.markdown("---")
                        st.markdown("### 📊 Metrics Visualization")
                        
                        metrics = selected_run["metrics"]
                        
                        # Identify task from params or config
                        task = selected_run.get("params", {}).get("task", "detection")
                        
                        # Display task-specific metrics
                        if task == "classification":
                            key_metrics = ["val_accuracy", "val_loss", "val_f1"]
                        elif task == "detection":
                            key_metrics = ["val_map", "val_map_50", "val_loss"]
                        else:
                            key_metrics = ["val_miou", "val_loss"]
                        
                        # Filter available metrics
                        available_metrics = [m for m in key_metrics if m in metrics]
                        
                        if available_metrics:
                            selected_metrics = st.multiselect(
                                "Select Metrics to Visualize",
                                options=list(metrics.keys()),
                                default=available_metrics[:3]
                            )
                            
                            if selected_metrics:
                                # Fetch metric history
                                cols = st.columns(min(len(selected_metrics), 2))
                                
                                for idx, metric_name in enumerate(selected_metrics):
                                    col_idx = idx % 2
                                    with cols[col_idx]:
                                        # Get metric history
                                        history = client.get_run_metrics_history(
                                            selected_run["run_id"],
                                            metric_name
                                        )
                                        
                                        if history:
                                            fig = VisualizationHelper.training_metrics_chart(
                                                history,
                                                metric_name,
                                                title=metric_name.replace("_", " ").title()
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.info(f"No history available for {metric_name}")
            
            except Exception as e:
                st.error(f"❌ Error loading runs: {str(e)}")
                st.info("Make sure the experiment name is correct and you have access to it")
    else:
        st.info("👆 Enter an experiment name and click 'Load Runs' to view metrics")

with tab2:
    st.markdown("### Evaluation Results (from `jobs/evaluate.py`)")

    st.info("Run `jobs/evaluate.py` to generate evaluation results, then view them here.")

    current_config = StateManager.get_current_config()
    results_dir = current_config.get("output", {}).get("results_dir", "/tmp/results") if current_config else "/tmp/results"

    results_dir_input = st.text_input("Results Directory", value=results_dir)

    if st.button("📂 Load Evaluation Results", type="primary"):
        rd = results_dir_input.rstrip("/")
        metrics_path = f"{rd}/evaluation_metrics.json"
        error_path = f"{rd}/error_analysis.json"
        bench_path = f"{rd}/benchmark.json"

        eval_metrics = client.read_json(metrics_path)
        if eval_metrics:
            st.markdown("#### mAP Metrics")
            metric_cols = st.columns(4)
            key_metrics = ["eval_map", "eval_map_50", "eval_map_75", "eval_loss"]
            for i, key in enumerate(key_metrics):
                if key in eval_metrics:
                    with metric_cols[i]:
                        st.metric(key, f"{eval_metrics[key]:.4f}")

            per_class = {k: v for k, v in eval_metrics.items() if "map_class_" in k}
            if per_class:
                st.markdown("#### Per-Class AP")
                fig = VisualizationHelper.class_distribution_chart(
                    {k.split("_")[-1]: v for k, v in sorted(per_class.items())},
                    "Per-Class Average Precision"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No evaluation_metrics.json found in {results_dir_input}")

        error_data = client.read_json(error_path)
        if error_data:
            st.markdown("---")
            st.markdown("#### Error Analysis")
            summary = error_data.get("summary", {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Positives", summary.get("true_positives", 0))
            with col2:
                fp = (summary.get("false_positives_background", 0)
                      + summary.get("false_positives_confusion", 0)
                      + summary.get("false_positives_localisation", 0))
                st.metric("False Positives", fp)
            with col3:
                st.metric("False Negatives", summary.get("false_negatives", 0))

        bench_data = client.read_json(bench_path)
        if bench_data:
            st.markdown("---")
            st.markdown("#### Benchmark")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("FPS", f"{bench_data.get('fps', 0):.1f}")
            with col2:
                lat = bench_data.get("latency_per_batch_ms", {})
                st.metric("P50 Latency (ms)", f"{lat.get('p50', 0):.0f}")
            with col3:
                st.metric("P95 Latency (ms)", f"{lat.get('p95', 0):.0f}")

    st.markdown("---")
    st.markdown("#### Run Evaluation")
    st.code("python jobs/evaluate.py --config_path <config> --checkpoint_path <model>", language="bash")

with tab3:
    st.markdown("### Compare Multiple Models")
    
    st.info("Compare performance across different model runs")
    
    # Experiment selection
    experiment_name = st.text_input(
        "MLflow Experiment Name",
        value="/Users/<email@databricks.com>/cv_experiments",
        key="compare_experiment"
    )
    
    if st.button("🔍 Load Runs for Comparison", type="primary"):
        with st.spinner("Loading runs..."):
            try:
                runs = client.get_mlflow_runs(experiment_name, max_results=50)
                
                if not runs:
                    st.info("ℹ️ No runs found in this experiment")
                else:
                    st.success(f"✅ Loaded {len(runs)} run(s)")
                    
                    # Multi-select runs
                    run_options = {
                        f"{run['run_name']} ({run['run_id'][:8]})": run
                        for run in runs
                    }
                    
                    selected_runs = st.multiselect(
                        "Select Runs to Compare",
                        options=list(run_options.keys()),
                        default=list(run_options.keys())[:3] if len(run_options) >= 3 else list(run_options.keys())
                    )
                    
                    if selected_runs:
                        runs_to_compare = [run_options[name] for name in selected_runs]
                        
                        # Determine metrics to compare
                        all_metrics = set()
                        for run in runs_to_compare:
                            all_metrics.update(run.get("metrics", {}).keys())
                        
                        if all_metrics:
                            # Select metrics
                            metrics_to_compare = st.multiselect(
                                "Select Metrics to Compare",
                                options=sorted(list(all_metrics)),
                                default=sorted(list(all_metrics))[:3]
                            )
                            
                            if metrics_to_compare:
                                # Display comparison table
                                MetricsDisplay.display_comparison_table(
                                    runs_to_compare,
                                    metrics_to_compare
                                )
                                
                                st.markdown("---")
                                
                                # Visualization
                                st.markdown("#### Visual Comparison")
                                
                                for metric in metrics_to_compare:
                                    fig = VisualizationHelper.model_comparison_chart(
                                        runs_to_compare,
                                        metric
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ No metrics found for selected runs")
                    else:
                        st.info("👆 Select runs to compare")
            
            except Exception as e:
                st.error(f"❌ Error loading runs: {str(e)}")

with tab4:
    st.markdown("### Evaluation Reports")
    
    st.info("Generate and export evaluation reports")
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            options=[
                "Performance Summary",
                "Detailed Metrics",
                "Error Analysis",
                "Model Comparison",
                "Full Report"
            ]
        )
    
    with col2:
        report_format = st.selectbox(
            "Export Format",
            options=["PDF", "HTML", "Markdown", "JSON"]
        )
    
    include_visualizations = st.checkbox("Include Visualizations", value=True)
    include_sample_predictions = st.checkbox("Include Sample Predictions", value=True)
    
    if st.button("📄 Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            import json as _json
            config = StateManager.get_current_config() or {}
            exp_name = config.get("mlflow", {}).get("experiment_name", "")
            report_data = {"generated_at": datetime.now().isoformat(), "report_type": report_type}

            if exp_name:
                try:
                    runs = client.get_mlflow_runs(exp_name, max_results=10)
                    report_data["experiment"] = exp_name
                    report_data["runs"] = [
                        {"run_id": r["run_id"], "run_name": r["run_name"], "status": r["status"], "metrics": r.get("metrics", {})}
                        for r in runs
                    ]
                except Exception:
                    report_data["runs"] = []

            results_dir = config.get("output", {}).get("results_dir", "")
            for fname in ["evaluation_metrics.json", "error_analysis.json", "benchmark.json"]:
                if results_dir:
                    fpath = f"{results_dir.rstrip('/')}/{fname}"
                    data = client.read_json(fpath)
                    if data:
                        report_data[fname.replace(".json", "")] = data

            if report_format == "JSON":
                content = _json.dumps(report_data, indent=2, default=str)
                mime = "application/json"
            else:
                lines = [f"# Evaluation Report — {report_type}", f"Generated: {report_data['generated_at']}", ""]
                for run in report_data.get("runs", []):
                    lines.append(f"## {run['run_name']} ({run['status']})")
                    for k, v in sorted(run.get("metrics", {}).items()):
                        lines.append(f"- **{k}:** {v:.4f}" if isinstance(v, float) else f"- **{k}:** {v}")
                    lines.append("")
                content = "\n".join(lines)
                mime = "text/markdown"

            st.success("✅ Report generated!")
            st.download_button(
                label="📥 Download Report",
                data=content,
                file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format.lower() if report_format == 'JSON' else 'md'}",
                mime=mime,
            )
    
    st.markdown("---")
    st.markdown("#### Recent Reports")
    
    st.info("📁 Your generated reports will appear here")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Evaluation Tools")
    
    current_config = StateManager.get_current_config()
    
    if current_config:
        task = current_config.get("model", {}).get("task_type", "N/A")
        st.markdown(f"**Task:** {task.replace('_', ' ').title()}")
        
        st.markdown("---")
        st.markdown("### Task-Specific Metrics")
        
        if task == "classification":
            st.markdown("- Accuracy")
            st.markdown("- Precision/Recall")
            st.markdown("- F1 Score")
            st.markdown("- Confusion Matrix")
        elif task == "detection":
            st.markdown("- mAP (mean Average Precision)")
            st.markdown("- mAP@50, mAP@75")
            st.markdown("- Precision/Recall curves")
            st.markdown("- Per-class AP")
        else:
            st.markdown("- mIoU (mean IoU)")
            st.markdown("- Per-class IoU")
            st.markdown("- Pixel Accuracy")
            st.markdown("- Dice Coefficient")
    else:
        st.info("Load a configuration to see task-specific metrics")
    
    st.markdown("---")
    
    if st.button("🚀 Train New Model", use_container_width=True):
        st.switch_page("pages/3_🚀_Training.py")


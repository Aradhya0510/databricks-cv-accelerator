"""
Page 4: Model Evaluation
Evaluate and analyze model performance
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    
    with col1:
        experiment_name = st.text_input(
            "MLflow Experiment Name",
            value="/Users/<email@databricks.com>/cv_experiments",
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
    st.markdown("### Model Predictions")
    
    st.info("🖼️ Visualize model predictions on validation/test data")
    
    # Get current config to determine task
    current_config = StateManager.get_current_config()
    
    if not current_config:
        st.warning("⚠️ No active configuration. Please load a configuration first.")
    else:
        task = current_config.get("model", {}).get("task_type", "detection")
        
        st.markdown(f"**Task:** {task.replace('_', ' ').title()}")
        
        # Mock prediction visualization
        st.markdown("#### Sample Predictions")
        
        num_samples = st.slider("Number of samples", 1, 12, 6)
        
        if st.button("🎲 Load Random Predictions", type="primary"):
            st.info("💡 In a real implementation, this would load predictions from the model output")
            
            st.markdown("#### Prediction Samples")
            
            if task == "detection":
                st.info("For detection tasks, predictions would show:")
                st.markdown("- Bounding boxes with class labels")
                st.markdown("- Confidence scores")
                st.markdown("- Ground truth vs. predictions comparison")
            
            elif task == "classification":
                st.info("For classification tasks, predictions would show:")
                st.markdown("- Predicted class with confidence")
                st.markdown("- Top-k predictions")
                st.markdown("- Comparison with ground truth")
            
            else:
                st.info("For segmentation tasks, predictions would show:")
                st.markdown("- Predicted segmentation masks")
                st.markdown("- Overlay on original images")
                st.markdown("- IoU scores per instance/class")
            
            st.code("""
# Example prediction loading code:
from PIL import Image
import torch

# Load model checkpoint
model = load_model_from_checkpoint(checkpoint_path)
model.eval()

# Load test images
test_dataset = load_dataset(test_data_path)
samples = random.sample(test_dataset, num_samples)

# Generate predictions
predictions = []
for image, label in samples:
    with torch.no_grad():
        pred = model(image)
    predictions.append((image, label, pred))

# Visualize
for img, gt, pred in predictions:
    if task == "detection":
        annotated = ImageViewer.draw_bounding_boxes(
            img, pred['boxes'], pred['labels'], pred['scores']
        )
    elif task == "classification":
        annotated = ImageViewer.annotate_classification(
            img, pred['class'], pred['confidence'], gt
        )
    else:  # segmentation
        annotated = ImageViewer.draw_segmentation_mask(
            img, pred['mask']
        )
    
    ImageViewer.display_image(annotated)
            """, language="python")
        
        # Error analysis
        st.markdown("---")
        st.markdown("#### Error Analysis")
        
        if st.button("🔍 Analyze Errors"):
            st.info("Error analysis would include:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Common Error Types:**")
                st.markdown("- False positives")
                st.markdown("- False negatives")
                st.markdown("- Misclassifications")
                st.markdown("- Localization errors")
            
            with col2:
                st.markdown("**Error Patterns:**")
                st.markdown("- Confused classes")
                st.markdown("- Confidence distribution")
                st.markdown("- Performance by object size")
                st.markdown("- Performance by scene complexity")

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
            st.success("✅ Report generated!")
            
            st.info(f"""
            Report would include:
            - **Type:** {report_type}
            - **Format:** {report_format}
            - **Visualizations:** {'Yes' if include_visualizations else 'No'}
            - **Sample Predictions:** {'Yes' if include_sample_predictions else 'No'}
            """)
            
            # Mock download button
            st.download_button(
                label="📥 Download Report",
                data="Mock report content",
                file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format.lower()}",
                mime="application/octet-stream"
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


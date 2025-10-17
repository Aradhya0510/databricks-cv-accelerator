"""
Metrics Display Component
Display training and evaluation metrics
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime


class MetricsDisplay:
    """Component for displaying metrics."""
    
    @staticmethod
    def display_metrics_grid(
        metrics: Dict[str, float],
        columns: int = 4,
        format_str: str = "{:.4f}"
    ):
        """
        Display metrics in a grid layout.
        
        Args:
            metrics: Dictionary of metric names to values
            columns: Number of columns
            format_str: Format string for metric values
        """
        if not metrics:
            st.info("No metrics available")
            return
        
        metric_items = list(metrics.items())
        cols = st.columns(columns)
        
        for idx, (name, value) in enumerate(metric_items):
            col_idx = idx % columns
            with cols[col_idx]:
                # Format metric name (make it readable)
                display_name = name.replace("_", " ").title()
                
                # Format value
                if isinstance(value, (int, float)):
                    display_value = format_str.format(value)
                else:
                    display_value = str(value)
                
                st.metric(label=display_name, value=display_value)
    
    @staticmethod
    def display_training_summary(
        run_info: Dict[str, Any],
        show_params: bool = True,
        show_metrics: bool = True
    ):
        """
        Display a comprehensive training run summary.
        
        Args:
            run_info: Dictionary with run information
            show_params: Whether to show parameters
            show_metrics: Whether to show metrics
        """
        st.markdown("### 📊 Training Run Summary")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Run ID:**")
            st.code(run_info.get('run_id', 'N/A'))
        with col2:
            st.markdown("**Status:**")
            status = run_info.get('status', 'UNKNOWN')
            status_color = {
                'FINISHED': '🟢',
                'RUNNING': '🟡',
                'FAILED': '🔴',
                'SCHEDULED': '⚪'
            }.get(status, '⚪')
            st.markdown(f"{status_color} {status}")
        with col3:
            st.markdown("**Duration:**")
            start_time = run_info.get('start_time')
            end_time = run_info.get('end_time')
            if start_time and end_time:
                duration = (end_time - start_time).total_seconds()
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                st.markdown(f"{hours}h {minutes}m")
            else:
                st.markdown("N/A")
        
        st.markdown("---")
        
        # Parameters
        if show_params and 'params' in run_info:
            with st.expander("⚙️ Parameters", expanded=False):
                params_df = pd.DataFrame([
                    {"Parameter": k, "Value": v}
                    for k, v in run_info['params'].items()
                ])
                st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        # Metrics
        if show_metrics and 'metrics' in run_info:
            with st.expander("📈 Metrics", expanded=True):
                MetricsDisplay.display_metrics_grid(run_info['metrics'])
        
        # Tags
        if 'tags' in run_info:
            with st.expander("🏷️ Tags", expanded=False):
                tags_df = pd.DataFrame([
                    {"Tag": k, "Value": v}
                    for k, v in run_info['tags'].items()
                ])
                st.dataframe(tags_df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def display_comparison_table(
        runs: List[Dict[str, Any]],
        metrics_to_compare: List[str],
        params_to_compare: Optional[List[str]] = None
    ):
        """
        Display a comparison table for multiple runs.
        
        Args:
            runs: List of run dictionaries
            metrics_to_compare: List of metric names to compare
            params_to_compare: Optional list of parameter names to compare
        """
        if not runs:
            st.info("No runs to compare")
            return
        
        st.markdown("### 🔄 Model Comparison")
        
        # Build comparison data
        comparison_data = []
        for run in runs:
            row = {
                "Run Name": run.get('run_name', 'Unnamed'),
                "Run ID": run.get('run_id', '')[:8],  # Short ID
                "Status": run.get('status', 'UNKNOWN'),
            }
            
            # Add metrics
            for metric in metrics_to_compare:
                value = run.get('metrics', {}).get(metric, None)
                if value is not None:
                    row[metric] = f"{value:.4f}"
                else:
                    row[metric] = "N/A"
            
            # Add params
            if params_to_compare:
                for param in params_to_compare:
                    row[param] = run.get('params', {}).get(param, "N/A")
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Highlight best metrics
        if metrics_to_compare:
            st.markdown("#### 🏆 Best Metrics")
            best_cols = st.columns(len(metrics_to_compare))
            
            for idx, metric in enumerate(metrics_to_compare):
                with best_cols[idx]:
                    # Find best value (assumes higher is better, adjust as needed)
                    metric_values = []
                    for run in runs:
                        value = run.get('metrics', {}).get(metric, None)
                        if value is not None:
                            metric_values.append(value)
                    
                    if metric_values:
                        # Determine if higher or lower is better
                        if 'loss' in metric.lower() or 'error' in metric.lower():
                            best_value = min(metric_values)
                        else:
                            best_value = max(metric_values)
                        
                        st.metric(
                            label=metric,
                            value=f"{best_value:.4f}"
                        )
    
    @staticmethod
    def display_evaluation_results(
        task: str,
        metrics: Dict[str, float],
        detailed_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Display evaluation results specific to task type.
        
        Args:
            task: Task type (classification, detection, etc.)
            metrics: Dictionary of main metrics
            detailed_metrics: Optional detailed metrics (per-class, etc.)
        """
        st.markdown("### 📈 Evaluation Results")
        
        # Main metrics
        MetricsDisplay.display_metrics_grid(metrics)
        
        st.markdown("---")
        
        # Task-specific detailed metrics
        if detailed_metrics:
            if task == "classification" and "per_class_accuracy" in detailed_metrics:
                with st.expander("📊 Per-Class Accuracy", expanded=False):
                    per_class = detailed_metrics["per_class_accuracy"]
                    df = pd.DataFrame([
                        {"Class": k, "Accuracy": f"{v:.4f}"}
                        for k, v in per_class.items()
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
            
            elif task == "detection" and "per_class_ap" in detailed_metrics:
                with st.expander("📊 Per-Class Average Precision", expanded=False):
                    per_class = detailed_metrics["per_class_ap"]
                    df = pd.DataFrame([
                        {"Class": k, "AP": f"{v:.4f}"}
                        for k, v in per_class.items()
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
            
            elif task in ["semantic_segmentation", "instance_segmentation"] and "per_class_iou" in detailed_metrics:
                with st.expander("📊 Per-Class IoU", expanded=False):
                    per_class = detailed_metrics["per_class_iou"]
                    df = pd.DataFrame([
                        {"Class": k, "IoU": f"{v:.4f}"}
                        for k, v in per_class.items()
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def display_progress_bar(
        current: int,
        total: int,
        label: str = "Progress",
        show_percentage: bool = True
    ):
        """
        Display a progress bar.
        
        Args:
            current: Current progress value
            total: Total value
            label: Progress bar label
            show_percentage: Whether to show percentage
        """
        progress = current / total if total > 0 else 0
        
        if show_percentage:
            st.markdown(f"**{label}:** {current}/{total} ({progress:.1%})")
        else:
            st.markdown(f"**{label}:** {current}/{total}")
        
        st.progress(progress)
    
    @staticmethod
    def display_status_timeline(events: List[Dict[str, Any]]):
        """
        Display a timeline of status events.
        
        Args:
            events: List of event dictionaries with timestamp, status, message
        """
        st.markdown("### 📅 Status Timeline")
        
        for event in events:
            timestamp = event.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            status = event.get('status', 'INFO')
            message = event.get('message', '')
            
            # Status icon
            icon = {
                'SUCCESS': '✅',
                'ERROR': '❌',
                'WARNING': '⚠️',
                'INFO': 'ℹ️',
                'RUNNING': '🔄'
            }.get(status, 'ℹ️')
            
            # Display event
            st.markdown(f"{icon} **{timestamp.strftime('%Y-%m-%d %H:%M:%S')}** - {message}")
    
    @staticmethod
    def display_model_card(model_info: Dict[str, Any]):
        """
        Display a model card with key information.
        
        Args:
            model_info: Dictionary with model information
        """
        st.markdown("### 📦 Model Card")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Information")
            st.markdown(f"**Model Name:** {model_info.get('name', 'N/A')}")
            st.markdown(f"**Task:** {model_info.get('task', 'N/A')}")
            st.markdown(f"**Version:** {model_info.get('version', 'N/A')}")
            st.markdown(f"**Framework:** {model_info.get('framework', 'PyTorch Lightning')}")
        
        with col2:
            st.markdown("#### Metadata")
            created = model_info.get('creation_timestamp')
            if created:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created)
                st.markdown(f"**Created:** {created.strftime('%Y-%m-%d %H:%M')}")
            
            updated = model_info.get('last_updated_timestamp')
            if updated:
                if isinstance(updated, str):
                    updated = datetime.fromisoformat(updated)
                st.markdown(f"**Updated:** {updated.strftime('%Y-%m-%d %H:%M')}")
            
            st.markdown(f"**Stage:** {model_info.get('stage', 'None')}")
        
        # Description
        if 'description' in model_info and model_info['description']:
            st.markdown("#### Description")
            st.markdown(model_info['description'])
        
        # Performance metrics
        if 'metrics' in model_info:
            st.markdown("#### Performance Metrics")
            MetricsDisplay.display_metrics_grid(model_info['metrics'], columns=3)


"""
Metrics Display Component
Display training and evaluation metrics following the design system.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from components.theme import metric_card, section_title, status_badge


class MetricsDisplay:
    """Component for displaying metrics."""

    @staticmethod
    def display_metrics_grid(
        metrics: Dict[str, float],
        columns: int = 4,
        format_str: str = "{:.4f}",
    ):
        if not metrics:
            st.info("No metrics available")
            return

        metric_items = list(metrics.items())
        cols = st.columns(columns)

        for idx, (name, value) in enumerate(metric_items):
            col_idx = idx % columns
            with cols[col_idx]:
                display_name = name.replace("_", " ").title()
                if isinstance(value, (int, float)):
                    display_value = format_str.format(value)
                else:
                    display_value = str(value)
                st.metric(label=display_name, value=display_value)

    @staticmethod
    def display_training_summary(
        run_info: Dict[str, Any],
        show_params: bool = True,
        show_metrics: bool = True,
    ):
        section_title("Training Run Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="raised-card">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#4E566A;'
                f'text-transform:uppercase;letter-spacing:0.08em;">RUN ID</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:12px;color:#EDF0F7;'
                f'margin-top:4px;">{run_info.get("run_id", "N/A")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            status = run_info.get("status", "UNKNOWN")
            badge = status_badge(status)
            st.markdown(
                f'<div class="raised-card">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#4E566A;'
                f'text-transform:uppercase;letter-spacing:0.08em;">STATUS</div>'
                f'<div style="margin-top:4px;">{badge}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col3:
            start_time = run_info.get("start_time")
            end_time = run_info.get("end_time")
            if start_time and end_time:
                duration = (end_time - start_time).total_seconds()
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                dur_str = f"{hours}h {minutes}m"
            else:
                dur_str = "N/A"
            st.markdown(
                f'<div class="raised-card">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#4E566A;'
                f'text-transform:uppercase;letter-spacing:0.08em;">DURATION</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:24px;font-weight:700;'
                f'color:#EDF0F7;margin-top:4px;">{dur_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if show_params and "params" in run_info:
            with st.expander("Parameters", expanded=False):
                params_df = pd.DataFrame([
                    {"Parameter": k, "Value": v}
                    for k, v in run_info["params"].items()
                ])
                st.dataframe(params_df, use_container_width=True, hide_index=True)

        if show_metrics and "metrics" in run_info:
            with st.expander("Metrics", expanded=True):
                MetricsDisplay.display_metrics_grid(run_info["metrics"])

        if "tags" in run_info:
            with st.expander("Tags", expanded=False):
                tags_df = pd.DataFrame([
                    {"Tag": k, "Value": v}
                    for k, v in run_info["tags"].items()
                ])
                st.dataframe(tags_df, use_container_width=True, hide_index=True)

    @staticmethod
    def display_comparison_table(
        runs: List[Dict[str, Any]],
        metrics_to_compare: List[str],
        params_to_compare: Optional[List[str]] = None,
    ):
        if not runs:
            st.info("No runs to compare")
            return

        section_title("Model Comparison")

        comparison_data = []
        for run in runs:
            row = {
                "Run Name": run.get("run_name", "Unnamed"),
                "Run ID": run.get("run_id", "")[:8],
                "Status": run.get("status", "UNKNOWN"),
            }
            for metric in metrics_to_compare:
                value = run.get("metrics", {}).get(metric, None)
                row[metric] = f"{value:.4f}" if value is not None else "N/A"
            if params_to_compare:
                for param in params_to_compare:
                    row[param] = run.get("params", {}).get(param, "N/A")
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if metrics_to_compare:
            section_title("Best Metrics")
            best_cols = st.columns(len(metrics_to_compare))
            for idx, metric in enumerate(metrics_to_compare):
                with best_cols[idx]:
                    metric_values = [
                        run.get("metrics", {}).get(metric)
                        for run in runs
                        if run.get("metrics", {}).get(metric) is not None
                    ]
                    if metric_values:
                        if "loss" in metric.lower() or "error" in metric.lower():
                            best_value = min(metric_values)
                        else:
                            best_value = max(metric_values)
                        metric_card(metric, f"{best_value:.4f}")

    @staticmethod
    def display_evaluation_results(
        task: str,
        metrics: Dict[str, float],
        detailed_metrics: Optional[Dict[str, Any]] = None,
    ):
        section_title("Evaluation Results")
        MetricsDisplay.display_metrics_grid(metrics)

        if detailed_metrics:
            task_detail_map = {
                "classification": ("per_class_accuracy", "Per-Class Accuracy", "Accuracy"),
                "detection": ("per_class_ap", "Per-Class Average Precision", "AP"),
                "segmentation": ("per_class_iou", "Per-Class IoU", "IoU"),
            }
            key, title, col_name = task_detail_map.get(task, (None, None, None))
            if key and key in detailed_metrics:
                with st.expander(title, expanded=False):
                    per_class = detailed_metrics[key]
                    df = pd.DataFrame([
                        {"Class": k, col_name: f"{v:.4f}"}
                        for k, v in per_class.items()
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)

    @staticmethod
    def display_progress_bar(
        current: int, total: int,
        label: str = "Progress",
        show_percentage: bool = True,
    ):
        progress = current / total if total > 0 else 0
        pct_text = f" ({progress:.1%})" if show_percentage else ""
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
            f'color:#8A91A8;margin-bottom:4px;">{label}: {current}/{total}{pct_text}</div>',
            unsafe_allow_html=True,
        )
        st.progress(progress)

    @staticmethod
    def display_status_timeline(events: List[Dict[str, Any]]):
        section_title("Status Timeline")
        for event in events:
            timestamp = event.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            status = event.get("status", "INFO")
            message = event.get("message", "")
            badge = status_badge(status)
            ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#4E566A;">'
                f'{ts_str}</span>'
                f'{badge}'
                f'<span style="font-family:Figtree,sans-serif;font-size:13px;color:#8A91A8;">'
                f'{message}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    @staticmethod
    def display_model_card(model_info: Dict[str, Any]):
        section_title("Model Card")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div style="font-family:Figtree,sans-serif;font-size:14px;color:#8A91A8;">'
                f'<strong style="color:#EDF0F7;">Name:</strong> {model_info.get("name", "N/A")}<br/>'
                f'<strong style="color:#EDF0F7;">Task:</strong> {model_info.get("task", "N/A")}<br/>'
                f'<strong style="color:#EDF0F7;">Version:</strong> {model_info.get("version", "N/A")}<br/>'
                f'<strong style="color:#EDF0F7;">Framework:</strong> {model_info.get("framework", "HF Trainer")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            created = model_info.get("creation_timestamp")
            if created:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created)
                created_str = created.strftime("%Y-%m-%d %H:%M")
            else:
                created_str = "N/A"
            updated = model_info.get("last_updated_timestamp")
            if updated:
                if isinstance(updated, str):
                    updated = datetime.fromisoformat(updated)
                updated_str = updated.strftime("%Y-%m-%d %H:%M")
            else:
                updated_str = "N/A"
            st.markdown(
                f'<div style="font-family:Figtree,sans-serif;font-size:14px;color:#8A91A8;">'
                f'<strong style="color:#EDF0F7;">Created:</strong> '
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;">{created_str}</span><br/>'
                f'<strong style="color:#EDF0F7;">Updated:</strong> '
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;">{updated_str}</span><br/>'
                f'<strong style="color:#EDF0F7;">Stage:</strong> {model_info.get("stage", "None")}'
                f'</div>',
                unsafe_allow_html=True,
            )

        if "description" in model_info and model_info["description"]:
            st.markdown(
                f'<div style="margin-top:12px;font-family:Figtree,sans-serif;font-size:14px;'
                f'color:#8A91A8;line-height:1.6;">{model_info["description"]}</div>',
                unsafe_allow_html=True,
            )

        if "metrics" in model_info:
            st.markdown("")
            MetricsDisplay.display_metrics_grid(model_info["metrics"], columns=3)

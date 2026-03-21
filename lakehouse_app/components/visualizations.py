"""
Visualization Helper
Charts and visualizations — dark theme Plotly layout matching the design system.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import pandas as pd

# Design-system Plotly template
_DS_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Figtree, sans-serif", color="#8A91A8", size=12),
    title_font=dict(family="Syne, sans-serif", color="#EDF0F7", size=16),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    legend=dict(font=dict(family="IBM Plex Mono, monospace", size=11, color="#8A91A8")),
    margin=dict(l=40, r=20, t=50, b=40),
)

ACCENT = "#00C2A8"
ACCENT_WARM = "#F4A742"
ACCENT_ALERT = "#F25C5C"
ACCENT_INFO = "#5B8AF5"


def _apply_ds(fig: go.Figure) -> go.Figure:
    """Apply design-system layout to a Plotly figure."""
    fig.update_layout(**_DS_LAYOUT)
    return fig


class VisualizationHelper:
    """Helper class for creating visualizations."""

    @staticmethod
    def training_metrics_chart(
        metrics_history: List[Dict[str, Any]],
        metric_name: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        if not metrics_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#8A91A8"),
            )
            return _apply_ds(fig)

        df = pd.DataFrame(metrics_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["step"], y=df["value"],
            mode="lines+markers",
            name=metric_name,
            line=dict(color=ACCENT, width=2),
            marker=dict(size=5),
        ))
        fig.update_layout(
            title=title or f"{metric_name} over Training",
            xaxis_title="Step", yaxis_title=metric_name,
            hovermode="x unified",
        )
        return _apply_ds(fig)

    @staticmethod
    def multi_metric_chart(
        train_metrics: List[Dict[str, Any]],
        val_metrics: List[Dict[str, Any]],
        metric_name: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        fig = go.Figure()
        if train_metrics:
            df_train = pd.DataFrame(train_metrics)
            fig.add_trace(go.Scatter(
                x=df_train["step"], y=df_train["value"],
                mode="lines", name=f"Train {metric_name}",
                line=dict(color=ACCENT_INFO, width=2),
            ))
        if val_metrics:
            df_val = pd.DataFrame(val_metrics)
            fig.add_trace(go.Scatter(
                x=df_val["step"], y=df_val["value"],
                mode="lines+markers", name=f"Val {metric_name}",
                line=dict(color=ACCENT_WARM, width=2),
                marker=dict(size=6),
            ))
        fig.update_layout(
            title=title or f"{metric_name}: Training vs Validation",
            xaxis_title="Step", yaxis_title=metric_name,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return _apply_ds(fig)

    @staticmethod
    def class_distribution_chart(class_counts: Dict[str, int], title: str = "Class Distribution") -> go.Figure:
        df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
        df = df.sort_values("Count", ascending=False)
        fig = go.Figure(go.Bar(
            x=df["Class"], y=df["Count"],
            marker_color=ACCENT,
            text=df["Count"], textposition="outside",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Class", yaxis_title="Count",
            xaxis_tickangle=-45,
        )
        return _apply_ds(fig)

    @staticmethod
    def confusion_matrix_chart(confusion_matrix: List[List[int]], class_names: List[str]) -> go.Figure:
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=class_names, y=class_names,
            colorscale=[[0, "#141720"], [1, ACCENT]],
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 10, "family": "IBM Plex Mono"},
            colorbar=dict(title="Count"),
        ))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted", yaxis_title="Actual",
            width=800, height=800,
        )
        return _apply_ds(fig)

    @staticmethod
    def model_comparison_chart(runs: List[Dict[str, Any]], metric_name: str) -> go.Figure:
        run_names = [run.get("run_name", f"Run {i}") for i, run in enumerate(runs)]
        metric_values = [run["metrics"].get(metric_name, 0) for run in runs]
        fig = go.Figure(data=[go.Bar(
            x=run_names, y=metric_values,
            marker_color=ACCENT,
            text=metric_values,
            texttemplate="%{text:.4f}",
            textposition="outside",
        )])
        fig.update_layout(
            title=f"Model Comparison: {metric_name}",
            xaxis_title="Run", yaxis_title=metric_name,
        )
        return _apply_ds(fig)

    @staticmethod
    def training_progress_gauge(
        current_epoch: int, total_epochs: int,
        metric_value: float, metric_name: str,
    ) -> go.Figure:
        progress_percent = (current_epoch / total_epochs) * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=progress_percent,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": (
                    f"Training Progress<br>"
                    f"<span style='font-size:0.8em;font-family:IBM Plex Mono'>"
                    f"{metric_name}: {metric_value:.4f}</span>"
                ),
            },
            delta={"reference": 100},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": ACCENT},
                "steps": [
                    {"range": [0, 33], "color": "#1C2030"},
                    {"range": [33, 66], "color": "#232840"},
                    {"range": [66, 100], "color": "#1C2030"},
                ],
                "threshold": {
                    "line": {"color": ACCENT_ALERT, "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=80, b=20))
        return _apply_ds(fig)

    @staticmethod
    def resource_usage_chart(training_history: List[Dict[str, Any]]) -> go.Figure:
        if not training_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No training history available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#8A91A8"),
            )
            return _apply_ds(fig)

        df = pd.DataFrame(training_history)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = go.Figure()
        if "duration_seconds" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"] if "timestamp" in df.columns else list(range(len(df))),
                y=df["duration_seconds"] / 3600,
                mode="lines+markers",
                name="Duration (hours)",
                line=dict(color=ACCENT, width=2),
                marker=dict(size=6),
            ))
        fig.update_layout(
            title="Training Duration Over Time",
            xaxis_title="Date", yaxis_title="Duration (hours)",
            hovermode="x unified",
        )
        return _apply_ds(fig)

    @staticmethod
    def metrics_summary_table(metrics: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([
            {"Metric": name, "Value": f"{value:.4f}"}
            for name, value in metrics.items()
        ])

    @staticmethod
    def display_metric_card(label: str, value, delta=None, icon: str = ""):
        if delta is not None:
            st.metric(
                label=f"{icon} {label}" if icon else label,
                value=value,
                delta=f"{delta:.2%}" if isinstance(delta, (int, float)) else str(delta),
            )
        else:
            st.metric(label=f"{icon} {label}" if icon else label, value=value)

    @staticmethod
    def display_status_badge(status: str, status_map=None):
        default_map = {
            "SUCCESS": "RUNNING", "RUNNING": "QUEUED",
            "FAILED": "FAILED", "PENDING": "QUEUED", "CANCELLED": "FAILED",
        }
        mapped = (status_map or default_map).get(status.upper(), status)
        from components.theme import status_badge
        st.markdown(status_badge(mapped), unsafe_allow_html=True)

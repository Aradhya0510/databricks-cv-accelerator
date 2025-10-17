"""
Visualization Helper
Charts and visualizations for the app
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime


class VisualizationHelper:
    """Helper class for creating visualizations."""
    
    @staticmethod
    def training_metrics_chart(
        metrics_history: List[Dict[str, Any]],
        metric_name: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create a line chart for training metrics.
        
        Args:
            metrics_history: List of metric dictionaries with step, value, timestamp
            metric_name: Name of the metric
            title: Optional chart title
            
        Returns:
            Plotly figure
        """
        if not metrics_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        df = pd.DataFrame(metrics_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['step'],
            y=df['value'],
            mode='lines+markers',
            name=metric_name,
            line=dict(color='#FF3621', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=title or f"{metric_name} over Training",
            xaxis_title="Step",
            yaxis_title=metric_name,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def multi_metric_chart(
        train_metrics: List[Dict[str, Any]],
        val_metrics: List[Dict[str, Any]],
        metric_name: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create a chart comparing training and validation metrics.
        
        Args:
            train_metrics: Training metric history
            val_metrics: Validation metric history
            metric_name: Name of the metric
            title: Optional chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if train_metrics:
            df_train = pd.DataFrame(train_metrics)
            fig.add_trace(go.Scatter(
                x=df_train['step'],
                y=df_train['value'],
                mode='lines',
                name=f'Train {metric_name}',
                line=dict(color='#1f77b4', width=2)
            ))
        
        if val_metrics:
            df_val = pd.DataFrame(val_metrics)
            fig.add_trace(go.Scatter(
                x=df_val['step'],
                y=df_val['value'],
                mode='lines+markers',
                name=f'Val {metric_name}',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=title or f"{metric_name}: Training vs Validation",
            xaxis_title="Step",
            yaxis_title=metric_name,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    @staticmethod
    def class_distribution_chart(class_counts: Dict[str, int], title: str = "Class Distribution") -> go.Figure:
        """
        Create a bar chart for class distribution.
        
        Args:
            class_counts: Dictionary mapping class names to counts
            title: Chart title
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
        df = df.sort_values('Count', ascending=False)
        
        fig = px.bar(
            df,
            x='Class',
            y='Count',
            title=title,
            color='Count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title="Class",
            yaxis_title="Number of Samples",
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def confusion_matrix_chart(confusion_matrix: List[List[int]], class_names: List[str]) -> go.Figure:
        """
        Create a confusion matrix heatmap.
        
        Args:
            confusion_matrix: 2D list representing confusion matrix
            class_names: List of class names
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Count")
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template='plotly_white',
            width=800,
            height=800
        )
        
        return fig
    
    @staticmethod
    def model_comparison_chart(runs: List[Dict[str, Any]], metric_name: str) -> go.Figure:
        """
        Create a bar chart comparing multiple model runs.
        
        Args:
            runs: List of run dictionaries with metrics
            metric_name: Metric to compare
            
        Returns:
            Plotly figure
        """
        run_names = [run.get('run_name', f"Run {i}") for i, run in enumerate(runs)]
        metric_values = [run['metrics'].get(metric_name, 0) for run in runs]
        
        fig = go.Figure(data=[
            go.Bar(
                x=run_names,
                y=metric_values,
                marker_color='#FF3621',
                text=metric_values,
                texttemplate='%{text:.4f}',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f"Model Comparison: {metric_name}",
            xaxis_title="Run",
            yaxis_title=metric_name,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def training_progress_gauge(current_epoch: int, total_epochs: int, metric_value: float, metric_name: str) -> go.Figure:
        """
        Create a gauge chart for training progress.
        
        Args:
            current_epoch: Current epoch number
            total_epochs: Total number of epochs
            metric_value: Current metric value
            metric_name: Name of the metric
            
        Returns:
            Plotly figure
        """
        progress_percent = (current_epoch / total_epochs) * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=progress_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Training Progress<br><span style='font-size:0.8em'>{metric_name}: {metric_value:.4f}</span>"},
            delta={'reference': 100},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#FF3621"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        return fig
    
    @staticmethod
    def resource_usage_chart(training_history: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a chart showing resource usage over time.
        
        Args:
            training_history: List of training run dictionaries
            
        Returns:
            Plotly figure
        """
        if not training_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No training history available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        df = pd.DataFrame(training_history)
        
        # Extract timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        # Add duration line if available
        if 'duration_seconds' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'] if 'timestamp' in df.columns else list(range(len(df))),
                y=df['duration_seconds'] / 3600,  # Convert to hours
                mode='lines+markers',
                name='Training Duration (hours)',
                line=dict(color='#FF3621', width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Training Duration Over Time",
            xaxis_title="Date",
            yaxis_title="Duration (hours)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def metrics_summary_table(metrics: Dict[str, float]) -> pd.DataFrame:
        """
        Create a formatted DataFrame for metrics display.
        
        Args:
            metrics: Dictionary of metric names to values
            
        Returns:
            Formatted pandas DataFrame
        """
        df = pd.DataFrame([
            {"Metric": name, "Value": f"{value:.4f}"}
            for name, value in metrics.items()
        ])
        return df
    
    @staticmethod
    def display_metric_card(label: str, value: Any, delta: Optional[float] = None, icon: str = "📊"):
        """
        Display a metric card with styling.
        
        Args:
            label: Metric label
            value: Metric value
            delta: Optional delta/change value
            icon: Optional icon emoji
        """
        if delta is not None:
            st.metric(
                label=f"{icon} {label}",
                value=value,
                delta=f"{delta:.2%}" if isinstance(delta, (int, float)) else str(delta)
            )
        else:
            st.metric(
                label=f"{icon} {label}",
                value=value
            )
    
    @staticmethod
    def display_status_badge(status: str, status_map: Optional[Dict[str, str]] = None):
        """
        Display a colored status badge.
        
        Args:
            status: Status text
            status_map: Optional mapping of status to colors
        """
        default_map = {
            "SUCCESS": "🟢",
            "RUNNING": "🟡",
            "FAILED": "🔴",
            "PENDING": "⚪",
            "CANCELLED": "⚫",
        }
        
        status_map = status_map or default_map
        icon = status_map.get(status.upper(), "⚪")
        
        st.markdown(f"{icon} **{status}**")


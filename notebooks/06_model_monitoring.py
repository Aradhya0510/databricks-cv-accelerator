# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Monitoring
# MAGIC 
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Set up monitoring for deployed models
# MAGIC 2. Track model performance metrics
# MAGIC 3. Monitor data drift and model drift
# MAGIC 4. Set up alerts for model degradation
# MAGIC 5. Generate monitoring reports

# COMMAND ----------

%pip install -r "../requirements.txt"
dbutils.library.restartPython()

# COMMAND ----------

%run ./01_data_preparation
%run ./02_model_training
%run ./04_model_evaluation
%run ./05_model_registration_deployment

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import yaml
import mlflow
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Add the project root to Python path
project_root = "/Volumes/<catalog>/<schema>/<volume>/<path>/<file_name>"
sys.path.append(project_root)

# Import project modules
from src.utils.logging import setup_logger
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule
from src.tasks.classification.model import ClassificationModel
from src.tasks.classification.data import ClassificationDataModule
from src.tasks.semantic_segmentation.model import SemanticSegmentationModel
from src.tasks.semantic_segmentation.data import SemanticSegmentationDataModule
from src.tasks.instance_segmentation.model import InstanceSegmentationModel
from src.tasks.instance_segmentation.data import InstanceSegmentationDataModule
from src.tasks.panoptic_segmentation.model import PanopticSegmentationModel
from src.tasks.panoptic_segmentation.data import PanopticSegmentationDataModule

# COMMAND ----------

# DBTITLE 1,Initialize Logging
volume_path = os.getenv("UNITY_CATALOG_VOLUME", project_root)
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="model_monitoring",
    log_file=f"{log_dir}/monitoring.log"
)

# COMMAND ----------

# DBTITLE 1,Load Configuration and Model
def load_task_config(task: str):
    """Load task-specific configuration."""
    config_path = f"{volume_path}/configs/{task}_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_model_class(task: str):
    """Get the appropriate model class based on task."""
    model_classes = {
        'detection': DetectionModel,
        'classification': ClassificationModel,
        'semantic_segmentation': SemanticSegmentationModel,
        'instance_segmentation': InstanceSegmentationModel,
        'panoptic_segmentation': PanopticSegmentationModel
    }
    
    if task not in model_classes:
        raise ValueError(f"Unsupported task: {task}")
    
    return model_classes[task]

def load_deployed_model(task: str):
    """Load the deployed model for monitoring."""
    config = load_task_config(task)
    
    # Load model from MLflow
    model_name = config['model']['model_name']
    model_uri = f"models:/{task}_{model_name}/Production"
    model = mlflow.pytorch.load_model(model_uri)
    
    model.eval()
    return model, config

# COMMAND ----------

# DBTITLE 1,Performance Monitoring
class PerformanceMonitor:
    """Monitor model performance metrics."""
    
    def __init__(self, task: str, config: dict, window_size: int = 100):
        self.task = task
        self.config = config
        self.window_size = window_size
        self.metrics_history = []
        
    def calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics for the given predictions."""
        if self.task == 'detection':
            return self._calculate_detection_metrics(predictions, ground_truth)
        elif self.task == 'classification':
            return self._calculate_classification_metrics(predictions, ground_truth)
        elif self.task == 'semantic_segmentation':
            return self._calculate_semantic_segmentation_metrics(predictions, ground_truth)
        elif self.task == 'instance_segmentation':
            return self._calculate_instance_segmentation_metrics(predictions, ground_truth)
        elif self.task == 'panoptic_segmentation':
            return self._calculate_panoptic_segmentation_metrics(predictions, ground_truth)
        else:
            raise ValueError(f"Unsupported task: {self.task}")
    
    def _calculate_detection_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate detection-specific metrics."""
        # Placeholder implementation
        return {
            'map': 0.75,
            'avg_confidence': 0.82,
            'num_detections': 15.5
        }
    
    def _calculate_classification_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate classification-specific metrics."""
        # Placeholder implementation
        return {
            'accuracy': 0.88,
            'avg_confidence': 0.85,
            'precision': 0.87,
            'recall': 0.86
        }
    
    def _calculate_semantic_segmentation_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate semantic segmentation-specific metrics."""
        # Placeholder implementation
        return {
            'iou': 0.72,
            'pixel_accuracy': 0.89,
            'avg_confidence': 0.78
        }
    
    def _calculate_instance_segmentation_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate instance segmentation-specific metrics."""
        # Placeholder implementation
        return {
            'map': 0.68,
            'mask_ap': 0.65,
            'avg_confidence': 0.79
        }
    
    def _calculate_panoptic_segmentation_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate panoptic segmentation-specific metrics."""
        # Placeholder implementation
        return {
            'pq': 0.71,
            'sq': 0.78,
            'rq': 0.65
        }
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics history."""
        timestamp = datetime.now()
        metrics_with_timestamp = {
            'timestamp': timestamp,
            **metrics
        }
        
        self.metrics_history.append(metrics_with_timestamp)
        
        # Keep only the last window_size entries
        if len(self.metrics_history) > self.window_size:
            self.metrics_history = self.metrics_history[-self.window_size:]
    
    def get_trends(self) -> Dict[str, float]:
        """Calculate trends in metrics over time."""
        if len(self.metrics_history) < 2:
            return {}
        
        trends = {}
        for key in self.metrics_history[0].keys():
            if key != 'timestamp':
                values = [entry[key] for entry in self.metrics_history]
                if len(values) >= 2:
                    # Simple linear trend calculation
                    trend = (values[-1] - values[0]) / len(values)
                    trends[f"{key}_trend"] = trend
        
        return trends

# COMMAND ----------

# DBTITLE 1,Drift Detection
class DriftDetector:
    """Detect data and model drift."""
    
    def __init__(self, reference_data_path: str, window_size: int = 1000, drift_threshold: float = 0.1):
        self.reference_data_path = reference_data_path
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_stats = self._load_reference_stats()
    
    def _load_reference_stats(self) -> Dict[str, Any]:
        """Load reference data statistics."""
        # Placeholder implementation
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'feature_distributions': {}
        }
    
    def calculate_drift(self, recent_data: List[Dict]) -> Dict[str, float]:
        """Calculate drift metrics for recent data."""
        # Placeholder implementation
        return {
            'feature_drift': [0.05, 0.03, 0.07],
            'distribution_drift': 0.08,
            'concept_drift': 0.12
        }
    
    def detect_drift(self, drift_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Detect if drift exceeds thresholds."""
        alerts = {}
        
        # Check feature drift
        feature_drift = drift_metrics.get('feature_drift', [])
        if isinstance(feature_drift, list):
            alerts['feature_drift'] = any(drift > self.drift_threshold for drift in feature_drift)
        else:
            alerts['feature_drift'] = feature_drift > self.drift_threshold
        
        # Check distribution drift
        distribution_drift = drift_metrics.get('distribution_drift', 0)
        alerts['distribution_drift'] = distribution_drift > self.drift_threshold
        
        # Check concept drift
        concept_drift = drift_metrics.get('concept_drift', 0)
        alerts['concept_drift'] = concept_drift > self.drift_threshold
        
        return alerts

# COMMAND ----------

# DBTITLE 1,Alert System
class AlertSystem:
    """System for generating and managing alerts."""
    
    def __init__(self, alert_thresholds: Dict[str, Dict[str, float]]):
        self.alert_thresholds = alert_thresholds
        self.alerts_history = []
    
    def check_performance_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for performance-based alerts."""
        alerts = []
        performance_thresholds = self.alert_thresholds.get('performance', {})
        
        for metric, threshold in performance_thresholds.items():
            if metric in metrics:
                current_value = metrics[metric]
                if current_value < threshold:
                    alerts.append({
                        'type': 'performance',
                        'metric': metric,
                        'value': current_value,
                        'threshold': threshold,
                        'timestamp': datetime.now(),
                        'severity': 'high' if current_value < threshold * 0.8 else 'medium'
                    })
        
        return alerts
    
    def check_drift_alerts(self, drift_alerts: Dict[str, bool]) -> List[Dict[str, Any]]:
        """Check for drift-based alerts."""
        alerts = []
        
        for drift_type, is_alert in drift_alerts.items():
            if is_alert:
                alerts.append({
                    'type': 'drift',
                    'drift_type': drift_type,
                    'timestamp': datetime.now(),
                    'severity': 'high'
                })
        
        return alerts
    
    def log_alerts(self, alerts: List[Dict[str, Any]]):
        """Log alerts to history."""
        self.alerts_history.extend(alerts)
        
        # Log to file
        for alert in alerts:
            logger.warning(f"Alert: {alert}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts_history if alert['timestamp'] > cutoff_time]

# COMMAND ----------

# DBTITLE 1,Monitoring Dashboard
class MonitoringDashboard:
    """Generate monitoring dashboard and reports."""
    
    def __init__(self, task: str, performance_monitor: PerformanceMonitor, 
                 drift_detector: DriftDetector, alert_system: AlertSystem):
        self.task = task
        self.performance_monitor = performance_monitor
        self.drift_detector = drift_detector
        self.alert_system = alert_system
    
    def generate_performance_plots(self):
        """Generate performance monitoring plots."""
        if not self.performance_monitor.metrics_history:
            print("No performance metrics available")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.performance_monitor.metrics_history)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.task.title()} Model Performance Monitoring', fontsize=16)
        
        # Plot different metrics
        metrics_to_plot = [col for col in df.columns if col != 'timestamp']
        
        for i, metric in enumerate(metrics_to_plot[:4]):  # Plot up to 4 metrics
            row, col = i // 2, i % 2
            axes[row, col].plot(df['timestamp'], df[metric])
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel(metric.replace("_", " ").title())
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_drift_plots(self, drift_metrics: Dict[str, float]):
        """Generate drift monitoring plots."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{self.task.title()} Model Drift Monitoring', fontsize=16)
        
        # Feature drift
        feature_drift = drift_metrics.get('feature_drift', [])
        if feature_drift:
            axes[0].bar(range(len(feature_drift)), feature_drift)
            axes[0].set_title('Feature Drift')
            axes[0].set_xlabel('Feature Index')
            axes[0].set_ylabel('Drift Score')
            axes[0].axhline(y=self.drift_detector.drift_threshold, color='r', linestyle='--', label='Threshold')
            axes[0].legend()
        
        # Distribution drift
        distribution_drift = drift_metrics.get('distribution_drift', 0)
        axes[1].bar(['Distribution Drift'], [distribution_drift])
        axes[1].set_title('Distribution Drift')
        axes[1].set_ylabel('Drift Score')
        axes[1].axhline(y=self.drift_detector.drift_threshold, color='r', linestyle='--', label='Threshold')
        axes[1].legend()
        
        # Concept drift
        concept_drift = drift_metrics.get('concept_drift', 0)
        axes[2].bar(['Concept Drift'], [concept_drift])
        axes[2].set_title('Concept Drift')
        axes[2].set_ylabel('Drift Score')
        axes[2].axhline(y=self.drift_detector.drift_threshold, color='r', linestyle='--', label='Threshold')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_alert_summary(self):
        """Generate alert summary."""
        recent_alerts = self.alert_system.get_recent_alerts()
        
        if not recent_alerts:
            print("No recent alerts")
            return
        
        # Count alerts by type and severity
        alert_counts = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            severity = alert['severity']
            key = f"{alert_type}_{severity}"
            alert_counts[key] = alert_counts.get(key, 0) + 1
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        alert_types = list(alert_counts.keys())
        counts = list(alert_counts.values())
        
        colors = ['red' if 'high' in alert_type else 'orange' if 'medium' in alert_type else 'yellow' 
                 for alert_type in alert_types]
        
        bars = ax.bar(alert_types, counts, color=colors)
        ax.set_title('Recent Alerts Summary (Last 24 Hours)')
        ax.set_xlabel('Alert Type and Severity')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def save_monitoring_report(self, report_path: str):
        """Save comprehensive monitoring report."""
        report = {
            'task': self.task,
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.performance_monitor.metrics_history[-1] if self.performance_monitor.metrics_history else {},
            'performance_trends': self.performance_monitor.get_trends(),
            'recent_alerts': self.alert_system.get_recent_alerts(),
            'alert_summary': {
                'total_alerts_24h': len(self.alert_system.get_recent_alerts()),
                'high_severity_alerts': len([a for a in self.alert_system.get_recent_alerts() if a['severity'] == 'high'])
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved to {report_path}")

# COMMAND ----------

# DBTITLE 1,Main Monitoring Function
def setup_model_monitoring(task: str):
    """Set up comprehensive model monitoring."""
    # Load model and configuration
    model, config = load_deployed_model(task)
    
    # Define alert thresholds
    alert_thresholds = {
        'performance': {
            'map': 0.5,
            'accuracy': 0.8,
            'iou': 0.6,
            'avg_confidence': 0.7
        },
        'drift': {
            'feature_drift_threshold': 0.2,
            'distribution_drift_threshold': 0.15,
            'concept_drift_threshold': 0.1
        }
    }
    
    # Initialize monitoring components
    performance_monitor = PerformanceMonitor(task, config)
    drift_detector = DriftDetector(
        reference_data_path=config['data']['data_path'],
        window_size=1000,
        drift_threshold=0.1
    )
    alert_system = AlertSystem(alert_thresholds)
    
    # Initialize dashboard
    dashboard = MonitoringDashboard(task, performance_monitor, drift_detector, alert_system)
    
    return {
        'model': model,
        'config': config,
        'performance_monitor': performance_monitor,
        'drift_detector': drift_detector,
        'alert_system': alert_system,
        'dashboard': dashboard
    }

def run_monitoring_cycle(monitoring_components: Dict[str, Any], recent_data: List[Dict], 
                        recent_predictions: List[Dict], ground_truth: List[Dict]):
    """Run a single monitoring cycle."""
    performance_monitor = monitoring_components['performance_monitor']
    drift_detector = monitoring_components['drift_detector']
    alert_system = monitoring_components['alert_system']
    dashboard = monitoring_components['dashboard']
    
    # Calculate performance metrics
    performance_metrics = performance_monitor.calculate_metrics(recent_predictions, ground_truth)
    performance_monitor.update_metrics(performance_metrics)
    
    # Calculate drift metrics
    drift_metrics = drift_detector.calculate_drift(recent_data)
    drift_alerts = drift_detector.detect_drift(drift_metrics)
    
    # Check for alerts
    performance_alerts = alert_system.check_performance_alerts(performance_metrics)
    drift_alert_list = alert_system.check_drift_alerts(drift_alerts)
    
    # Log all alerts
    all_alerts = performance_alerts + drift_alert_list
    alert_system.log_alerts(all_alerts)
    
    # Generate dashboard
    dashboard.generate_performance_plots()
    dashboard.generate_drift_plots(drift_metrics)
    dashboard.generate_alert_summary()
    
    return {
        'performance_metrics': performance_metrics,
        'drift_metrics': drift_metrics,
        'alerts': all_alerts
    }

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Set up monitoring for detection model
task = "detection"

# Set up monitoring
monitoring_components = setup_model_monitoring(task)

# Simulate monitoring cycle with sample data
sample_data = [{'image': np.random.rand(3, 640, 640)} for _ in range(100)]
sample_predictions = [{'boxes': np.random.rand(5, 4), 'scores': np.random.rand(5)} for _ in range(100)]
sample_ground_truth = [{'boxes': np.random.rand(5, 4), 'labels': np.random.randint(0, 10, 5)} for _ in range(100)]

# Run monitoring cycle
monitoring_results = run_monitoring_cycle(
    monitoring_components, 
    sample_data, 
    sample_predictions, 
    sample_ground_truth
)

# Save monitoring report
report_path = f"{volume_path}/results/{task}_monitoring_report.json"
monitoring_components['dashboard'].save_monitoring_report(report_path)

# Display results
print("Monitoring Results:")
print(f"Performance Metrics: {monitoring_results['performance_metrics']}")
print(f"Drift Metrics: {monitoring_results['drift_metrics']}")
print(f"Number of Alerts: {len(monitoring_results['alerts'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Set up automated monitoring schedules
# MAGIC 2. Configure alert notifications
# MAGIC 3. Implement model retraining triggers
# MAGIC 4. Set up performance dashboards 
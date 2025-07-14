# Databricks notebook source
# MAGIC %md
# MAGIC # 04. Model Evaluation and Analysis
# MAGIC 
# MAGIC This notebook demonstrates comprehensive evaluation of the trained DETR model:
# MAGIC 1. Load the trained model and checkpoint
# MAGIC 2. Run evaluation on the test set
# MAGIC 3. Analyze detailed metrics (mAP, precision, recall, F1)
# MAGIC 4. Visualize predictions and analyze errors
# MAGIC 5. Measure inference speed and performance
# MAGIC 6. Generate evaluation reports and recommendations
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **Model Evaluation Process:**
# MAGIC 1. **Model Loading**: Load trained model from checkpoint or MLflow
# MAGIC 2. **Data Preparation**: Set up test dataset with proper preprocessing
# MAGIC 3. **Evaluation Metrics**: Calculate comprehensive detection metrics
# MAGIC 4. **Visualization**: Generate prediction visualizations
# MAGIC 5. **Performance Analysis**: Measure speed and resource usage
# MAGIC 6. **Error Analysis**: Identify and analyze failure cases
# MAGIC 
# MAGIC ### Key Evaluation Metrics:
# MAGIC - **mAP (mean Average Precision)**: Primary detection metric
# MAGIC - **Precision/Recall**: Per-class performance analysis
# MAGIC - **F1 Score**: Balanced precision-recall metric
# MAGIC - **Inference Speed**: FPS and latency measurements
# MAGIC - **Memory Usage**: GPU/CPU resource consumption
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 
# MAGIC 1. **Model Loading**: Load best checkpoint or MLflow model
# MAGIC 2. **Test Evaluation**: Run comprehensive evaluation on test set
# MAGIC 3. **Metric Analysis**: Calculate and analyze all detection metrics
# MAGIC 4. **Visualization**: Create prediction visualizations
# MAGIC 5. **Performance Testing**: Measure inference speed and efficiency
# MAGIC 6. **Report Generation**: Create detailed evaluation report
# MAGIC 
# MAGIC ---

# COMMAND ----------

# (Databricks only) Install requirements and restart Python if running interactively
# %pip install -r "../requirements.txt"
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import Dependencies and Load Configuration

# COMMAND ----------

import sys
import os
import yaml
import torch
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Any
import time
from collections import defaultdict
from pathlib import Path
import json

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')

from config import load_config, get_default_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from tasks.detection.evaluate import DetectionEvaluator
from tasks.detection.adapters import get_input_adapter
from utils.logging import create_databricks_logger
from utils.coco_handler import COCOHandler

# Load configuration from previous notebooks
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

# Set up volume directories
EVALUATION_RESULTS_DIR = f"{BASE_VOLUME_PATH}/evaluation_results"
EVALUATION_LOGS_DIR = f"{BASE_VOLUME_PATH}/logs"

# Create directories
os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
os.makedirs(EVALUATION_LOGS_DIR, exist_ok=True)

print(f"📁 Volume directories created:")
print(f"   Evaluation Results: {EVALUATION_RESULTS_DIR}")
print(f"   Evaluation Logs: {EVALUATION_LOGS_DIR}")

# Load configuration
if os.path.exists(CONFIG_PATH):
    config = load_config(CONFIG_PATH)
    # Update checkpoint directory to use volume path
    config['training']['checkpoint_dir'] = f"{BASE_VOLUME_PATH}/checkpoints"
    print(f"✅ Fixed config: updated checkpoint directory")
else:
    print("⚠️  Config file not found. Using default detection config.")
    config = get_default_config("detection")
    # Update checkpoint directory to use volume path
    config['training']['checkpoint_dir'] = f"{BASE_VOLUME_PATH}/checkpoints"

print("✅ Configuration loaded successfully!")
print(f"📁 Checkpoint directory: {config['training']['checkpoint_dir']}")

# Initialize logging
logger = create_databricks_logger(
    name="model_evaluation",
    log_file=f"{EVALUATION_LOGS_DIR}/evaluation.log"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Find and Load Best Checkpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ### Locate the Best Model Checkpoint

# COMMAND ----------

def find_best_checkpoint():
    """Find the best checkpoint from training."""
    
    print("🔍 Finding best checkpoint...")
    
    # Check MLflow for registered model
    try:
        model_name = config['model']['model_name']
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pytorch.load_model(model_uri)
        print(f"✅ Found production model: {model_name}")
        return model, "mlflow_production"
    except Exception as e:
        print(f"⚠️  No production model found: {e}")
    
    # Check for local checkpoints
    checkpoint_dir = f"{BASE_VOLUME_PATH}/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        
        if checkpoint_files:
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            best_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
            print(f"✅ Found checkpoint: {best_checkpoint}")
            return best_checkpoint, "local_checkpoint"
    
    print("❌ No checkpoint found")
    return None, None

best_checkpoint, checkpoint_type = find_best_checkpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Model and Data

# COMMAND ----------

def load_model_and_data():
    """Load the trained model and prepare data for evaluation."""
    
    if not best_checkpoint:
        print("❌ No checkpoint available for evaluation")
        return None, None
    
    print("📦 Loading model and data...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # Load model
    if checkpoint_type == "mlflow_production":
        model = best_checkpoint  # Already loaded
    else:
        # Prepare model config with num_workers from data config
        model_config = config["model"].copy()
        model_config["num_workers"] = config["data"]["num_workers"]
        
        model = DetectionModel.load_from_checkpoint(best_checkpoint, config=model_config)
    
    model.eval()
    model.to(device)
    
    # Prepare data module
    # Setup adapter first
    adapter = get_input_adapter(config["model"]["model_name"], image_size=config["data"].get("image_size", 800))
    if adapter is None:
        print("❌ Failed to create adapter")
        return None, None
    
    # Create data module with data config only
    data_module = DetectionDataModule(config["data"])
    
    # Assign adapter to data module
    data_module.adapter = adapter
    
    # Setup for testing
    data_module.setup(stage='test')
    
    print(f"✅ Model loaded successfully")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, data_module

model, data_module = load_model_and_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Comprehensive Evaluation

# COMMAND ----------

def evaluate_model(model, data_module):
    """Run comprehensive model evaluation."""
    
    if not model or not data_module:
        print("❌ Model or data not available")
        return None
    
    print("🔬 Running model evaluation...")
    
    # Initialize evaluator
    evaluator = DetectionEvaluator(config)
    
    # Run evaluation
    evaluation_results = evaluator.evaluate(model, data_module.test_dataloader())
    
    print("✅ Evaluation completed")
    return evaluation_results

evaluation_results = evaluate_model(model, data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Evaluation Metrics

# COMMAND ----------

def analyze_metrics(evaluation_results):
    """Analyze the evaluation metrics in detail."""
    
    if not evaluation_results:
        print("❌ No evaluation results available")
        return None
    
    print("📊 Analyzing evaluation metrics...")
    
    # Extract key metrics
    metrics = {
        'mAP': evaluation_results.get('mAP', 0.0),
        'mAP_50': evaluation_results.get('mAP_50', 0.0),
        'mAP_75': evaluation_results.get('mAP_75', 0.0),
        'precision': evaluation_results.get('precision', 0.0),
        'recall': evaluation_results.get('recall', 0.0),
        'f1_score': evaluation_results.get('f1_score', 0.0)
    }
    
    print("🎯 Key Metrics:")
    for metric_name, value in metrics.items():
        print(f"   {metric_name}: {value:.4f}")
    
    # Analyze per-class performance if available
    if 'per_class_metrics' in evaluation_results:
        per_class = evaluation_results['per_class_metrics']
        print(f"\n📈 Per-Class Performance (Top 10):")
        
        # Sort by mAP
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1].get('mAP', 0), reverse=True)
        
        for i, (class_name, class_metrics) in enumerate(sorted_classes[:10]):
            print(f"   {i+1:2d}. {class_name}: mAP={class_metrics.get('mAP', 0):.4f}, "
                  f"precision={class_metrics.get('precision', 0):.4f}, "
                  f"recall={class_metrics.get('recall', 0):.4f}")
    
    # Performance analysis
    if 'inference_time' in evaluation_results:
        inference_time = evaluation_results['inference_time']
        fps = 1.0 / inference_time if inference_time > 0 else 0
        print(f"\n⚡ Performance Analysis:")
        print(f"   Inference time per image: {inference_time:.4f} seconds")
        print(f"   FPS: {fps:.2f}")
    
    return metrics

metrics = analyze_metrics(evaluation_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Visualization and Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Predictions

# COMMAND ----------

def visualize_predictions(model, data_module, num_samples=6):
    """Visualize model predictions on sample images."""
    
    if not model or not data_module:
        print("❌ Model or data not available")
        return
    
    print("🎨 Creating prediction visualizations...")
    
    # Get sample data
    test_loader = data_module.test_dataloader()
    sample_batch = next(iter(test_loader))
    
    # Move to device
    device = next(model.parameters()).device
    images = sample_batch['image'].to(device)
    targets = sample_batch['target']
    
    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DETR Model Predictions on Test Images', fontsize=16)
    
    for i in range(min(num_samples, len(images))):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Get image and predictions
        image = images[i].cpu().permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0,1]
        
        # Display image
        ax.imshow(image)
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
        
        # Add predictions if available
        if hasattr(predictions, 'pred_boxes') and len(predictions.pred_boxes[i]) > 0:
            boxes = predictions.pred_boxes[i].cpu().numpy()
            scores = predictions.scores[i].cpu().numpy()
            labels = predictions.pred_classes[i].cpu().numpy()
            
            # Draw bounding boxes
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{EVALUATION_RESULTS_DIR}/predictions_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {plot_path}")
    
    plt.show()

visualize_predictions(model, data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Per-Class Performance

# COMMAND ----------

def analyze_per_class_performance(model, data_module):
    """Analyze model performance for each class."""
    
    if not model or not data_module:
        print("❌ Model or data not available")
        return None
    
    print("📊 Analyzing per-class performance...")
    
    # Initialize evaluator
    evaluator = DetectionEvaluator(config)
    
    # Get per-class metrics
    per_class_metrics = evaluator.evaluate_per_class(model, data_module.test_dataloader())
    
    if not per_class_metrics:
        print("❌ No per-class metrics available")
        return None
    
    # Create performance summary
    performance_summary = []
    for class_name, metrics in per_class_metrics.items():
        performance_summary.append({
            'class': class_name,
            'mAP': metrics.get('mAP', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0)
        })
    
    # Sort by mAP
    performance_summary.sort(key=lambda x: x['mAP'], reverse=True)
    
    print("🏆 Top 10 Performing Classes:")
    for i, summary in enumerate(performance_summary[:10]):
        print(f"   {i+1:2d}. {summary['class']}: mAP={summary['mAP']:.4f}, "
              f"precision={summary['precision']:.4f}, recall={summary['recall']:.4f}")
    
    print(f"\n❌ Bottom 10 Performing Classes:")
    for i, summary in enumerate(performance_summary[-10:]):
        print(f"   {len(performance_summary)-9+i:2d}. {summary['class']}: mAP={summary['mAP']:.4f}, "
              f"precision={summary['precision']:.4f}, recall={summary['recall']:.4f}")
    
    return performance_summary

class_performance = analyze_per_class_performance(model, data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Model Errors

# COMMAND ----------

def analyze_model_errors(model, data_module):
    """Analyze common error patterns in model predictions."""
    
    if not model or not data_module:
        print("❌ Model or data not available")
        return None
    
    print("🔍 Analyzing model errors...")
    
    # Initialize evaluator
    evaluator = DetectionEvaluator(config)
    
    # Get error analysis
    error_analysis = evaluator.analyze_errors(model, data_module.test_dataloader())
    
    if not error_analysis:
        print("❌ No error analysis available")
        return None
    
    print("📊 Error Analysis Summary:")
    
    # False positive analysis
    if 'false_positives' in error_analysis:
        fp_analysis = error_analysis['false_positives']
        print(f"   False Positives:")
        print(f"     Total: {fp_analysis.get('total', 0)}")
        print(f"     Average confidence: {fp_analysis.get('avg_confidence', 0):.4f}")
        print(f"     Most common classes: {fp_analysis.get('top_classes', [])}")
    
    # False negative analysis
    if 'false_negatives' in error_analysis:
        fn_analysis = error_analysis['false_negatives']
        print(f"   False Negatives:")
        print(f"     Total: {fn_analysis.get('total', 0)}")
        print(f"     Most common classes: {fn_analysis.get('top_classes', [])}")
    
    # Localization errors
    if 'localization_errors' in error_analysis:
        loc_errors = error_analysis['localization_errors']
        print(f"   Localization Errors:")
        print(f"     Average IoU: {loc_errors.get('avg_iou', 0):.4f}")
        print(f"     Poor localization rate: {loc_errors.get('poor_localization_rate', 0):.4f}")
    
    # Size-based errors
    if 'size_errors' in error_analysis:
        size_errors = error_analysis['size_errors']
        print(f"   Size-based Errors:")
        print(f"     Small object errors: {size_errors.get('small_object_errors', 0)}")
        print(f"     Large object errors: {size_errors.get('large_object_errors', 0)}")
    
    return error_analysis

error_analysis = analyze_model_errors(model, data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Performance Testing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Inference Speed

# COMMAND ----------

def analyze_inference_speed(model, data_module):
    """Measure and analyze inference speed and performance."""
    
    if not model or not data_module:
        print("❌ Model or data not available")
        return None
    
    print("⚡ Analyzing inference speed...")
    
    # Get device
    device = next(model.parameters()).device
    print(f"   Device: {device}")
    
    # Warm up
    print("   Warming up model...")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 800, 800).to(device)
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure inference time
    print("   Measuring inference time...")
    num_samples = 100
    inference_times = []
    
    test_loader = data_module.test_dataloader()
    sample_batch = next(iter(test_loader))
    images = sample_batch['image'].to(device)
    
    with torch.no_grad():
        for i in range(num_samples):
            start_time = time.time()
            _ = model(images)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            inference_times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    fps = 1.0 / avg_time
    throughput = fps * images.shape[0]  # Images per second
    
    print(f"✅ Speed Analysis:")
    print(f"   Average inference time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"   FPS: {fps:.2f}")
    print(f"   Throughput: {throughput:.2f} images/second")
    
    # Memory usage
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"   GPU Memory - Allocated: {gpu_memory:.2f} GB")
        print(f"   GPU Memory - Reserved: {gpu_memory_reserved:.2f} GB")
    
    speed_analysis = {
        'avg_inference_time': avg_time,
        'std_inference_time': std_time,
        'fps': fps,
        'throughput': throughput,
        'gpu_memory_allocated': gpu_memory if torch.cuda.is_available() else 0,
        'gpu_memory_reserved': gpu_memory_reserved if torch.cuda.is_available() else 0
    }
    
    return speed_analysis

speed_analysis = analyze_inference_speed(model, data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Evaluation Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Evaluation Results

# COMMAND ----------

def save_evaluation_results(evaluation_results, metrics, class_performance, error_analysis, speed_analysis):
    """Save all evaluation results to files."""
    
    print("💾 Saving evaluation results...")
    
    # Create results dictionary
    results = {
        'evaluation_results': evaluation_results,
        'metrics': metrics,
        'class_performance': class_performance,
        'error_analysis': error_analysis,
        'speed_analysis': speed_analysis,
        'evaluation_date': str(pd.Timestamp.now()),
        'model_config': config['model'],
        'data_config': config['data']
    }
    
    # Save detailed results
    results_path = f"{EVALUATION_RESULTS_DIR}/evaluation_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"✅ Detailed results saved to: {results_path}")
    
    # Save metrics summary
    if metrics:
        summary_path = f"{EVALUATION_RESULTS_DIR}/metrics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics summary saved to: {summary_path}")
    
    # Save class performance
    if class_performance:
        class_perf_path = f"{EVALUATION_RESULTS_DIR}/class_performance.json"
        with open(class_perf_path, 'w') as f:
            json.dump(class_performance, f, indent=2)
        print(f"✅ Class performance saved to: {class_perf_path}")
    
    # Save speed analysis
    if speed_analysis:
        speed_path = f"{EVALUATION_RESULTS_DIR}/speed_analysis.json"
        with open(speed_path, 'w') as f:
            json.dump(speed_analysis, f, indent=2)
        print(f"✅ Speed analysis saved to: {speed_path}")
    
    return True

# Save results
results_saved = save_evaluation_results(
    evaluation_results, metrics, class_performance, error_analysis, speed_analysis
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary and Recommendations

# COMMAND ----------

print("=" * 60)
print("MODEL EVALUATION SUMMARY")
print("=" * 60)

if metrics:
    print(f"✅ Model Performance:")
    print(f"   mAP: {metrics.get('mAP', 0):.4f}")
    print(f"   mAP@50: {metrics.get('mAP_50', 0):.4f}")
    print(f"   mAP@75: {metrics.get('mAP_75', 0):.4f}")
    print(f"   Precision: {metrics.get('precision', 0):.4f}")
    print(f"   Recall: {metrics.get('recall', 0):.4f}")
    print(f"   F1 Score: {metrics.get('f1_score', 0):.4f}")

if speed_analysis:
    print(f"\n⚡ Performance Metrics:")
    print(f"   FPS: {speed_analysis.get('fps', 0):.2f}")
    print(f"   Throughput: {speed_analysis.get('throughput', 0):.2f} images/second")
    print(f"   Avg inference time: {speed_analysis.get('avg_inference_time', 0):.4f} seconds")

if class_performance:
    print(f"\n📊 Class Performance:")
    best_class = max(class_performance, key=lambda x: x['mAP'])
    worst_class = min(class_performance, key=lambda x: x['mAP'])
    print(f"   Best performing: {best_class['class']} (mAP: {best_class['mAP']:.4f})")
    print(f"   Worst performing: {worst_class['class']} (mAP: {worst_class['mAP']:.4f})")

print(f"\n📁 Results saved to:")
print(f"   Evaluation results: {EVALUATION_RESULTS_DIR}")
print(f"   Detailed results: {EVALUATION_RESULTS_DIR}/evaluation_results.yaml")
print(f"   Metrics summary: {EVALUATION_RESULTS_DIR}/metrics_summary.json")

print(f"\n🎯 Recommendations:")
if metrics and metrics.get('mAP', 0) < 0.3:
    print("   ⚠️  Model performance is low. Consider:")
    print("      - Retraining with more epochs")
    print("      - Hyperparameter tuning")
    print("      - Data augmentation improvements")
elif metrics and metrics.get('mAP', 0) < 0.5:
    print("   📈 Model performance is moderate. Consider:")
    print("      - Fine-tuning on specific classes")
    print("      - Ensemble methods")
    print("      - Advanced augmentation techniques")
else:
    print("   ✅ Model performance is good!")

if speed_analysis and speed_analysis.get('fps', 0) < 10:
    print("   ⚡ Inference speed is slow. Consider:")
    print("      - Model quantization")
    print("      - TensorRT optimization")
    print("      - Batch processing")

print("=" * 60) 
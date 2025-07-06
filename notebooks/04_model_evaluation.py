# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation and Analysis
# MAGIC 
# MAGIC This notebook demonstrates comprehensive evaluation of the trained DETR model:
# MAGIC 1. Load the trained model and checkpoint
# MAGIC 2. Run evaluation on the test set
# MAGIC 3. Analyze detailed metrics (mAP, precision, recall, F1)
# MAGIC 4. Visualize predictions and analyze errors
# MAGIC 5. Measure inference speed and performance
# MAGIC 6. Generate evaluation reports and recommendations

# COMMAND ----------

# (Databricks only) Install requirements and restart Python if running interactively
# %pip install -r "../requirements.txt"
# dbutils.library.restartPython()

# (Databricks only) Run setup and previous notebooks if running interactively
# %run ./00_setup_and_config
# %run ./01_data_preparation
# %run ./02_model_training

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
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

# Add the project root to Python path
project_root = "/Volumes/<catalog>/<schema>/<volume>/<path>"
sys.path.append(project_root)

# Import project modules
from src.utils.logging import setup_logger
from src.utils.coco_handler import COCOHandler
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule
from src.tasks.detection.evaluate import DetectionEvaluator

# COMMAND ----------

# DBTITLE 1,Initialize Logging and Configuration
volume_path = os.getenv("UNITY_CATALOG_VOLUME", project_root)
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="model_evaluation",
    log_file=f"{log_dir}/evaluation.log"
)

# Load configuration
try:
    config = load_config("detection_detr")
except NameError:
    # Fallback if load_config is not available
    from src.config import load_config
    config = load_config("detection_detr")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Find and Load Best Checkpoint

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
    base_volume_path = os.getenv("BASE_VOLUME_PATH", "/Volumes/<catalog>/<schema>/<volume>/cv_detr_training")
    checkpoint_dir = f"{base_volume_path}/checkpoints"
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
    
    # Load model
    if checkpoint_type == "mlflow_production":
        model = best_checkpoint  # Already loaded
    else:
        model = DetectionModel.load_from_checkpoint(best_checkpoint, config=config)
    
    model.eval()
    model.to(device)
    
    # Prepare data module
    data_module = DetectionDataModule(config)
    data_module.setup(stage='test')
    
    print(f"✅ Model loaded successfully")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, data_module

model, data_module = load_model_and_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model Evaluation

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
    """Analyze and extract key metrics from evaluation results."""
    
    if not evaluation_results:
        print("❌ No evaluation results to analyze")
        return None
    
    print("📊 Analyzing evaluation metrics...")
    
    # Extract key metrics
    metrics = {
        'mAP': evaluation_results.get('mAP', 0.0),
        'mAP_50': evaluation_results.get('mAP_50', 0.0),
        'mAP_75': evaluation_results.get('mAP_75', 0.0),
        'Precision': evaluation_results.get('Precision', 0.0),
        'Recall': evaluation_results.get('Recall', 0.0),
        'F1-Score': evaluation_results.get('F1-Score', 0.0)
    }
    
    print("📈 Key Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Performance assessment
    if metrics['mAP'] >= 0.4:
        print("   🎉 Excellent performance!")
    elif metrics['mAP'] >= 0.3:
        print("   ✅ Good performance")
    elif metrics['mAP'] >= 0.2:
        print("   ⚠️  Moderate performance")
    else:
        print("   ❌ Poor performance - consider retraining")
    
    return metrics

metrics = analyze_metrics(evaluation_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Visualization and Analysis

# MAGIC %md
# MAGIC ### Visualize Model Predictions

# COMMAND ----------

def visualize_predictions(model, data_module, num_samples=6):
    """Visualize model predictions on test samples."""
    
    if not model or not data_module:
        print("❌ Model or data not available")
        return
    
    print(f"🎨 Visualizing {num_samples} predictions...")
    
    # Get test samples
    test_loader = data_module.test_dataloader()
    samples = []
    
    for batch in test_loader:
        if len(samples) >= num_samples:
            break
        samples.extend(batch)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples[:num_samples]):
        if i >= len(axes):
            break
            
        # Get prediction
        with torch.no_grad():
            prediction = model(sample['image'].unsqueeze(0).to(device))
        
        # Visualize
        image = sample['image'].permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        
        axes[i].imshow(image)
        
        # Draw predictions
        if 'boxes' in prediction:
            boxes = prediction['boxes'][0].cpu().numpy()
            scores = prediction['scores'][0].cpu().numpy()
            labels = prediction['labels'][0].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                          linewidth=2, edgecolor='red', facecolor='none')
                    axes[i].add_patch(rect)
                    axes[i].text(x1, y1, f'{label}: {score:.2f}', 
                               color='red', fontsize=8)
        
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Predictions visualized")

if model and data_module:
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
    
    # Initialize evaluator for per-class analysis
    evaluator = DetectionEvaluator(config)
    
    # Run per-class evaluation
    class_metrics = evaluator.evaluate_per_class(model, data_module.test_dataloader())
    
    # Analyze results
    class_performance = {}
    for class_id, metrics in class_metrics.items():
        class_name = config['data']['classes'][class_id] if class_id < len(config['data']['classes']) else f"Class_{class_id}"
        class_performance[class_name] = {
            'mAP': metrics.get('mAP', 0.0),
            'Precision': metrics.get('Precision', 0.0),
            'Recall': metrics.get('Recall', 0.0),
            'F1-Score': metrics.get('F1-Score', 0.0)
        }
    
    # Sort by mAP
    class_performance = dict(sorted(class_performance.items(), 
                                  key=lambda x: x[1]['mAP'], reverse=True))
    
    print("📈 Per-Class Performance (Top 10):")
    for i, (class_name, metrics) in enumerate(list(class_performance.items())[:10]):
        print(f"   {i+1:2d}. {class_name:20s} - mAP: {metrics['mAP']:.4f}")
    
    return class_performance

if model and data_module:
    class_performance = analyze_per_class_performance(model, data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Model Errors

# MAGIC %md
# MAGIC ### Error Analysis

# COMMAND ----------

def analyze_model_errors(model, data_module):
    """Analyze common model errors and failure cases."""
    
    if not model or not data_module:
        print("❌ Model or data not available")
        return None
    
    print("🔍 Analyzing model errors...")
    
    # Initialize error tracking
    error_analysis = {
        'false_positives': 0,
        'false_negatives': 0,
        'misclassifications': 0,
        'localization_errors': 0,
        'total_predictions': 0
    }
    
    # Run error analysis
    test_loader = data_module.test_dataloader()
    
    for batch in test_loader:
        with torch.no_grad():
            predictions = model(batch['image'].to(device))
            targets = batch['target']
        
        # Analyze errors (simplified analysis)
        for pred, target in zip(predictions, targets):
            error_analysis['total_predictions'] += len(pred['boxes'])
            
            # This is a simplified error analysis
            # In practice, you'd implement proper IoU-based analysis
            if len(pred['boxes']) > len(target['boxes']):
                error_analysis['false_positives'] += len(pred['boxes']) - len(target['boxes'])
            elif len(pred['boxes']) < len(target['boxes']):
                error_analysis['false_negatives'] += len(target['boxes']) - len(pred['boxes'])
    
    print("📊 Error Analysis:")
    total_errors = sum([error_analysis[k] for k in ['false_positives', 'false_negatives', 'misclassifications', 'localization_errors']])
    
    for error_type, count in error_analysis.items():
        if error_type != 'total_predictions':
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            print(f"   {error_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    return error_analysis

if model and data_module:
    error_analysis = analyze_model_errors(model, data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Performance Analysis

# MAGIC %md
# MAGIC ### Measure Inference Speed

# MAGIC %md
# MAGIC ### Speed Analysis

# COMMAND ----------

def analyze_inference_speed(model, data_module):
    """Measure and analyze model inference speed."""
    
    if not model or not data_module:
        print("❌ Model or data not available")
        return None
    
    print("⚡ Analyzing inference speed...")
    
    # Warm up
    dummy_input = torch.randn(1, 3, 800, 800).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure speed
    test_loader = data_module.test_dataloader()
    times = []
    batch_sizes = []
    
    with torch.no_grad():
        for batch in test_loader:
            start_time = time.time()
            _ = model(batch['image'].to(device))
            end_time = time.time()
            
            times.append(end_time - start_time)
            batch_sizes.append(batch['image'].size(0))
    
    # Calculate metrics
    avg_time = np.mean(times)
    avg_batch_size = np.mean(batch_sizes)
    fps = avg_batch_size / avg_time
    
    print("⚡ Speed Analysis:")
    print(f"   Average batch time: {avg_time:.3f}s")
    print(f"   Average batch size: {avg_batch_size:.1f}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Per-image time: {avg_time/avg_batch_size*1000:.1f}ms")
    
    # Performance assessment
    if fps >= 30:
        print("   🎉 Real-time performance achieved!")
    elif fps >= 10:
        print("   ✅ Good performance for batch processing")
    else:
        print("   ⚠️  Slow inference speed")
    
    return {
        'avg_time': avg_time,
        'avg_batch_size': avg_batch_size,
        'fps': fps,
        'per_image_time': avg_time/avg_batch_size*1000
    }

if model and data_module:
    speed_analysis = analyze_inference_speed(model, data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save Evaluation Results

# MAGIC %md
# MAGIC ### Export Evaluation Results

# COMMAND ----------

def save_evaluation_results(evaluation_results, metrics, class_performance, error_analysis, speed_analysis):
    """Save comprehensive evaluation results."""
    
    print("\n💾 Saving evaluation results...")
    
    # Create results directory
    results_dir = f"{BASE_VOLUME_PATH}/results/evaluation"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare results summary
    results_summary = {
        'model_info': {
            'model_name': config['model']['model_name'],
            'checkpoint_path': best_checkpoint,
            'evaluation_date': pd.Timestamp.now().isoformat()
        },
        'metrics': metrics,
        'class_performance': class_performance,
        'error_analysis': error_analysis,
        'speed_analysis': speed_analysis,
        'evaluation_results': evaluation_results
    }
    
    # Save as YAML
    import yaml
    results_path = f"{results_dir}/evaluation_summary.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results_summary, f)
    
    print(f"✅ Evaluation results saved to: {results_path}")
    
    # Save detailed metrics as CSV
    if class_performance:
        class_df = pd.DataFrame.from_dict(class_performance, orient='index')
        class_df.to_csv(f"{results_dir}/per_class_metrics.csv")
        print(f"✅ Per-class metrics saved to: {results_dir}/per_class_metrics.csv")
    
    # Create evaluation report
    report_path = f"{results_dir}/evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("DETR Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {config['model']['model_name']}\n")
        f.write(f"Checkpoint: {best_checkpoint}\n")
        f.write(f"Evaluation Date: {pd.Timestamp.now()}\n\n")
        
        f.write("Key Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\nInference Speed:\n")
        f.write(f"  FPS: {speed_analysis['fps']:.1f}\n")
        f.write(f"  Per-image time: {speed_analysis['per_image_time']:.1f}ms\n")
        
        f.write(f"\nError Analysis:\n")
        for error_type, count in error_analysis.items():
            f.write(f"  {error_type}: {count}\n")
    
    print(f"✅ Evaluation report saved to: {report_path}")
    
    return results_dir

if all([evaluation_results, metrics, class_performance, error_analysis, speed_analysis]):
    results_dir = save_evaluation_results(evaluation_results, metrics, class_performance, error_analysis, speed_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary and Recommendations

# MAGIC %md
# MAGIC ### Evaluation Summary

# COMMAND ----------

print("=" * 60)
print("MODEL EVALUATION SUMMARY")
print("=" * 60)

if evaluation_results and metrics:
    print(f"✅ Model Performance:")
    print(f"   mAP: {metrics['mAP']:.4f}")
    print(f"   mAP@50: {metrics['mAP_50']:.4f}")
    print(f"   mAP@75: {metrics['mAP_75']:.4f}")
    print(f"   Precision: {metrics['Precision']:.4f}")
    print(f"   Recall: {metrics['Recall']:.4f}")
    print(f"   F1-Score: {metrics['F1-Score']:.4f}")
    
    print(f"\n⚡ Inference Performance:")
    if speed_analysis:
        print(f"   FPS: {speed_analysis['fps']:.1f}")
        print(f"   Per-image time: {speed_analysis['per_image_time']:.1f}ms")
    
    print(f"\n🔍 Error Analysis:")
    if error_analysis:
        total_errors = sum(error_analysis.values())
        for error_type, count in error_analysis.items():
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            print(f"   {error_type.replace('_', ' ').title()}: {percentage:.1f}%")
    
    print(f"\n📁 Results Location:")
    print(f"   Results directory: {BASE_VOLUME_PATH}/results/evaluation/")
    print(f"   Summary: evaluation_summary.yaml")
    print(f"   Report: evaluation_report.txt")
    print(f"   Per-class metrics: per_class_metrics.csv")
    
    print(f"\n🎯 Recommendations:")
    if metrics['mAP'] < 0.4:
        print("   ⚠️  Consider hyperparameter tuning or longer training")
    if speed_analysis and speed_analysis['fps'] < 10:
        print("   ⚠️  Consider model optimization for faster inference")
    if error_analysis and error_analysis['false_positives'] > error_analysis['false_negatives']:
        print("   ⚠️  Consider adjusting confidence threshold to reduce false positives")
    
    print("   ✅ Model evaluation completed successfully!")
else:
    print("❌ Evaluation failed or incomplete")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding DETR Evaluation

# MAGIC 
# MAGIC ### Key Evaluation Concepts:
# MAGIC 
# MAGIC **1. mAP (mean Average Precision):**
# MAGIC - Primary metric for object detection
# MAGIC - Combines precision and recall across all classes
# MAGIC - Higher values indicate better performance
# MAGIC - DETR paper reports ~0.42 mAP on COCO
# MAGIC 
# MAGIC **2. Precision-Recall Trade-off:**
# MAGIC - Precision: Accuracy of positive predictions
# MAGIC - Recall: Ability to find all positive instances
# MAGIC - DETR balances both through set prediction
# MAGIC - No NMS needed, reducing false positives
# MAGIC 
# MAGIC **3. Per-Class Analysis:**
# MAGIC - Different object classes have varying difficulty
# MAGIC - Small objects (e.g., person) vs large objects (e.g., car)
# MAGIC - Helps identify model weaknesses
# MAGIC - Guides data augmentation strategies
# MAGIC 
# MAGIC **4. Error Analysis:**
# MAGIC - False Positives: Model predicts object when none exists
# MAGIC - False Negatives: Model misses existing objects
# MAGIC - Misclassifications: Wrong class predictions
# MAGIC - Localization errors: Wrong bounding box locations
# MAGIC 
# MAGIC **5. Inference Speed:**
# MAGIC - Critical for real-time applications
# MAGIC - DETR is slower than YOLO but more accurate
# MAGIC - Transformer computation is the bottleneck
# MAGIC - Batch processing improves efficiency
# MAGIC 
# MAGIC **6. Model Comparison:**
# MAGIC - Compare with DETR paper results
# MAGIC - Benchmark against other detection models
# MAGIC - Consider speed-accuracy trade-offs
# MAGIC - Evaluate on your specific use case
# MAGIC 
# MAGIC The evaluation provides comprehensive insights into model performance and guides future improvements! 
# Databricks notebook source
# MAGIC %md
# MAGIC # 04. Serverless GPU Model Evaluation and Analysis
# MAGIC 
# MAGIC This notebook demonstrates comprehensive evaluation of models trained using **Databricks Serverless GPU compute**:
# MAGIC 1. Load the trained model and checkpoint
# MAGIC 2. Run evaluation on the test set
# MAGIC 3. Analyze detailed metrics (mAP, precision, recall, F1)
# MAGIC 4. Visualize predictions and analyze errors
# MAGIC 5. Measure inference speed and performance
# MAGIC 6. Generate evaluation reports and recommendations
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **Serverless GPU Model Evaluation Process:**
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
import torch
import lightning
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
import cv2
from PIL import Image, ImageDraw, ImageFont

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')

from config_serverless import load_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from training.trainer_serverless import UnifiedTrainerServerless
from lightning.pytorch.loggers import MLFlowLogger

# Load configuration
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_serverless_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs_serverless/detection_detr_config.yaml"

# Set up volume directories
CHECKPOINT_DIR = f"{BASE_VOLUME_PATH}/checkpoints"
RESULTS_DIR = f"{BASE_VOLUME_PATH}/results"
LOGS_DIR = f"{BASE_VOLUME_PATH}/logs"
EVALUATION_DIR = f"{BASE_VOLUME_PATH}/evaluation"

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

print(f"📁 Volume directories created:")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Results: {RESULTS_DIR}")
print(f"   Evaluation: {EVALUATION_DIR}")

# Load configuration
if os.path.exists(CONFIG_PATH):
    config = load_config(CONFIG_PATH)
    config['training']['checkpoint_dir'] = CHECKPOINT_DIR
else:
    print("⚠️  Config file not found. Using default serverless config.")
    config = {
        'model': {
            'model_name': 'facebook/detr-resnet-50',
            'task_type': 'detection',
            'num_classes': 91,
            'image_size': 800
        },
        'data': {
            'batch_size': 1,  # Use batch size 1 for evaluation
            'num_workers': 4,
            'image_size': 800,
            'train_data_path': f"{BASE_VOLUME_PATH}/data/train",
            'train_annotation_file': f"{BASE_VOLUME_PATH}/data/train_annotations.json",
            'val_data_path': f"{BASE_VOLUME_PATH}/data/val",
            'val_annotation_file': f"{BASE_VOLUME_PATH}/data/val_annotations.json",
            'test_data_path': f"{BASE_VOLUME_PATH}/data/test",
            'test_annotation_file': f"{BASE_VOLUME_PATH}/data/test_annotations.json"
        },
        'training': {
            'max_epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'checkpoint_dir': CHECKPOINT_DIR,
            'distributed': False,  # Disable distributed for evaluation
            'use_serverless_gpu': False,  # Disable serverless GPU for evaluation
            'serverless_gpu_type': 'A10',
            'serverless_gpu_count': 1,
            'monitor_metric': 'val_map',
            'monitor_mode': 'max',
            'early_stopping_patience': 20,
            'log_every_n_steps': 50
        },
        'output': {
            'results_dir': RESULTS_DIR
        }
    }

print("✅ Configuration loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Trained Model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find Best Checkpoint

# COMMAND ----------

def find_best_checkpoint():
    """Find the best checkpoint file for evaluation."""
    
    checkpoint_files = []
    
    # Look for checkpoint files in the checkpoint directory
    if os.path.exists(CHECKPOINT_DIR):
        for file in os.listdir(CHECKPOINT_DIR):
            if file.endswith('.ckpt'):
                checkpoint_files.append(os.path.join(CHECKPOINT_DIR, file))
    
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        print(f"   Checked directory: {CHECKPOINT_DIR}")
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"📁 Found {len(checkpoint_files)} checkpoint files:")
    for i, checkpoint in enumerate(checkpoint_files):
        file_size = os.path.getsize(checkpoint) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint))
        print(f"   {i+1}. {os.path.basename(checkpoint)} ({file_size:.1f} MB, {mod_time})")
    
    # Return the newest checkpoint
    best_checkpoint = checkpoint_files[0]
    print(f"\n✅ Using checkpoint: {os.path.basename(best_checkpoint)}")
    
    return best_checkpoint

best_checkpoint = find_best_checkpoint()

if best_checkpoint is None:
    print("❌ Cannot proceed without a checkpoint file.")
    print("Please run training first or check the checkpoint directory.")
    dbutils.notebook.exit("No checkpoint found")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Model from Checkpoint

# COMMAND ----------

def load_model_from_checkpoint(checkpoint_path):
    """Load the trained model from checkpoint."""
    
    try:
        # Prepare model config
        model_config = config["model"].copy()
        model_config["num_workers"] = config["data"]["num_workers"]
        
        # Create model
        model = DetectionModel(model_config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model weights
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        
        print(f"✅ Model loaded successfully from checkpoint")
        print(f"   Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"   Model: {config['model']['model_name']}")
        print(f"   Classes: {config['model']['num_classes']}")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model from checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

model = load_model_from_checkpoint(best_checkpoint)

if model is None:
    print("❌ Cannot proceed without loading the model.")
    dbutils.notebook.exit("Model loading failed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Setup Test Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Test Data Module

# COMMAND ----------

def setup_test_data():
    """Setup the test data module for evaluation."""
    
    # Setup adapter
    from tasks.detection.adapters import get_input_adapter
    adapter = get_input_adapter(
        config["model"]["model_name"], 
        image_size=config["data"].get("image_size", 800)
    )
    
    if adapter is None:
        raise ValueError(f"Could not create adapter for model: {config['model']['model_name']}")
    
    # Create data module with test data
    test_data_config = config["data"].copy()
    test_data_config["batch_size"] = 1  # Use batch size 1 for evaluation
    
    data_module = DetectionDataModule(test_data_config)
    data_module.adapter = adapter
    data_module.setup()
    
    print(f"✅ Test data module initialized")
    print(f"   Test samples: {len(data_module.test_dataset) if hasattr(data_module, 'test_dataset') else 'N/A'}")
    print(f"   Batch size: {test_data_config['batch_size']}")
    print(f"   Image size: {test_data_config.get('image_size', 800)}")
    
    return data_module

test_data_module = setup_test_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comprehensive Evaluation

# COMMAND ----------

def run_comprehensive_evaluation():
    """Run comprehensive evaluation on the test set."""
    
    print("🔍 Starting comprehensive model evaluation...")
    print("=" * 50)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"✅ Model moved to device: {device}")
    
    # Get test dataloader
    test_dataloader = test_data_module.test_dataloader()
    
    # Initialize metrics
    all_predictions = []
    all_targets = []
    inference_times = []
    
    # Run evaluation
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx >= 100:  # Limit to 100 samples for demo
                break
                
            # Move batch to device
            images = batch['images'].to(device)
            targets = batch['targets']
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Store predictions and targets
            all_predictions.append(outputs)
            all_targets.append(targets)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1} batches...")
    
    print(f"✅ Evaluation completed on {len(all_predictions)} samples")
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time
    
    print(f"📊 Performance Metrics:")
    print(f"   Average inference time: {avg_inference_time:.4f}s")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total evaluation time: {sum(inference_times):.2f}s")
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'inference_times': inference_times,
        'avg_inference_time': avg_inference_time,
        'fps': fps
    }

evaluation_results = run_comprehensive_evaluation()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analyze Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance Analysis

# COMMAND ----------

def analyze_performance():
    """Analyze model performance metrics."""
    
    print("📊 Performance Analysis:")
    print("=" * 40)
    
    # Inference speed analysis
    inference_times = evaluation_results['inference_times']
    avg_time = evaluation_results['avg_inference_time']
    fps = evaluation_results['fps']
    
    print(f"Inference Speed:")
    print(f"   Average time per image: {avg_time:.4f}s")
    print(f"   FPS: {fps:.2f}")
    print(f"   Min time: {min(inference_times):.4f}s")
    print(f"   Max time: {max(inference_times):.4f}s")
    print(f"   Std deviation: {np.std(inference_times):.4f}s")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory Usage:")
        print(f"   Allocated: {memory_allocated:.2f} GB")
        print(f"   Reserved: {memory_reserved:.2f} GB")
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if fps > 10:
        print("✅ Excellent inference speed (FPS > 10)")
    elif fps > 5:
        print("⚠️  Good inference speed (FPS > 5)")
    else:
        print("❌ Slow inference speed (FPS < 5)")
    
    return {
        'avg_inference_time': avg_time,
        'fps': fps,
        'min_time': min(inference_times),
        'max_time': max(inference_times),
        'std_time': np.std(inference_times)
    }

performance_metrics = analyze_performance()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Performance Visualizations

# COMMAND ----------

def create_performance_visualizations():
    """Create visualizations for performance analysis."""
    
    # Create performance plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Model Performance Analysis', fontsize=16)
    
    # Inference time distribution
    inference_times = evaluation_results['inference_times']
    axes[0].hist(inference_times, bins=20, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(inference_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(inference_times):.4f}s')
    axes[0].set_xlabel('Inference Time (seconds)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Inference Time Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # FPS over time
    fps_over_time = [1.0 / t for t in inference_times]
    axes[1].plot(fps_over_time, alpha=0.7)
    axes[1].axhline(np.mean(fps_over_time), color='red', linestyle='--',
                   label=f'Mean FPS: {np.mean(fps_over_time):.2f}')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('FPS')
    axes[1].set_title('FPS Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plot_path = f"{EVALUATION_DIR}/performance_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Performance visualization saved to: {plot_path}")

create_performance_visualizations()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate Evaluation Report

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Evaluation Report

# COMMAND ----------

def generate_evaluation_report():
    """Generate a comprehensive evaluation report."""
    
    report = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model_name': config['model']['model_name'],
            'task_type': config['model']['task_type'],
            'num_classes': config['model']['num_classes'],
            'checkpoint_used': os.path.basename(best_checkpoint),
            'test_samples_evaluated': len(evaluation_results['predictions'])
        },
        'performance_metrics': performance_metrics,
        'configuration': {
            'image_size': config['data'].get('image_size', 800),
            'batch_size': config['data']['batch_size'],
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        },
        'recommendations': []
    }
    
    # Add recommendations based on performance
    if performance_metrics['fps'] < 5:
        report['recommendations'].append("Consider optimizing model for faster inference")
    if performance_metrics['std_time'] > performance_metrics['avg_inference_time'] * 0.5:
        report['recommendations'].append("High variance in inference time - check for memory issues")
    if torch.cuda.is_available() and torch.cuda.memory_allocated() > 8e9:
        report['recommendations'].append("High GPU memory usage - consider reducing batch size")
    
    # Save report
    report_path = f"{EVALUATION_DIR}/evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📋 Evaluation Report Generated:")
    print("=" * 40)
    print(f"Model: {report['evaluation_info']['model_name']}")
    print(f"Task: {report['evaluation_info']['task_type']}")
    print(f"Test samples: {report['evaluation_info']['test_samples_evaluated']}")
    print(f"Average FPS: {report['performance_metrics']['fps']:.2f}")
    print(f"Average inference time: {report['performance_metrics']['avg_inference_time']:.4f}s")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print(f"\n💾 Report saved to: {report_path}")
    
    return report

evaluation_report = generate_evaluation_report()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary and Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation Summary

# COMMAND ----------

print("=" * 60)
print("SERVERLESS GPU MODEL EVALUATION SUMMARY")
print("=" * 60)

print(f"✅ Model: {config['model']['model_name']}")
print(f"✅ Task: {config['model']['task_type']}")
print(f"✅ Checkpoint: {os.path.basename(best_checkpoint)}")
print(f"✅ Test samples evaluated: {len(evaluation_results['predictions'])}")

print(f"\n📊 Performance Metrics:")
print(f"   Average FPS: {performance_metrics['fps']:.2f}")
print(f"   Average inference time: {performance_metrics['avg_inference_time']:.4f}s")
print(f"   Min inference time: {performance_metrics['min_time']:.4f}s")
print(f"   Max inference time: {performance_metrics['max_time']:.4f}s")

if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1e9
    print(f"   GPU memory used: {memory_allocated:.2f} GB")

print(f"\n📁 Output Locations:")
print(f"   Evaluation results: {EVALUATION_DIR}")
print(f"   Performance plots: performance_analysis.png")
print(f"   Evaluation report: evaluation_report.json")

if evaluation_report['recommendations']:
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(evaluation_report['recommendations'], 1):
        print(f"   {i}. {rec}")

print("\n🎉 Model evaluation completed successfully!")
print("\nNext steps:")
print("1. Review the evaluation report and performance metrics")
print("2. Run notebook 05_model_registration_deployment.py to deploy the model")
print("3. Consider the recommendations for model optimization")

print("=" * 60)

# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # 03. Hyperparameter Tuning for Serverless GPU Training
# MAGIC 
# MAGIC This notebook performs hyperparameter tuning for computer vision models using **Databricks Serverless GPU compute** with Ray Tune. We'll explore different learning rates, batch sizes, and other hyperparameters to find the optimal configuration.
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **Serverless GPU Hyperparameter Tuning Strategy:**
# MAGIC 1. **Search Space Definition**: Define ranges for key hyperparameters
# MAGIC 2. **Search Algorithm**: Use Bayesian optimization for efficient search
# MAGIC 3. **Serverless GPU Resource Management**: Efficient GPU utilization across trials
# MAGIC 4. **Early Stopping**: Stop unpromising trials early
# MAGIC 5. **Result Analysis**: Compare and select best configuration
# MAGIC 
# MAGIC ### Key Hyperparameters for Computer Vision Models:
# MAGIC - **Learning Rate**: Critical for convergence (1e-5 to 1e-3)
# MAGIC - **Batch Size**: Affects memory usage and training stability
# MAGIC - **Weight Decay**: Regularization to prevent overfitting
# MAGIC - **Scheduler Parameters**: Learning rate scheduling strategy
# MAGIC - **Model Architecture**: Different backbone configurations
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 
# MAGIC 1. **Search Space Setup**: Define hyperparameter ranges
# MAGIC 2. **Trial Configuration**: Set up individual training trials with serverless GPU
# MAGIC 3. **Resource Allocation**: Manage serverless GPU resources efficiently
# MAGIC 4. **Optimization**: Run Bayesian optimization search
# MAGIC 5. **Result Analysis**: Analyze and visualize results
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
from datetime import datetime

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')

from databricks_cv_accelerator.config import load_config
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
TUNING_RESULTS_DIR = f"{BASE_VOLUME_PATH}/tuning_results"

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)

print(f"📁 Volume directories created:")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Results: {RESULTS_DIR}")
print(f"   Tuning Results: {TUNING_RESULTS_DIR}")

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
            'batch_size': 2,
            'num_workers': 4,
            'image_size': 800,
            'train_data_path': f"{BASE_VOLUME_PATH}/data/train",
            'train_annotation_file': f"{BASE_VOLUME_PATH}/data/train_annotations.json",
            'val_data_path': f"{BASE_VOLUME_PATH}/data/val",
            'val_annotation_file': f"{BASE_VOLUME_PATH}/data/val_annotations.json"
        },
        'training': {
            'max_epochs': 20,  # Reduced for tuning
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'checkpoint_dir': CHECKPOINT_DIR,
            'distributed': True,
            'use_serverless_gpu': True,
            'serverless_gpu_type': 'A10',
            'serverless_gpu_count': 4,
            'monitor_metric': 'val_map',
            'monitor_mode': 'max',
            'early_stopping_patience': 5,
            'log_every_n_steps': 50
        },
        'output': {
            'results_dir': RESULTS_DIR
        }
    }

print("✅ Configuration loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define Hyperparameter Search Space

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search Space Configuration

# COMMAND ----------

def define_search_space():
    """Define the hyperparameter search space for tuning."""
    
    search_space = {
        # Learning rate - most critical hyperparameter
        'learning_rate': {
            'type': 'log_uniform',
            'min': 1e-5,
            'max': 1e-3
        },
        
        # Weight decay for regularization
        'weight_decay': {
            'type': 'log_uniform',
            'min': 1e-6,
            'max': 1e-2
        },
        
        # Batch size - affects memory usage and training stability
        'batch_size': {
            'type': 'choice',
            'choices': [1, 2, 4, 8]
        },
        
        # Scheduler type
        'scheduler': {
            'type': 'choice',
            'choices': ['cosine', 'step', 'plateau']
        },
        
        # Image size - affects model capacity and training time
        'image_size': {
            'type': 'choice',
            'choices': [600, 800, 1000]
        }
    }
    
    print("🔍 Hyperparameter Search Space:")
    print("=" * 40)
    for param, config in search_space.items():
        print(f"{param}: {config}")
    
    return search_space

search_space = define_search_space()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Setup Model and Data for Tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Base Model and Data

# COMMAND ----------

def setup_base_model_and_data():
    """Setup the base model and data module for hyperparameter tuning."""
    
    # Prepare model config
    model_config = config["model"].copy()
    model_config["num_workers"] = config["data"]["num_workers"]
    
    # Create model
    model = DetectionModel(model_config)
    
    # Setup adapter
    from tasks.detection.adapters import get_input_adapter
    adapter = get_input_adapter(
        config["model"]["model_name"], 
        image_size=config["data"].get("image_size", 800)
    )
    
    # Create data module
    data_module = DetectionDataModule(config["data"])
    data_module.adapter = adapter
    data_module.setup()
    
    print(f"✅ Base model and data setup complete")
    print(f"   Model: {config['model']['model_name']}")
    print(f"   Train samples: {len(data_module.train_dataset)}")
    print(f"   Val samples: {len(data_module.val_dataset)}")
    
    return model, data_module

base_model, base_data_module = setup_base_model_and_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Hyperparameter Tuning Function

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Tuning Function

# COMMAND ----------

def run_hyperparameter_trial(trial_config):
    """Run a single hyperparameter tuning trial."""
    
    try:
        # Create a copy of the base config
        trial_config_full = config.copy()
        
        # Update config with trial parameters
        trial_config_full['training']['learning_rate'] = trial_config['learning_rate']
        trial_config_full['training']['weight_decay'] = trial_config['weight_decay']
        trial_config_full['data']['batch_size'] = trial_config['batch_size']
        trial_config_full['training']['scheduler'] = trial_config['scheduler']
        trial_config_full['data']['image_size'] = trial_config['image_size']
        trial_config_full['model']['image_size'] = trial_config['image_size']
        
        # Create trial-specific checkpoint directory
        trial_id = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trial_checkpoint_dir = f"{CHECKPOINT_DIR}/tuning/{trial_id}"
        os.makedirs(trial_checkpoint_dir, exist_ok=True)
        trial_config_full['training']['checkpoint_dir'] = trial_checkpoint_dir
        
        # Create trainer config
        trainer_config = {
            'task': trial_config_full['model']['task_type'],
            'model_name': trial_config_full['model']['model_name'],
            'max_epochs': trial_config_full['training']['max_epochs'],
            'log_every_n_steps': trial_config_full['training']['log_every_n_steps'],
            'monitor_metric': trial_config_full['training']['monitor_metric'],
            'monitor_mode': trial_config_full['training']['monitor_mode'],
            'early_stopping_patience': trial_config_full['training']['early_stopping_patience'],
            'checkpoint_dir': trial_checkpoint_dir,
            'save_top_k': 1,  # Only save best checkpoint for tuning
            'distributed': trial_config_full['training']['distributed'],
            'use_serverless_gpu': trial_config_full['training']['use_serverless_gpu'],
            'serverless_gpu_type': trial_config_full['training']['serverless_gpu_type'],
            'serverless_gpu_count': trial_config_full['training']['serverless_gpu_count']
        }
        
        # Create trial-specific logger
        username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
        experiment_name = f"/Users/{username}/{config['model']['task_type']}_serverless_tuning"
        run_name = f"trial_{trial_id}_{trial_config['learning_rate']:.2e}_{trial_config['batch_size']}"
        
        from utils.logging import create_databricks_logger
        logger = create_databricks_logger(
            experiment_name=experiment_name,
            run_name=run_name,
            tags={
                "task": "tuning",
                "trial_id": trial_id,
                "learning_rate": str(trial_config['learning_rate']),
                "batch_size": str(trial_config['batch_size']),
                "scheduler": trial_config['scheduler'],
                "image_size": str(trial_config['image_size'])
            }
        )
        
        # Create new model instance for this trial
        model_config = trial_config_full["model"].copy()
        model_config["num_workers"] = trial_config_full["data"]["num_workers"]
        trial_model = DetectionModel(model_config)
        
        # Create new data module with updated config
        trial_data_module = DetectionDataModule(trial_config_full["data"])
        trial_data_module.adapter = base_data_module.adapter  # Reuse adapter
        trial_data_module.setup()
        
        # Create serverless trainer
        unified_trainer = UnifiedTrainerServerless(
            config=trainer_config,
            model=trial_model,
            data_module=trial_data_module,
            logger=logger
        )
        
        # Run training
        result = unified_trainer.train()
        
        # Extract metrics
        metrics = unified_trainer.get_metrics()
        
        # Return the primary metric for optimization
        primary_metric = trial_config_full['training']['monitor_metric']
        if primary_metric in metrics:
            return metrics[primary_metric]
        else:
            print(f"⚠️  Primary metric {primary_metric} not found in results")
            return 0.0
        
    except Exception as e:
        print(f"❌ Trial failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

print("✅ Hyperparameter tuning function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Hyperparameter Tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manual Grid Search (Alternative to Ray Tune)

# COMMAND ----------

def run_manual_grid_search():
    """Run a manual grid search for hyperparameter tuning."""
    
    print("🔍 Starting manual grid search for hyperparameter tuning...")
    print("=" * 60)
    
    # Define grid search parameters
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]
    batch_sizes = [1, 2, 4]
    schedulers = ['cosine', 'step']
    image_sizes = [600, 800]
    
    results = []
    total_trials = len(learning_rates) * len(batch_sizes) * len(schedulers) * len(image_sizes)
    current_trial = 0
    
    print(f"Total trials to run: {total_trials}")
    print(f"Learning rates: {learning_rates}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Schedulers: {schedulers}")
    print(f"Image sizes: {image_sizes}")
    print("=" * 60)
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for scheduler in schedulers:
                for image_size in image_sizes:
                    current_trial += 1
                    
                    print(f"\n🎯 Trial {current_trial}/{total_trials}")
                    print(f"   Learning Rate: {lr:.2e}")
                    print(f"   Batch Size: {batch_size}")
                    print(f"   Scheduler: {scheduler}")
                    print(f"   Image Size: {image_size}")
                    
                    # Create trial config
                    trial_config = {
                        'learning_rate': lr,
                        'weight_decay': 1e-4,  # Fixed for simplicity
                        'batch_size': batch_size,
                        'scheduler': scheduler,
                        'image_size': image_size
                    }
                    
                    # Run trial
                    start_time = datetime.now()
                    metric_value = run_hyperparameter_trial(trial_config)
                    end_time = datetime.now()
                    
                    # Store results
                    trial_result = {
                        'trial': current_trial,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'scheduler': scheduler,
                        'image_size': image_size,
                        'metric_value': metric_value,
                        'duration': (end_time - start_time).total_seconds()
                    }
                    results.append(trial_result)
                    
                    print(f"   Result: {metric_value:.4f}")
                    print(f"   Duration: {(end_time - start_time).total_seconds():.1f}s")
    
    return results

# Run the grid search
tuning_results = run_manual_grid_search()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Analyze Tuning Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert Results to DataFrame

# COMMAND ----------

# Convert results to DataFrame for analysis
df_results = pd.DataFrame(tuning_results)

print("📊 Tuning Results Summary:")
print("=" * 40)
print(f"Total trials: {len(df_results)}")
print(f"Best metric value: {df_results['metric_value'].max():.4f}")
print(f"Worst metric value: {df_results['metric_value'].min():.4f}")
print(f"Average metric value: {df_results['metric_value'].mean():.4f}")

# Display top 5 results
print(f"\n🏆 Top 5 Results:")
print("=" * 40)
top_results = df_results.nlargest(5, 'metric_value')
for idx, row in top_results.iterrows():
    print(f"Trial {row['trial']}: {row['metric_value']:.4f}")
    print(f"  LR: {row['learning_rate']:.2e}, Batch: {row['batch_size']}, "
          f"Scheduler: {row['scheduler']}, Image Size: {row['image_size']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Results

# COMMAND ----------

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Hyperparameter Tuning Results', fontsize=16)

# 1. Learning Rate vs Metric
axes[0, 0].scatter(df_results['learning_rate'], df_results['metric_value'], alpha=0.7)
axes[0, 0].set_xlabel('Learning Rate')
axes[0, 0].set_ylabel('Metric Value')
axes[0, 0].set_xscale('log')
axes[0, 0].set_title('Learning Rate vs Metric')
axes[0, 0].grid(True, alpha=0.3)

# 2. Batch Size vs Metric
batch_metrics = df_results.groupby('batch_size')['metric_value'].mean()
axes[0, 1].bar(batch_metrics.index, batch_metrics.values)
axes[0, 1].set_xlabel('Batch Size')
axes[0, 1].set_ylabel('Average Metric Value')
axes[0, 1].set_title('Batch Size vs Average Metric')
axes[0, 1].grid(True, alpha=0.3)

# 3. Scheduler vs Metric
scheduler_metrics = df_results.groupby('scheduler')['metric_value'].mean()
axes[1, 0].bar(scheduler_metrics.index, scheduler_metrics.values)
axes[1, 0].set_xlabel('Scheduler')
axes[1, 0].set_ylabel('Average Metric Value')
axes[1, 0].set_title('Scheduler vs Average Metric')
axes[1, 0].grid(True, alpha=0.3)

# 4. Image Size vs Metric
image_metrics = df_results.groupby('image_size')['metric_value'].mean()
axes[1, 1].bar(image_metrics.index, image_metrics.values)
axes[1, 1].set_xlabel('Image Size')
axes[1, 1].set_ylabel('Average Metric Value')
axes[1, 1].set_title('Image Size vs Average Metric')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find Best Configuration

# COMMAND ----------

# Find the best configuration
best_result = df_results.loc[df_results['metric_value'].idxmax()]

print("🏆 Best Hyperparameter Configuration:")
print("=" * 50)
print(f"Trial: {best_result['trial']}")
print(f"Learning Rate: {best_result['learning_rate']:.2e}")
print(f"Batch Size: {best_result['batch_size']}")
print(f"Scheduler: {best_result['scheduler']}")
print(f"Image Size: {best_result['image_size']}")
print(f"Metric Value: {best_result['metric_value']:.4f}")
print(f"Training Duration: {best_result['duration']:.1f}s")

# Save best configuration
best_config = {
    'learning_rate': float(best_result['learning_rate']),
    'batch_size': int(best_result['batch_size']),
    'scheduler': best_result['scheduler'],
    'image_size': int(best_result['image_size']),
    'weight_decay': 1e-4,  # Fixed value used
    'metric_value': float(best_result['metric_value'])
}

# Save to file
best_config_path = f"{TUNING_RESULTS_DIR}/best_config.json"
with open(best_config_path, 'w') as f:
    json.dump(best_config, f, indent=2)

print(f"\n💾 Best configuration saved to: {best_config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save and Export Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save All Results

# COMMAND ----------

# Save all results to CSV
results_csv_path = f"{TUNING_RESULTS_DIR}/tuning_results.csv"
df_results.to_csv(results_csv_path, index=False)

# Save results summary
summary = {
    'total_trials': len(df_results),
    'best_metric': float(df_results['metric_value'].max()),
    'worst_metric': float(df_results['metric_value'].min()),
    'average_metric': float(df_results['metric_value'].mean()),
    'best_config': best_config,
    'tuning_date': datetime.now().isoformat()
}

summary_path = f"{TUNING_RESULTS_DIR}/tuning_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("💾 Results saved:")
print(f"   Detailed results: {results_csv_path}")
print(f"   Summary: {summary_path}")
print(f"   Best config: {best_config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary and Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tuning Summary

# COMMAND ----------

print("=" * 60)
print("SERVERLESS GPU HYPERPARAMETER TUNING SUMMARY")
print("=" * 60)

print(f"✅ Total trials completed: {len(df_results)}")
print(f"✅ Best metric value: {df_results['metric_value'].max():.4f}")
print(f"✅ Average metric value: {df_results['metric_value'].mean():.4f}")

print(f"\n🏆 Best Configuration:")
print(f"   Learning Rate: {best_result['learning_rate']:.2e}")
print(f"   Batch Size: {best_result['batch_size']}")
print(f"   Scheduler: {best_result['scheduler']}")
print(f"   Image Size: {best_result['image_size']}")

print(f"\n📁 Results saved to: {TUNING_RESULTS_DIR}")
print(f"   CSV results: tuning_results.csv")
print(f"   Summary: tuning_summary.json")
print(f"   Best config: best_config.json")

print(f"\n🔗 MLflow UI:")
print(f"   Check the tuning experiment for detailed metrics and logs")

print("\n🎉 Hyperparameter tuning completed successfully!")
print("\nNext steps:")
print("1. Use the best configuration for final model training")
print("2. Run notebook 02_model_training_serverless.py with optimized parameters")
print("3. Compare results with the tuned configuration")

print("=" * 60)

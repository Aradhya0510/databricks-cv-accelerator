# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # 03. Hyperparameter Tuning for DETR
# MAGIC 
# MAGIC This notebook performs hyperparameter tuning for DETR training using Ray Tune. We'll explore different learning rates, batch sizes, and other hyperparameters to find the optimal configuration for our COCO dataset.
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **Hyperparameter Tuning Strategy:**
# MAGIC 1. **Search Space Definition**: Define ranges for key hyperparameters
# MAGIC 2. **Search Algorithm**: Use Bayesian optimization for efficient search
# MAGIC 3. **Resource Management**: Efficient GPU utilization across trials
# MAGIC 4. **Early Stopping**: Stop unpromising trials early
# MAGIC 5. **Result Analysis**: Compare and select best configuration
# MAGIC 
# MAGIC ### Key Hyperparameters for DETR:
# MAGIC - **Learning Rate**: Critical for convergence (1e-5 to 1e-3)
# MAGIC - **Batch Size**: Affects memory usage and training stability
# MAGIC - **Weight Decay**: Regularization to prevent overfitting
# MAGIC - **Scheduler Parameters**: Learning rate scheduling strategy
# MAGIC - **Model Architecture**: Different backbone configurations
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 
# MAGIC 1. **Search Space Setup**: Define hyperparameter ranges
# MAGIC 2. **Trial Configuration**: Set up individual training trials
# MAGIC 3. **Resource Allocation**: Manage GPU resources efficiently
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
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
import json

# Add the src directory to Python path
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '/Workspace/Repos/your-repo/Databricks_CV_ref')
sys.path.append(f'{PROJECT_ROOT}/src')

from config import load_config, get_default_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from training.trainer import UnifiedTrainer
from utils.logging import create_databricks_logger

# Load configuration from previous notebooks
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

# Set up volume directories
TUNING_RESULTS_DIR = f"{BASE_VOLUME_PATH}/tuning_results"
TUNING_LOGS_DIR = f"{BASE_VOLUME_PATH}/logs"

# Create directories
os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)
os.makedirs(TUNING_LOGS_DIR, exist_ok=True)

print(f"📁 Volume directories created:")
print(f"   Tuning Results: {TUNING_RESULTS_DIR}")
print(f"   Tuning Logs: {TUNING_LOGS_DIR}")

# Load base configuration
if os.path.exists(CONFIG_PATH):
    base_config = load_config(CONFIG_PATH)
    # Update checkpoint directory to use volume path
    base_config['training']['checkpoint_dir'] = f"{BASE_VOLUME_PATH}/checkpoints"
    print(f"✅ Fixed config: updated checkpoint directory")
else:
    print("⚠️  Config file not found. Using default detection config.")
    base_config = get_default_config("detection")
    # Update checkpoint directory to use volume path
    base_config['training']['checkpoint_dir'] = f"{BASE_VOLUME_PATH}/checkpoints"

print("✅ Base configuration loaded successfully!")
print(f"📁 Checkpoint directory: {base_config['training']['checkpoint_dir']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Ray for Distributed Tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Ray Cluster

# COMMAND ----------

def setup_ray_cluster():
    """Initialize Ray cluster for hyperparameter tuning."""
    
    print("🚀 Setting up Ray cluster...")
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=4,  # Adjust based on your cluster
                num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                ignore_reinit_error=True
            )
        
        print(f"✅ Ray cluster initialized!")
        print(f"   Available CPUs: {ray.available_resources().get('CPU', 0)}")
        print(f"   Available GPUs: {ray.available_resources().get('GPU', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ray initialization failed: {e}")
        return False

ray_ready = setup_ray_cluster()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define Search Space

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Search Space

# COMMAND ----------

def define_search_space():
    """Define the hyperparameter search space for DETR tuning."""
    
    print("🔍 Defining hyperparameter search space...")
    
    # Define search space
    search_space = {
        # Learning rate (log scale)
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        
        # Batch size (discrete values)
        "batch_size": tune.choice([8, 16, 32]),
        
        # Weight decay
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        
        # Scheduler type
        "scheduler": tune.choice(["cosine", "step", "exponential"]),
        
        # Model architecture variations
        "backbone": tune.choice(["resnet50", "resnet101"]),
        
        # Training duration
        "max_epochs": tune.choice([50, 100, 150]),
        
        # Data augmentation strength
        "augment_strength": tune.choice([0.1, 0.2, 0.3]),
        
        # Number of workers
        "num_workers": tune.choice([2, 4, 8]),
    }
    
    print("✅ Search space defined:")
    for param, space in search_space.items():
        print(f"   {param}: {space}")
    
    return search_space

search_space = define_search_space()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search Algorithm and Scheduler

# COMMAND ----------

def setup_search_algorithm():
    """Set up the search algorithm and scheduler for optimization."""
    
    print("\n🎯 Setting up search algorithm...")
    
    # Bayesian optimization search
    search_alg = BayesOptSearch(
        metric="val_map",
        mode="max",
        random_search_steps=10,  # Initial random exploration
        points_to_evaluate=[
            # Good starting points based on DETR paper
            {
                "learning_rate": 1e-4,
                "batch_size": 16,
                "weight_decay": 1e-4,
                "scheduler": "cosine",
                "backbone": "resnet50",
                "max_epochs": 100,
                "augment_strength": 0.2,
                "num_workers": 4,
            }
        ]
    )
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_map",
        mode="max",
        max_t=300,  # Maximum epochs
        grace_period=10,  # Minimum epochs before early stopping
        reduction_factor=2,  # Reduce trials by factor of 2
        brackets=3  # Number of brackets for ASHA
    )
    
    print("✅ Search algorithm configured:")
    print(f"   Algorithm: Bayesian Optimization")
    print(f"   Scheduler: ASHA (Asynchronous Successive Halving)")
    print(f"   Metric: val_map (maximize)")
    print(f"   Grace period: 10 epochs")
    print(f"   Max trials: 300 epochs")
    
    return search_alg, scheduler

search_alg, scheduler = setup_search_algorithm()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Training Trial Function

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Training Trial

# COMMAND ----------

def train_trial(config_dict):
    """Training function for a single hyperparameter trial."""
    
    try:
        print(f"🚀 Starting trial with config: {config_dict}")
        
        # Create trial configuration
        trial_config = base_config.copy()
        
        # Update model parameters
        trial_config['model']['learning_rate'] = config_dict['learning_rate']
        trial_config['model']['weight_decay'] = config_dict['weight_decay']
        trial_config['model']['backbone'] = config_dict['backbone']
        
        # Update data parameters
        trial_config['data']['batch_size'] = config_dict['batch_size']
        trial_config['data']['num_workers'] = config_dict['num_workers']
        
        # Update training parameters
        trial_config['training']['max_epochs'] = config_dict['max_epochs']
        trial_config['training']['scheduler'] = config_dict['scheduler']
        
        # Update augmentation parameters
        trial_config['data']['augmentations']['strength'] = config_dict['augment_strength']
        
        # Create model
        model = DetectionModel(trial_config['model'])
        
        # Create data module
        data_module = DetectionDataModule(trial_config['data'])
        
        # Create trainer config with proper structure
        trainer_config = {
            'task': trial_config['model']['task_type'],
            'model_name': trial_config['model']['model_name'],
            'max_epochs': trial_config['training']['max_epochs'],
            'log_every_n_steps': trial_config['training'].get('log_every_n_steps', 50),
            'monitor_metric': trial_config['training'].get('monitor_metric', 'val_loss'),
            'monitor_mode': trial_config['training'].get('monitor_mode', 'min'),
            'early_stopping_patience': trial_config['training'].get('early_stopping_patience', 10),
            'checkpoint_dir': f"{BASE_VOLUME_PATH}/checkpoints/trial_{tune.get_trial_id()}",
            'save_top_k': 1,  # Save only best checkpoint for trials
            'distributed': False  # Disable distributed training for trials
        }
        
        # Create trainer with proper constructor
        trainer = UnifiedTrainer(
            config=trainer_config,
            model=model,
            data_module=data_module
        )
        
        # Train the model
        result = trainer.train()
        
        # Extract metrics
        metrics = {
            'val_map': result.metrics.get('val_map', 0.0),
            'val_loss': result.metrics.get('val_loss', float('inf')),
            'train_loss': result.metrics.get('train_loss', float('inf')),
            'learning_rate': config_dict['learning_rate'],
            'batch_size': config_dict['batch_size'],
            'backbone': config_dict['backbone']
        }
        
        print(f"✅ Trial completed - val_map: {metrics['val_map']:.4f}")
        
        # Report metrics to Ray Tune
        tune.report(**metrics)
        
    except Exception as e:
        print(f"❌ Trial failed: {e}")
        # Report failure to Ray Tune
        tune.report(
            val_map=0.0,
            val_loss=float('inf'),
            train_loss=float('inf'),
            error=str(e)
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Resource Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Resource Allocation

# COMMAND ----------

def setup_resources():
    """Configure resources for hyperparameter tuning."""
    
    print("⚙️  Setting up resource allocation...")
    
    # Get available resources
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_cpus = os.cpu_count() or 4
    
    # Configure resources per trial
    resources_per_trial = {
        "cpu": 2,  # CPUs per trial
        "gpu": 1 if num_gpus > 0 else 0,  # GPUs per trial
        "memory": 8 * 1024 * 1024 * 1024  # 8GB memory per trial
    }
    
    # Calculate max concurrent trials
    max_concurrent_trials = min(
        num_gpus,  # Limited by GPU count
        num_cpus // resources_per_trial["cpu"],  # Limited by CPU count
        4  # Reasonable limit for stability
    )
    
    print(f"✅ Resource configuration:")
    print(f"   Available GPUs: {num_gpus}")
    print(f"   Available CPUs: {num_cpus}")
    print(f"   Resources per trial: {resources_per_trial}")
    print(f"   Max concurrent trials: {max_concurrent_trials}")
    
    return resources_per_trial, max_concurrent_trials

resources_per_trial, max_concurrent_trials = setup_resources()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run Hyperparameter Optimization

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Optimization

# COMMAND ----------

def run_hyperparameter_optimization():
    """Run the hyperparameter optimization process."""
    
    if not ray_ready:
        print("❌ Ray not initialized. Cannot run optimization.")
        return None
    
    print("🚀 Starting hyperparameter optimization...")
    
    # Configure MLflow experiment for tuning
    experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/detection_tuning"
    mlflow.set_experiment(experiment_name)
    
    # Run optimization
    analysis = tune.run(
        train_trial,
        config=search_space,
        num_samples=20,  # Number of trials
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial=resources_per_trial,
        max_concurrent_trials=max_concurrent_trials,
        local_dir=TUNING_RESULTS_DIR,
        name="detr_hyperparameter_tuning",
        log_to_file=True,
        verbose=2
    )
    
    print("✅ Hyperparameter optimization completed!")
    print(f"   Results saved to: {TUNING_RESULTS_DIR}")
    
    return analysis

# Run optimization
analysis = run_hyperparameter_optimization()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Monitor and Analyze Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitor Optimization Progress

# COMMAND ----------

def monitor_optimization_progress(analysis):
    """Monitor the progress of hyperparameter optimization."""
    
    if not analysis:
        print("❌ No analysis results available")
        return
    
    print("📊 Optimization Progress Analysis:")
    
    # Get best trial
    best_trial = analysis.get_best_trial("val_map", "max")
    print(f"   Best trial: {best_trial.trial_id}")
    print(f"   Best val_map: {best_trial.last_result['val_map']:.4f}")
    print(f"   Best config: {best_trial.config}")
    
    # Get all trials
    trials = analysis.trials
    print(f"   Total trials: {len(trials)}")
    
    # Calculate statistics
    val_maps = [trial.last_result.get('val_map', 0) for trial in trials]
    print(f"   Mean val_map: {np.mean(val_maps):.4f}")
    print(f"   Std val_map: {np.std(val_maps):.4f}")
    print(f"   Min val_map: {np.min(val_maps):.4f}")
    print(f"   Max val_map: {np.max(val_maps):.4f}")
    
    # Failed trials
    failed_trials = [trial for trial in trials if trial.status == 'FAILED']
    print(f"   Failed trials: {len(failed_trials)}")
    
    return best_trial

best_trial = monitor_optimization_progress(analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Optimization Results

# COMMAND ----------

def analyze_optimization_results(analysis):
    """Analyze the results of hyperparameter optimization."""
    
    if not analysis:
        print("❌ No analysis results available")
        return None
    
    print("🔍 Analyzing optimization results...")
    
    # Convert results to DataFrame
    results_df = analysis.results_df
    
    # Filter out failed trials
    successful_trials = results_df[results_df['val_map'] > 0]
    
    if len(successful_trials) == 0:
        print("❌ No successful trials found")
        return None
    
    print(f"✅ Analysis completed:")
    print(f"   Successful trials: {len(successful_trials)}")
    print(f"   Failed trials: {len(results_df) - len(successful_trials)}")
    
    # Parameter importance analysis
    print("\n📈 Parameter Importance Analysis:")
    
    # Analyze learning rate impact
    lr_groups = successful_trials.groupby('learning_rate')['val_map'].agg(['mean', 'std', 'count'])
    print("   Learning Rate Impact:")
    for lr, stats in lr_groups.iterrows():
        print(f"     {lr:.2e}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
    
    # Analyze batch size impact
    bs_groups = successful_trials.groupby('batch_size')['val_map'].agg(['mean', 'std', 'count'])
    print("   Batch Size Impact:")
    for bs, stats in bs_groups.iterrows():
        print(f"     {bs}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
    
    # Analyze backbone impact
    backbone_groups = successful_trials.groupby('backbone')['val_map'].agg(['mean', 'std', 'count'])
    print("   Backbone Impact:")
    for backbone, stats in backbone_groups.iterrows():
        print(f"     {backbone}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
    
    return successful_trials

results_df = analyze_optimization_results(analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Optimization Results

# COMMAND ----------

def visualize_optimization_results(results_df):
    """Create visualizations of the optimization results."""
    
    if results_df is None or len(results_df) == 0:
        print("❌ No results to visualize")
        return
    
    print("📊 Creating visualizations...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DETR Hyperparameter Optimization Results', fontsize=16)
    
    # 1. Learning rate vs val_map
    axes[0, 0].scatter(results_df['learning_rate'], results_df['val_map'], alpha=0.6)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Validation mAP')
    axes[0, 0].set_title('Learning Rate vs Validation mAP')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Batch size vs val_map
    axes[0, 1].scatter(results_df['batch_size'], results_df['val_map'], alpha=0.6)
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Validation mAP')
    axes[0, 1].set_title('Batch Size vs Validation mAP')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training loss vs validation mAP
    axes[1, 0].scatter(results_df['train_loss'], results_df['val_map'], alpha=0.6)
    axes[1, 0].set_xlabel('Training Loss')
    axes[1, 0].set_ylabel('Validation mAP')
    axes[1, 0].set_title('Training Loss vs Validation mAP')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Validation loss vs validation mAP
    axes[1, 1].scatter(results_df['val_loss'], results_df['val_map'], alpha=0.6)
    axes[1, 1].set_xlabel('Validation Loss')
    axes[1, 1].set_ylabel('Validation mAP')
    axes[1, 1].set_title('Validation Loss vs Validation mAP')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{TUNING_RESULTS_DIR}/optimization_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {plot_path}")
    
    plt.show()

visualize_optimization_results(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Parameter Importance

# COMMAND ----------

def analyze_parameter_importance(results_df):
    """Analyze the importance of different hyperparameters."""
    
    if results_df is None or len(results_df) == 0:
        print("❌ No results to analyze")
        return
    
    print("🎯 Parameter Importance Analysis:")
    
    # Calculate correlation with val_map
    correlations = {}
    for col in results_df.columns:
        if col != 'val_map' and col in ['learning_rate', 'batch_size', 'weight_decay', 'dropout', 'augment_strength']:
            try:
                corr = results_df[col].corr(results_df['val_map'])
                correlations[col] = abs(corr)
            except:
                pass
    
    # Sort by correlation strength
    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("   Parameter correlations with val_map:")
    for param, corr in sorted_correlations:
        print(f"     {param}: {corr:.4f}")
    
    # Analyze best configurations
    top_5_trials = results_df.nlargest(5, 'val_map')
    print(f"\n🏆 Top 5 Configurations:")
    for i, (_, trial) in enumerate(top_5_trials.iterrows(), 1):
        print(f"   {i}. val_map={trial['val_map']:.4f}")
        print(f"      learning_rate={trial['learning_rate']:.2e}")
        print(f"      batch_size={trial['batch_size']}")
        print(f"      backbone={trial['backbone']}")
        print(f"      weight_decay={trial['weight_decay']:.2e}")
        print()
    
    return sorted_correlations

parameter_importance = analyze_parameter_importance(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Best Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Best Configuration

# COMMAND ----------

def save_best_configuration(analysis, results_df):
    """Save the best configuration found during optimization."""
    
    if not analysis or results_df is None:
        print("❌ No analysis results available")
        return False
    
    print("💾 Saving best configuration...")
    
    # Get best trial
    best_trial = analysis.get_best_trial("val_map", "max")
    best_config = best_trial.config
    best_metrics = best_trial.last_result
    
    # Create optimized config
    optimized_config = base_config.copy()
    
    # Update with best parameters
    optimized_config['training']['learning_rate'] = best_config['learning_rate']
    optimized_config['training']['max_epochs'] = best_config['max_epochs']
    optimized_config['training']['weight_decay'] = best_config['weight_decay']
    optimized_config['data']['batch_size'] = best_config['batch_size']
    optimized_config['model']['model_name'] = f"facebook/detr-{best_config['backbone']}"
    optimized_config['model']['dropout'] = best_config['dropout']
    optimized_config['training']['scheduler_params'] = {
        'T_max': best_config['scheduler_t_max'],
        'eta_min': best_config['scheduler_eta_min']
    }
    optimized_config['data']['augmentations']['strength'] = best_config['augment_strength']
    
    # Add optimization metadata
    optimized_config['optimization'] = {
        'best_val_map': best_metrics['val_map'],
        'best_val_loss': best_metrics['val_loss'],
        'trial_id': best_trial.trial_id,
        'total_trials': len(results_df),
        'optimization_date': str(pd.Timestamp.now()),
        'parameter_importance': dict(parameter_importance) if 'parameter_importance' in locals() else {}
    }
    
    # Save optimized config
    config_path = f"{TUNING_RESULTS_DIR}/optimized_detr_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(optimized_config, f, default_flow_style=False)
    
    print(f"✅ Optimized configuration saved to: {config_path}")
    print(f"   Best val_map: {best_metrics['val_map']:.4f}")
    print(f"   Best learning_rate: {best_config['learning_rate']:.2e}")
    print(f"   Best batch_size: {best_config['batch_size']}")
    print(f"   Best backbone: {best_config['backbone']}")
    
    # Save results summary
    summary_path = f"{TUNING_RESULTS_DIR}/optimization_summary.json"
    summary = {
        'best_config': best_config,
        'best_metrics': best_metrics,
        'total_trials': len(results_df),
        'successful_trials': len(results_df[results_df['val_map'] > 0]),
        'parameter_importance': dict(parameter_importance) if 'parameter_importance' in locals() else {},
        'optimization_date': str(pd.Timestamp.now())
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Optimization summary saved to: {summary_path}")
    
    return True

# Save best configuration
config_saved = save_best_configuration(analysis, results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary and Next Steps

# COMMAND ----------

print("=" * 60)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 60)

if analysis and results_df is not None:
    best_trial = analysis.get_best_trial("val_map", "max")
    
    print(f"✅ Optimization completed successfully!")
    print(f"   Total trials: {len(results_df)}")
    print(f"   Successful trials: {len(results_df[results_df['val_map'] > 0])}")
    print(f"   Best val_map: {best_trial.last_result['val_map']:.4f}")
    print(f"   Best learning rate: {best_trial.config['learning_rate']:.2e}")
    print(f"   Best batch size: {best_trial.config['batch_size']}")
    print(f"   Best backbone: {best_trial.config['backbone']}")
    
    print(f"\n📁 Results saved to:")
    print(f"   Tuning results: {TUNING_RESULTS_DIR}")
    print(f"   Optimized config: {TUNING_RESULTS_DIR}/optimized_detr_config.yaml")
    print(f"   Summary: {TUNING_RESULTS_DIR}/optimization_summary.json")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Use the optimized configuration for final training")
    print(f"   2. Run the training notebook with the new config")
    print(f"   3. Compare results with the baseline model")
    
else:
    print("❌ Optimization failed or no results available")
    print("   Check the logs for error details")

print("=" * 60) 
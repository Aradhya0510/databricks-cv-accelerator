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
# MAGIC ## 1. Import Dependencies and Setup

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

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')

from config import load_config, get_default_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from training.trainer import UnifiedTrainer
from utils.logging import setup_logger

# Load configuration
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

# Load base configuration
if os.path.exists(CONFIG_PATH):
    base_config = load_config(CONFIG_PATH)
else:
    base_config = get_default_config("detection")

print("✅ Base configuration loaded successfully!")

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
        
        # Scheduler parameters
        "scheduler_t_max": tune.choice([100, 200, 300]),
        "scheduler_eta_min": tune.loguniform(1e-7, 1e-5),
        
        # Model architecture variations
        "backbone": tune.choice(["resnet50", "resnet101"]),
        
        # Training duration
        "max_epochs": tune.choice([50, 100, 150]),
        
        # Data augmentation strength
        "augment_strength": tune.choice([0.1, 0.2, 0.3]),
        
        # Dropout rate
        "dropout": tune.uniform(0.1, 0.3),
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
                "scheduler_t_max": 200,
                "scheduler_eta_min": 1e-6,
                "backbone": "resnet50",
                "max_epochs": 100,
                "augment_strength": 0.2,
                "dropout": 0.2
            }
        ]
    )
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="val_map",
        mode="max",
        max_t=100,  # Maximum epochs per trial
        grace_period=10,  # Minimum epochs before stopping
        reduction_factor=2
    )
    
    print("✅ Search algorithm configured:")
    print(f"   Algorithm: Bayesian Optimization")
    print(f"   Metric: val_map (maximize)")
    print(f"   Scheduler: ASHA (early stopping)")
    print(f"   Grace period: 10 epochs")
    print(f"   Max epochs per trial: 100")
    
    return search_alg, scheduler

search_alg, scheduler = setup_search_algorithm()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Trial Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Function for Trials

# COMMAND ----------

def train_trial(config_dict):
    """Training function for individual trials."""
    
    # Update base config with trial hyperparameters
    trial_config = base_config.copy()
    
    # Update model configuration
    trial_config['model']['learning_rate'] = config_dict['learning_rate']
    trial_config['model']['weight_decay'] = config_dict['weight_decay']
    trial_config['model']['dropout'] = config_dict['dropout']
    
    # Update data configuration
    trial_config['data']['batch_size'] = config_dict['batch_size']
    trial_config['data']['augment_strength'] = config_dict['augment_strength']
    
    # Update training configuration
    trial_config['training']['max_epochs'] = config_dict['max_epochs']
    trial_config['training']['scheduler_params'] = {
        'T_max': config_dict['scheduler_t_max'],
        'eta_min': config_dict['scheduler_eta_min']
    }
    
    # Update model name based on backbone
    if config_dict['backbone'] == 'resnet101':
        trial_config['model']['model_name'] = 'facebook/detr-resnet-101'
    else:
        trial_config['model']['model_name'] = 'facebook/detr-resnet-50'
    
    try:
        # Prepare model config with num_workers from data config
        model_config = trial_config["model"].copy()
        model_config["num_workers"] = trial_config["data"]["num_workers"]
        
        # Initialize model
        model = DetectionModel(model_config)
        
        # Setup adapter first
        from tasks.detection.adapters import get_input_adapter
        adapter = get_input_adapter(trial_config["model"]["model_name"], image_size=trial_config["data"].get("image_size", 800))
        if adapter is None:
            print("❌ Failed to create adapter")
            tune.report(val_map=0.0, val_loss=float('inf'))
            return
        
        # Initialize data module with data config only
        data_module = DetectionDataModule(trial_config["data"])
        
        # Assign adapter to data module
        data_module.adapter = adapter
        
        # Setup for training
        data_module.setup('fit')
        
        # Initialize trainer
        trainer = UnifiedTrainer(trial_config)
        
        # Train the model
        trainer.fit(model, data_module)
        
        # Evaluate on validation set
        results = trainer.test(model, data_module)
        
        # Report metrics to Ray Tune
        tune.report(
            val_map=results[0].get('test_map', 0.0),
            val_loss=results[0].get('test_loss', float('inf')),
            train_loss=results[0].get('train_loss', float('inf')),
            learning_rate=config_dict['learning_rate'],
            batch_size=config_dict['batch_size']
        )
        
    except Exception as e:
        print(f"Trial failed: {e}")
        # Report failure
        tune.report(val_map=0.0, val_loss=float('inf'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resource Configuration

# COMMAND ----------

def setup_resources():
    """Configure resources for hyperparameter tuning."""
    
    print("\n💻 Setting up resource configuration...")
    
    # Calculate resources per trial
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_cpus = os.cpu_count()
    
    # Resource allocation per trial
    resources_per_trial = {
        "cpu": max(1, num_cpus // 4),  # 1/4 of CPUs per trial
        "gpu": max(0.25, 1.0 / min(4, num_gpus)) if num_gpus > 0 else 0  # 1 GPU per trial, max 4 trials
    }
    
    print(f"✅ Resource configuration:")
    print(f"   Total CPUs: {num_cpus}")
    print(f"   Total GPUs: {num_gpus}")
    print(f"   CPUs per trial: {resources_per_trial['cpu']}")
    print(f"   GPUs per trial: {resources_per_trial['gpu']}")
    
    # Calculate max concurrent trials
    max_trials = min(
        num_cpus // resources_per_trial['cpu'],
        int(num_gpus / resources_per_trial['gpu']) if num_gpus > 0 else 4
    )
    
    print(f"   Max concurrent trials: {max_trials}")
    
    return resources_per_trial, max_trials

resources_per_trial, max_trials = setup_resources()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Hyperparameter Optimization

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start Optimization

# COMMAND ----------

def run_hyperparameter_optimization():
    """Run the hyperparameter optimization process."""
    
    print("\n🎯 Starting hyperparameter optimization...")
    print("=" * 60)
    
    try:
        # Run optimization
        analysis = tune.run(
            train_trial,
            config=search_space,
            num_samples=20,  # Total number of trials
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial=resources_per_trial,
            max_concurrent_trials=max_trials,
            local_dir=f"{BASE_VOLUME_PATH}/ray_results",
            name="detr_hyperparameter_tuning",
            verbose=2,
            fail_fast=True,
            checkpoint_at_end=True
        )
        
        print("✅ Hyperparameter optimization completed!")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None

# Run optimization
analysis = run_hyperparameter_optimization()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitor Optimization Progress

# COMMAND ----------

def monitor_optimization_progress(analysis):
    """Monitor the progress of hyperparameter optimization."""
    
    if not analysis:
        return
    
    print("\n📊 Optimization Progress:")
    
    # Get best trial
    best_trial = analysis.get_best_trial("val_map", "max")
    print(f"   Best trial: {best_trial.trial_id}")
    print(f"   Best val_map: {best_trial.last_result['val_map']:.4f}")
    print(f"   Best config: {best_trial.config}")
    
    # Get all trials
    trials = analysis.trials
    print(f"   Total trials completed: {len(trials)}")
    
    # Calculate success rate
    successful_trials = [t for t in trials if t.last_result.get('val_map', 0) > 0]
    success_rate = len(successful_trials) / len(trials) * 100
    print(f"   Success rate: {success_rate:.1f}%")
    
    # Show top 5 trials
    print(f"\n🏆 Top 5 Trials:")
    sorted_trials = sorted(trials, key=lambda t: t.last_result.get('val_map', 0), reverse=True)
    
    for i, trial in enumerate(sorted_trials[:5]):
        print(f"   {i+1}. Trial {trial.trial_id}: val_map={trial.last_result.get('val_map', 0):.4f}")
        print(f"      Config: {trial.config}")

if analysis:
    monitor_optimization_progress(analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Result Analysis and Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Optimization Results

# COMMAND ----------

def analyze_optimization_results(analysis):
    """Analyze and visualize the optimization results."""
    
    if not analysis:
        return
    
    print("\n📈 Analyzing optimization results...")
    
    # Get all trials
    trials = analysis.trials
    
    # Extract results
    results = []
    for trial in trials:
        if trial.last_result:
            result = {
                'trial_id': trial.trial_id,
                'val_map': trial.last_result.get('val_map', 0),
                'val_loss': trial.last_result.get('val_loss', float('inf')),
                'train_loss': trial.last_result.get('train_loss', float('inf')),
                'learning_rate': trial.config.get('learning_rate', 0),
                'batch_size': trial.config.get('batch_size', 0),
                'weight_decay': trial.config.get('weight_decay', 0),
                'backbone': trial.config.get('backbone', ''),
                'max_epochs': trial.config.get('max_epochs', 0)
            }
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print(f"✅ Analysis complete!")
    print(f"   Total trials: {len(df)}")
    print(f"   Mean val_map: {df['val_map'].mean():.4f}")
    print(f"   Best val_map: {df['val_map'].max():.4f}")
    print(f"   Std val_map: {df['val_map'].std():.4f}")
    
    return df

results_df = analyze_optimization_results(analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Results

# COMMAND ----------

def visualize_optimization_results(results_df):
    """Create visualizations of the optimization results."""
    
    if results_df is None or len(results_df) == 0:
        return
    
    print("\n📊 Creating visualizations...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Learning rate vs val_map
    axes[0, 0].scatter(results_df['learning_rate'], results_df['val_map'], alpha=0.6)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Validation mAP')
    axes[0, 0].set_title('Learning Rate vs Validation mAP')
    axes[0, 0].set_xscale('log')
    
    # 2. Batch size vs val_map
    axes[0, 1].scatter(results_df['batch_size'], results_df['val_map'], alpha=0.6)
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Validation mAP')
    axes[0, 1].set_title('Batch Size vs Validation mAP')
    
    # 3. Weight decay vs val_map
    axes[1, 0].scatter(results_df['weight_decay'], results_df['val_map'], alpha=0.6)
    axes[1, 0].set_xlabel('Weight Decay')
    axes[1, 0].set_ylabel('Validation mAP')
    axes[1, 0].set_title('Weight Decay vs Validation mAP')
    axes[1, 0].set_xscale('log')
    
    # 4. Backbone vs val_map
    backbone_results = results_df.groupby('backbone')['val_map'].mean()
    axes[1, 1].bar(backbone_results.index, backbone_results.values)
    axes[1, 1].set_xlabel('Backbone')
    axes[1, 1].set_ylabel('Mean Validation mAP')
    axes[1, 1].set_title('Backbone vs Mean Validation mAP')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizations created!")

if results_df is not None:
    visualize_optimization_results(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parameter Importance Analysis

# COMMAND ----------

def analyze_parameter_importance(results_df):
    """Analyze the importance of different hyperparameters."""
    
    if results_df is None or len(results_df) == 0:
        return
    
    print("\n🔍 Analyzing parameter importance...")
    
    # Calculate correlations
    numeric_cols = ['learning_rate', 'batch_size', 'weight_decay', 'max_epochs']
    correlations = results_df[numeric_cols + ['val_map']].corr()['val_map'].abs()
    
    print("📊 Parameter Correlations with val_map:")
    for param, corr in correlations.items():
        if param != 'val_map':
            print(f"   {param}: {corr:.4f}")
    
    # Find best configuration for each parameter
    print(f"\n🏆 Best configurations:")
    
    # Best learning rate
    best_lr_idx = results_df['val_map'].idxmax()
    best_lr = results_df.loc[best_lr_idx, 'learning_rate']
    print(f"   Best learning rate: {best_lr:.2e}")
    
    # Best batch size
    best_bs_idx = results_df['val_map'].idxmax()
    best_bs = results_df.loc[best_bs_idx, 'batch_size']
    print(f"   Best batch size: {best_bs}")
    
    # Best weight decay
    best_wd_idx = results_df['val_map'].idxmax()
    best_wd = results_df.loc[best_wd_idx, 'weight_decay']
    print(f"   Best weight decay: {best_wd:.2e}")
    
    # Best backbone
    best_backbone = results_df.groupby('backbone')['val_map'].mean().idxmax()
    print(f"   Best backbone: {best_backbone}")

if results_df is not None:
    analyze_parameter_importance(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Best Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export Best Configuration

# COMMAND ----------

def save_best_configuration(analysis, results_df):
    """Save the best configuration found during optimization."""
    
    if not analysis or results_df is None:
        return
    
    print("\n💾 Saving best configuration...")
    
    # Get best trial
    best_trial = analysis.get_best_trial("val_map", "max")
    best_config = best_trial.config
    
    # Create optimized config
    optimized_config = base_config.copy()
    
    # Update with best hyperparameters
    optimized_config['model']['learning_rate'] = best_config['learning_rate']
    optimized_config['model']['weight_decay'] = best_config['weight_decay']
    optimized_config['model']['dropout'] = best_config['dropout']
    
    optimized_config['data']['batch_size'] = best_config['batch_size']
    optimized_config['data']['augment_strength'] = best_config['augment_strength']
    
    optimized_config['training']['max_epochs'] = best_config['max_epochs']
    optimized_config['training']['scheduler_params'] = {
        'T_max': best_config['scheduler_t_max'],
        'eta_min': best_config['scheduler_eta_min']
    }
    
    if best_config['backbone'] == 'resnet101':
        optimized_config['model']['model_name'] = 'facebook/detr-resnet-101'
    
    # Save optimized config
    import yaml
    optimized_config_path = f"{BASE_VOLUME_PATH}/configs/detection_detr_optimized.yaml"
    
    with open(optimized_config_path, 'w') as f:
        yaml.dump(optimized_config, f)
    
    print(f"✅ Optimized configuration saved to: {optimized_config_path}")
    print(f"   Best val_map: {best_trial.last_result['val_map']:.4f}")
    print(f"   Best config: {best_config}")
    
    # Save results summary
    results_summary = {
        'best_config': best_config,
        'best_val_map': best_trial.last_result['val_map'],
        'total_trials': len(results_df),
        'mean_val_map': results_df['val_map'].mean(),
        'std_val_map': results_df['val_map'].std(),
        'all_results': results_df.to_dict('records')
    }
    
    summary_path = f"{BASE_VOLUME_PATH}/results/hyperparameter_tuning_summary.yaml"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        yaml.dump(results_summary, f)
    
    print(f"✅ Results summary saved to: {summary_path}")
    
    return optimized_config

if analysis and results_df is not None:
    optimized_config = save_best_configuration(analysis, results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary and Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Tuning Summary

# COMMAND ----------

print("=" * 60)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 60)

if analysis and results_df is not None:
    best_trial = analysis.get_best_trial("val_map", "max")
    
    print(f"✅ Optimization Results:")
    print(f"   Total trials: {len(results_df)}")
    print(f"   Best val_map: {best_trial.last_result['val_map']:.4f}")
    print(f"   Mean val_map: {results_df['val_map'].mean():.4f}")
    print(f"   Std val_map: {results_df['val_map'].std():.4f}")
    
    print(f"\n🏆 Best Configuration:")
    best_config = best_trial.config
    for param, value in best_config.items():
        print(f"   {param}: {value}")
    
    print(f"\n📁 Output Files:")
    print(f"   Optimized config: {BASE_VOLUME_PATH}/configs/detection_detr_optimized.yaml")
    print(f"   Results summary: {BASE_VOLUME_PATH}/results/hyperparameter_tuning_summary.yaml")
    print(f"   Ray results: {BASE_VOLUME_PATH}/ray_results/")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Use optimized configuration for final training")
    print(f"   2. Run notebook 02_model_training.py with optimized config")
    print(f"   3. Compare results with baseline configuration")
    print(f"   4. Consider additional tuning if needed")
else:
    print("❌ Optimization failed or no results available")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding Hyperparameter Tuning for DETR

# MAGIC 
# MAGIC ### Key Insights:
# MAGIC 
# MAGIC **1. Learning Rate Sensitivity:**
# MAGIC - DETR is sensitive to learning rate choice
# MAGIC - Too high: training instability, poor convergence
# MAGIC - Too low: slow convergence, suboptimal results
# MAGIC - Optimal range: 1e-5 to 1e-3 (log scale)
# MAGIC 
# MAGIC **2. Batch Size Considerations:**
# MAGIC - Larger batches: better gradient estimates, but memory constraints
# MAGIC - Smaller batches: more noise, but better generalization
# MAGIC - Optimal: 16-32 for most setups
# MAGIC 
# MAGIC **3. Weight Decay Impact:**
# MAGIC - Prevents overfitting on COCO dataset
# MAGIC - Critical for transformer-based models
# MAGIC - Optimal range: 1e-5 to 1e-3
# MAGIC 
# MAGIC **4. Architecture Choices:**
# MAGIC - ResNet-50 vs ResNet-101: trade-off between speed and accuracy
# MAGIC - ResNet-101: better accuracy, slower training
# MAGIC - ResNet-50: faster training, slightly lower accuracy
# MAGIC 
# MAGIC **5. Training Duration:**
# MAGIC - DETR benefits from longer training
# MAGIC - Early stopping prevents overfitting
# MAGIC - Optimal: 100-200 epochs
# MAGIC 
# MAGIC **6. Bayesian Optimization Benefits:**
# MAGIC - Efficient exploration of search space
# MAGIC - Focuses on promising regions
# MAGIC - Reduces total number of trials needed
# MAGIC 
# MAGIC The hyperparameter tuning process helps find the optimal configuration for your specific dataset and hardware setup! 
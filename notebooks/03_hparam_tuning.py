# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # 03. Hyperparameter Tuning for DETR
# MAGIC
# MAGIC This notebook performs hyperparameter tuning for DETR training using Optuna. We explore different learning rates, batch sizes, and other hyperparameters to find the optimal configuration for our COCO dataset.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC **Hyperparameter Tuning Strategy:**
# MAGIC 1. **Search Space Definition**: Define ranges for key hyperparameters
# MAGIC 2. **Search Algorithm**: Use TPE (Tree-structured Parzen Estimator) for efficient Bayesian search
# MAGIC 3. **Pruning**: MedianPruner stops unpromising trials early
# MAGIC 4. **Sequential Execution**: Trials run sequentially on a single GPU
# MAGIC 5. **Result Analysis**: Compare and select best configuration
# MAGIC
# MAGIC ### Key Hyperparameters for DETR:
# MAGIC - **Learning Rate**: Critical for convergence (1e-5 to 1e-3)
# MAGIC - **Batch Size**: Affects memory usage and training stability
# MAGIC - **Weight Decay**: Regularization to prevent overfitting
# MAGIC - **Scheduler**: Learning rate scheduling strategy
# MAGIC - **Max Epochs**: Training duration
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC
# MAGIC 1. **Search Space Setup**: Define hyperparameter ranges via Optuna trial API
# MAGIC 2. **Trial Configuration**: Set up individual training trials
# MAGIC 3. **Optimization**: Run TPE-based Bayesian optimization
# MAGIC 4. **Result Analysis**: Analyze and visualize results with Optuna visualizations
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
import copy
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis
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
from training.trainer import Trainer
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

print(f"Volume directories created:")
print(f"   Tuning Results: {TUNING_RESULTS_DIR}")
print(f"   Tuning Logs: {TUNING_LOGS_DIR}")

# Load base configuration
if os.path.exists(CONFIG_PATH):
    base_config = load_config(CONFIG_PATH)
    base_config['training']['checkpoint_dir'] = f"{BASE_VOLUME_PATH}/checkpoints"
    print(f"Fixed config: updated checkpoint directory")
else:
    print("Config file not found. Using default detection config.")
    base_config = get_default_config("detection")
    base_config['training']['checkpoint_dir'] = f"{BASE_VOLUME_PATH}/checkpoints"

print("Base configuration loaded successfully!")
print(f"Checkpoint directory: {base_config['training']['checkpoint_dir']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure MLflow Experiment

# COMMAND ----------

experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/detection_tuning"
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define the Objective Function

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optuna Objective
# MAGIC
# MAGIC Each trial samples hyperparameters, trains the model for a reduced number of epochs,
# MAGIC and returns the monitor metric. Trials are logged to MLflow for tracking.

# COMMAND ----------

MONITOR_METRIC = base_config.get('training', {}).get('monitor_metric', 'val_loss')
MONITOR_MODE = base_config.get('training', {}).get('monitor_mode', 'min')

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for a single hyperparameter trial."""

    # ---- Sample hyperparameters ----
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    scheduler = trial.suggest_categorical("scheduler", ["cosine", "step", "exponential"])
    max_epochs = trial.suggest_categorical("max_epochs", [50, 100, 150])

    try:
        # ---- Build trial config ----
        trial_config = copy.deepcopy(base_config)

        trial_config['model']['learning_rate'] = learning_rate
        trial_config['model']['weight_decay'] = weight_decay
        trial_config['data']['batch_size'] = batch_size
        trial_config['training']['max_epochs'] = max_epochs
        trial_config['training']['scheduler'] = scheduler

        # ---- Create model and data module ----
        model = DetectionModel(trial_config['model'])
        data_module = DetectionDataModule(trial_config['data'])

        # ---- Create Trainer ----
        trainer_config = {
            'task': trial_config['model']['task_type'],
            'model_name': trial_config['model']['model_name'],
            'max_epochs': max_epochs,
            'log_every_n_steps': trial_config['training'].get('log_every_n_steps', 50),
            'monitor_metric': MONITOR_METRIC,
            'monitor_mode': MONITOR_MODE,
            'early_stopping_patience': trial_config['training'].get('early_stopping_patience', 10),
            'checkpoint_dir': f"{BASE_VOLUME_PATH}/checkpoints/trial_{trial.number}",
            'save_top_k': 1,
            'distributed': False,
        }

        trainer = Trainer(
            config=trainer_config,
            model=model,
            data_module=data_module,
        )

        # ---- Log to MLflow ----
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params({
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "scheduler": scheduler,
                "max_epochs": max_epochs,
            })

            result = trainer.train()

            # Extract the monitor metric
            metric_value = result.metrics.get(MONITOR_METRIC, None)
            if metric_value is None:
                raise ValueError(f"Metric '{MONITOR_METRIC}' not found in training results")

            mlflow.log_metrics({
                MONITOR_METRIC: metric_value,
                "val_loss": result.metrics.get("val_loss", float("inf")),
                "train_loss": result.metrics.get("train_loss", float("inf")),
            })

        print(f"Trial {trial.number} completed -- {MONITOR_METRIC}: {metric_value:.4f}")
        return metric_value

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        # Return a value that the study will consider poor
        if MONITOR_MODE == "min":
            return float("inf")
        else:
            return 0.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Optuna Study and Run Optimization

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Study
# MAGIC
# MAGIC We use the TPE sampler for Bayesian optimization and the MedianPruner
# MAGIC to stop unpromising trials early.

# COMMAND ----------

N_TRIALS = 20

study = optuna.create_study(
    study_name="detr_hyperparameter_tuning",
    direction="minimize" if MONITOR_MODE == "min" else "maximize",
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
    ),
)

print(f"Optuna study created: {study.study_name}")
print(f"   Direction : {study.direction.name}")
print(f"   Sampler   : TPESampler")
print(f"   Pruner    : MedianPruner")
print(f"   Trials    : {N_TRIALS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Optimization

# COMMAND ----------

print("Starting hyperparameter optimization ...")

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Hyperparameter optimization completed!")
print(f"   Results saved to: {TUNING_RESULTS_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analyze Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Best Trial Summary

# COMMAND ----------

best = study.best_trial

print("=" * 60)
print("BEST TRIAL")
print("=" * 60)
print(f"   Trial number : {best.number}")
print(f"   {MONITOR_METRIC}: {best.value:.4f}")
print(f"   Parameters:")
for k, v in best.params.items():
    print(f"      {k}: {v}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trials DataFrame

# COMMAND ----------

trials_df = study.trials_dataframe()
trials_df = trials_df.sort_values("value", ascending=(MONITOR_MODE == "min"))
print(f"Total trials  : {len(trials_df)}")
print(f"Completed     : {len(trials_df[trials_df['state'] == 'COMPLETE'])}")
print(f"Pruned        : {len(trials_df[trials_df['state'] == 'PRUNED'])}")
print(f"Failed        : {len(trials_df[trials_df['state'] == 'FAIL'])}")
display(trials_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optuna Visualizations

# COMMAND ----------

# MAGIC %md
# MAGIC #### Optimization History

# COMMAND ----------

try:
    fig = vis.plot_optimization_history(study)
    fig.show()
except Exception as e:
    print(f"Could not render optimization history: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Parameter Importances

# COMMAND ----------

try:
    fig = vis.plot_param_importances(study)
    fig.show()
except Exception as e:
    print(f"Could not render parameter importances: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Parallel Coordinate Plot

# COMMAND ----------

try:
    fig = vis.plot_parallel_coordinate(study)
    fig.show()
except Exception as e:
    print(f"Could not render parallel coordinate plot: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Slice Plot

# COMMAND ----------

try:
    fig = vis.plot_slice(study)
    fig.show()
except Exception as e:
    print(f"Could not render slice plot: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Best Configuration

# COMMAND ----------

def save_best_configuration(study):
    """Save the best configuration found during optimization."""

    best_trial = study.best_trial

    # Build optimized config from base + best params
    optimized_config = copy.deepcopy(base_config)
    optimized_config['model']['learning_rate'] = best_trial.params['learning_rate']
    optimized_config['model']['weight_decay'] = best_trial.params['weight_decay']
    optimized_config['data']['batch_size'] = best_trial.params['batch_size']
    optimized_config['training']['max_epochs'] = best_trial.params['max_epochs']
    optimized_config['training']['scheduler'] = best_trial.params['scheduler']

    # Add optimization metadata
    optimized_config['optimization'] = {
        'best_metric_name': MONITOR_METRIC,
        'best_metric_value': float(best_trial.value),
        'trial_number': best_trial.number,
        'total_trials': len(study.trials),
        'optimization_date': str(pd.Timestamp.now()),
    }

    # Save optimized config as YAML
    config_path = f"{TUNING_RESULTS_DIR}/optimized_detr_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(optimized_config, f, default_flow_style=False)
    print(f"Optimized configuration saved to: {config_path}")

    # Save results summary as JSON
    summary_path = f"{TUNING_RESULTS_DIR}/optimization_summary.json"
    summary = {
        'best_params': best_trial.params,
        'best_metric': {MONITOR_METRIC: float(best_trial.value)},
        'total_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'optimization_date': str(pd.Timestamp.now()),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Optimization summary saved to: {summary_path}")

    return True

config_saved = save_best_configuration(study)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary and Next Steps

# COMMAND ----------

print("=" * 60)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 60)

best_trial = study.best_trial

print(f"Optimization completed successfully!")
print(f"   Total trials       : {len(study.trials)}")
print(f"   Best trial         : {best_trial.number}")
print(f"   Best {MONITOR_METRIC}: {best_trial.value:.4f}")
for k, v in best_trial.params.items():
    print(f"   Best {k}: {v}")

print(f"\nResults saved to:")
print(f"   Tuning results  : {TUNING_RESULTS_DIR}")
print(f"   Optimized config: {TUNING_RESULTS_DIR}/optimized_detr_config.yaml")
print(f"   Summary         : {TUNING_RESULTS_DIR}/optimization_summary.json")

print(f"\nNext steps:")
print(f"   1. Use the optimized configuration for final training")
print(f"   2. Run the training notebook with the new config")
print(f"   3. Compare results with the baseline model")

print("=" * 60)

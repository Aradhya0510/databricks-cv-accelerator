# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperparameter Tuning
# MAGIC 
# MAGIC This notebook handles hyperparameter optimization for computer vision models.
# MAGIC 
# MAGIC ## Hyperparameter Tuning Guide
# MAGIC 
# MAGIC ### 1. Search Space Configuration
# MAGIC 
# MAGIC Configure the hyperparameter search space in the YAML config:
# MAGIC 
# MAGIC ```yaml
# MAGIC tuning:
# MAGIC   # Base search space
# MAGIC   base_space:
# MAGIC     learning_rate:
# MAGIC       type: "loguniform"
# MAGIC       min: 1e-5
# MAGIC       max: 1e-3
# MAGIC     weight_decay:
# MAGIC       type: "loguniform"
# MAGIC       min: 1e-5
# MAGIC       max: 1e-2
# MAGIC     batch_size:
# MAGIC       type: "choice"
# MAGIC       values: [16, 32, 64]
# MAGIC     scheduler:
# MAGIC       type: "choice"
# MAGIC       values: ["cosine", "linear", "step"]
# MAGIC   
# MAGIC   # Task-specific search spaces
# MAGIC   detection_space:
# MAGIC     confidence_threshold:
# MAGIC       type: "uniform"
# MAGIC       min: 0.3
# MAGIC       max: 0.7
# MAGIC     iou_threshold:
# MAGIC       type: "uniform"
# MAGIC       min: 0.3
# MAGIC       max: 0.7
# MAGIC     max_detections:
# MAGIC       type: "choice"
# MAGIC       values: [50, 100, 200]
# MAGIC   
# MAGIC   classification_space:
# MAGIC     dropout:
# MAGIC       type: "uniform"
# MAGIC       min: 0.1
# MAGIC       max: 0.5
# MAGIC     mixup_alpha:
# MAGIC       type: "uniform"
# MAGIC       min: 0.1
# MAGIC       max: 0.5
# MAGIC   
# MAGIC   segmentation_space:
# MAGIC     aux_loss_weight:
# MAGIC       type: "uniform"
# MAGIC       min: 0.1
# MAGIC       max: 0.5
# MAGIC     mask_threshold:
# MAGIC       type: "uniform"
# MAGIC       min: 0.3
# MAGIC       max: 0.7
# MAGIC ```
# MAGIC 
# MAGIC ### 2. Tuning Configuration
# MAGIC 
# MAGIC Configure the tuning process in the YAML config:
# MAGIC 
# MAGIC ```yaml
# MAGIC tuning:
# MAGIC   # Tuning settings
# MAGIC   num_trials: 20               # Number of trials
# MAGIC   metric: "val_loss"          # Metric to optimize
# MAGIC   mode: "min"                 # Optimization mode
# MAGIC   grace_period: 10            # Grace period for early stopping
# MAGIC   reduction_factor: 2         # Reduction factor for ASHA
# MAGIC   
# MAGIC   # Resource settings
# MAGIC   resources:
# MAGIC     cpu: 4                    # CPUs per trial
# MAGIC     gpu: 1                    # GPUs per trial
# MAGIC   
# MAGIC   # Logging settings
# MAGIC   log_dir: "/Volumes/main/cv_ref/logs/tuning"  # Log directory
# MAGIC   results_dir: "/Volumes/main/cv_ref/results/tuning"  # Results directory
# MAGIC ```
# MAGIC 
# MAGIC ### 3. Available Schedulers
# MAGIC 
# MAGIC The project supports various hyperparameter tuning schedulers:
# MAGIC 
# MAGIC - `ASHAScheduler`: Asynchronous Successive Halving Algorithm
# MAGIC - `HyperBandScheduler`: HyperBand algorithm
# MAGIC - `MedianStoppingRule`: Median stopping rule
# MAGIC - `PopulationBasedTraining`: Population-based training
# MAGIC 
# MAGIC Configure the scheduler in the YAML config:
# MAGIC 
# MAGIC ```yaml
# MAGIC tuning:
# MAGIC   scheduler:
# MAGIC     type: "asha"              # or "hyperband", "median", "pbt"
# MAGIC     metric: "val_loss"
# MAGIC     mode: "min"
# MAGIC     grace_period: 10
# MAGIC     reduction_factor: 2
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import mlflow
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.mlflow import MLflowLoggerCallback

# Add the project root to Python path
project_root = "/Workspace/Repos/Databricks_CV_ref"
sys.path.append(project_root)

from src.utils.logging import setup_logger, get_metric_logger
from src.training.trainer import UnifiedTrainer
from src.tasks.detection.model import DetectionModel
from src.tasks.classification.model import ClassificationModel
from src.tasks.semantic_segmentation.model import SemanticSegmentationModel

# COMMAND ----------

# DBTITLE 1,Initialize Logging
logger = setup_logger(
    name="hparam_tuning",
    log_file="/Volumes/main/cv_ref/logs/tuning.log"
)

# COMMAND ----------

# DBTITLE 1,Load Configuration
def load_task_config(task: str):
    """Load configuration for the specified task."""
    config_path = f"/Volumes/main/cv_ref/configs/{task}_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# COMMAND ----------

# DBTITLE 1,Define Search Space
def get_search_space(task: str, config: dict):
    """Define hyperparameter search space for the task."""
    tuning_config = config['tuning']
    
    # Get base space
    base_space = {}
    for param, settings in tuning_config['base_space'].items():
        if settings['type'] == 'loguniform':
            base_space[param] = tune.loguniform(
                settings['min'],
                settings['max']
            )
        elif settings['type'] == 'uniform':
            base_space[param] = tune.uniform(
                settings['min'],
                settings['max']
            )
        elif settings['type'] == 'choice':
            base_space[param] = tune.choice(settings['values'])
    
    # Get task-specific space
    task_space = {}
    task_space_key = f"{task}_space"
    if task_space_key in tuning_config:
        for param, settings in tuning_config[task_space_key].items():
            if settings['type'] == 'loguniform':
                task_space[param] = tune.loguniform(
                    settings['min'],
                    settings['max']
                )
            elif settings['type'] == 'uniform':
                task_space[param] = tune.uniform(
                    settings['min'],
                    settings['max']
                )
            elif settings['type'] == 'choice':
                task_space[param] = tune.choice(settings['values'])
    
    # Combine spaces
    search_space = {**base_space, **task_space}
    return search_space

# COMMAND ----------

# DBTITLE 1,Training Function for Tuning
def train_tune(config, task: str, data_loaders: dict):
    """Training function for hyperparameter tuning."""
    # Initialize model with trial config
    model_class = get_model_class(task)
    model = model_class(**config)
    
    # Setup trainer
    trainer = UnifiedTrainer(
        task=task,
        model_class=model_class,
        config=config
    )
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloader=data_loaders['train_loader'],
        val_dataloader=data_loaders['val_loader']
    )
    
    # Get validation metrics
    val_metrics = trainer.get_validation_metrics()
    
    # Report metrics to Ray Tune
    tune.report(**val_metrics)

# COMMAND ----------

# DBTITLE 1,Setup Ray Tune
def setup_tune(task: str, config: dict, data_loaders: dict):
    """Setup Ray Tune for hyperparameter optimization."""
    tuning_config = config['tuning']
    
    # Initialize Ray
    ray.init()
    
    # Get search space
    search_space = get_search_space(task, config)
    
    # Setup scheduler
    scheduler_config = tuning_config['scheduler']
    if scheduler_config['type'] == 'asha':
        scheduler = ASHAScheduler(
            metric=scheduler_config['metric'],
            mode=scheduler_config['mode'],
            max_t=100,
            grace_period=scheduler_config['grace_period'],
            reduction_factor=scheduler_config['reduction_factor']
        )
    # Add other scheduler types here
    
    # Setup MLflow callback
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        registry_uri=mlflow.get_registry_uri(),
        experiment_name=f"{task}_tuning"
    )
    
    # Run hyperparameter tuning
    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            task=task,
            data_loaders=data_loaders
        ),
        config=search_space,
        num_samples=tuning_config['num_trials'],
        scheduler=scheduler,
        callbacks=[mlflow_callback],
        resources_per_trial=tuning_config['resources']
    )
    
    return analysis

# COMMAND ----------

# DBTITLE 1,Main Tuning Function
def tune_hyperparameters(
    task: str,
    config: dict,
    data_loaders: dict
):
    """Main function for hyperparameter tuning."""
    tuning_config = config['tuning']
    
    # Setup MLflow
    mlflow_logger = get_metric_logger(f"{task}_tuning")
    
    # Run tuning
    analysis = setup_tune(task, config, data_loaders)
    
    # Get best trial
    best_trial = analysis.get_best_trial(
        metric=tuning_config['metric'],
        mode=tuning_config['mode']
    )
    
    # Log best parameters
    logger.info(f"Best trial parameters: {best_trial.config}")
    mlflow_logger.log_params(best_trial.config)
    
    # Save tuning results
    results = {
        'best_trial': best_trial.config,
        'best_metric': best_trial.last_result[tuning_config['metric']],
        'all_trials': analysis.results
    }
    
    results_path = f"/Volumes/main/cv_ref/results/{task}_tuning_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    # Update config with best parameters
    config.update(best_trial.config)
    
    return config, analysis

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Tune detection model
task = "detection"
config = load_task_config(task)

# Load data (from previous notebook)
data_loaders = {
    'train_loader': train_loader,
    'val_loader': val_loader,
    'test_loader': test_loader
}

# Run hyperparameter tuning
best_config, analysis = tune_hyperparameters(
    task=task,
    config=config,
    data_loaders=data_loaders
)

# Display results
print("Best trial parameters:")
print(yaml.dump(best_config, default_flow_style=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review tuning results
# MAGIC 2. Train model with best parameters
# MAGIC 3. Proceed to model evaluation notebook 
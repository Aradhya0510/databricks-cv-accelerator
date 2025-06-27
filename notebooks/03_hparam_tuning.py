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
# MAGIC   semantic_segmentation_space:
# MAGIC     aux_loss_weight:
# MAGIC       type: "uniform"
# MAGIC       min: 0.1
# MAGIC       max: 0.5
# MAGIC     mask_threshold:
# MAGIC       type: "uniform"
# MAGIC       min: 0.3
# MAGIC       max: 0.7
# MAGIC   
# MAGIC   instance_segmentation_space:
# MAGIC     rpn_batch_size_per_image:
# MAGIC       type: "choice"
# MAGIC       values: [256, 512]
# MAGIC     box_loss_weight:
# MAGIC       type: "uniform"
# MAGIC       min: 0.5
# MAGIC       max: 2.0
# MAGIC   
# MAGIC   panoptic_segmentation_space:
# MAGIC     mask_threshold:
# MAGIC       type: "uniform"
# MAGIC       min: 0.3
# MAGIC       max: 0.7
# MAGIC     overlap_threshold:
# MAGIC       type: "uniform"
# MAGIC       min: 0.5
# MAGIC       max: 0.9
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

%pip install -r "../requirements.txt"
dbutils.library.restartPython()

# COMMAND ----------

%run ./01_data_preparation
%run ./02_model_training

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
import yaml

# Add the project root to Python path
project_root = "/Volumes/<catalog>/<schema>/<volume>/<path>/<file_name>"
sys.path.append(project_root)

from src.utils.logging import setup_logger, get_mlflow_logger
from src.training.trainer import UnifiedTrainer
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
    name="hparam_tuning",
    log_file=f"{log_dir}/tuning.log"
)

# COMMAND ----------

# DBTITLE 1,Load Configuration
def load_task_config(task: str):
    """Load task-specific configuration."""
    config_path = f"{volume_path}/configs/{task}_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# COMMAND ----------

# DBTITLE 1,Get Model and Data Module Classes
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

def get_data_module_class(task: str):
    """Get the appropriate data module class based on task."""
    data_module_classes = {
        'detection': DetectionDataModule,
        'classification': ClassificationDataModule,
        'semantic_segmentation': SemanticSegmentationDataModule,
        'instance_segmentation': InstanceSegmentationDataModule,
        'panoptic_segmentation': PanopticSegmentationDataModule
    }
    
    if task not in data_module_classes:
        raise ValueError(f"Unsupported task: {task}")
    
    return data_module_classes[task]

# COMMAND ----------

# DBTITLE 1,Define Search Space
def get_search_space(task: str, config: dict):
    """Define hyperparameter search space for the task."""
    tuning_config = config.get('tuning', {})
    
    # Get base space
    base_space = {}
    base_space_config = tuning_config.get('base_space', {})
    for param, settings in base_space_config.items():
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
    task_space_config = tuning_config.get(task_space_key, {})
    for param, settings in task_space_config.items():
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
    
    # Combine base and task-specific spaces
    search_space = {**base_space, **task_space}
    
    return search_space

# COMMAND ----------

# DBTITLE 1,Training Function for Tune
def train_tune(config, task: str, data_loaders: dict):
    """Training function for Ray Tune."""
    # Initialize model
    model_class = get_model_class(task)
    model = model_class(config=config)
    
    # Initialize data module
    data_module_class = get_data_module_class(task)
    data_module = data_module_class(config=config)
    
    # Set data loaders
    data_module.train_dataloader = lambda: data_loaders['train']
    data_module.val_dataloader = lambda: data_loaders['val']
    if 'test' in data_loaders:
        data_module.test_dataloader = lambda: data_loaders['test']
    
    # Setup trainer
    trainer = UnifiedTrainer(
        task=task,
        model=model,
        config=config,
        data_module_class=data_module_class
    )
    
    # Train model
    results = trainer.train()
    
    # Report metrics to Tune
    tune.report(
        val_loss=results['val_loss'],
        val_map=results.get('val_map', 0.0),
        val_accuracy=results.get('val_accuracy', 0.0),
        val_iou=results.get('val_iou', 0.0)
    )

# COMMAND ----------

# DBTITLE 1,Setup Tune
def setup_tune(task: str, config: dict, data_loaders: dict):
    """Setup Ray Tune for hyperparameter optimization."""
    # Get search space
    search_space = get_search_space(task, config)
    
    # Get tuning configuration
    tuning_config = config.get('tuning', {})
    
    # Setup scheduler
    scheduler_type = tuning_config.get('scheduler', {}).get('type', 'asha')
    if scheduler_type == 'asha':
        scheduler = ASHAScheduler(
            metric=tuning_config.get('metric', 'val_loss'),
            mode=tuning_config.get('mode', 'min'),
            grace_period=tuning_config.get('grace_period', 10),
            reduction_factor=tuning_config.get('reduction_factor', 2)
        )
    else:
        scheduler = ASHAScheduler(
            metric=tuning_config.get('metric', 'val_loss'),
            mode=tuning_config.get('mode', 'min'),
            grace_period=tuning_config.get('grace_period', 10),
            reduction_factor=tuning_config.get('reduction_factor', 2)
        )
    
    # Setup MLflow logger
    mlflow_logger = MLflowLoggerCallback(
        experiment_name=f"{task}_hparam_tuning",
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
    )
    
    return search_space, scheduler, mlflow_logger

# COMMAND ----------

# DBTITLE 1,Main Tuning Function
def tune_hyperparameters(
    task: str,
    config: dict,
    data_loaders: dict
):
    """Main function to run hyperparameter tuning."""
    # Setup tune
    search_space, scheduler, mlflow_logger = setup_tune(task, config, data_loaders)
    
    # Get tuning configuration
    tuning_config = config.get('tuning', {})
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Run tuning
    analysis = tune.run(
        lambda tune_config: train_tune(tune_config, task, data_loaders),
        config=search_space,
        num_samples=tuning_config.get('num_trials', 20),
        scheduler=scheduler,
        callbacks=[mlflow_logger],
        resources_per_trial=tuning_config.get('resources', {'cpu': 4, 'gpu': 1}),
        local_dir=tuning_config.get('log_dir', f"{volume_path}/logs/tuning"),
        name=f"{task}_hparam_tuning"
    )
    
    # Get best trial
    best_trial = analysis.get_best_trial(
        metric=tuning_config.get('metric', 'val_loss'),
        mode=tuning_config.get('mode', 'min')
    )
    
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    
    # Save best configuration
    best_config_path = f"{volume_path}/configs/{task}_best_config.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_trial.config, f)
    
    return analysis, best_trial

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Tune hyperparameters for detection task
task = "detection"

# Load configuration
config = load_task_config(task)

# Prepare data (using functions from previous notebooks)
train_loader, val_loader = prepare_data(task)

# Run hyperparameter tuning
analysis, best_trial = tune_hyperparameters(
    task=task,
    config=config,
    data_loaders={'train': train_loader, 'val': val_loader}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review the tuning results
# MAGIC 2. Analyze the best hyperparameters
# MAGIC 3. Use the best configuration for final training
# MAGIC 4. Proceed to model evaluation notebook 
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.mlflow import MLflowLoggerCallback
from pathlib import Path
import sys
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class UnifiedTrainerConfig:
    """Configuration for the unified trainer."""
    # Task and model info
    task: str
    model_name: str
    
    # Training parameters
    max_epochs: int
    log_every_n_steps: int
    monitor_metric: str
    monitor_mode: str
    early_stopping_patience: int
    checkpoint_dir: str
    save_top_k: int
    
    # Ray distributed training settings
    distributed: bool
    num_workers: int
    use_gpu: bool
    resources_per_worker: Dict[str, int]
    
    # MLflow settings
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set default values."""
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set default resources if not provided
        if self.resources_per_worker is None:
            self.resources_per_worker = {
                "CPU": 1,
                "GPU": 1 if self.use_gpu else 0
            }

class UnifiedTrainer:
    """Unified trainer for all computer vision tasks using Ray on Databricks."""
    
    def __init__(
        self,
        config: Union[Dict[str, Any], UnifiedTrainerConfig],
        model: Optional[pl.LightningModule] = None,
        data_module: Optional[pl.LightningDataModule] = None
    ):
        """Initialize the trainer.
        
        Args:
            config: Either a UnifiedTrainerConfig object or a dictionary containing the full configuration
            model: Optional pre-initialized model
            data_module: Optional pre-initialized data module
        """
        # Initialize config
        if isinstance(config, dict):
            # Extract all parameters we need from the full config
            trainer_config = {
                'task': config['model']['task_type'],
                'model_name': config['model']['model_name'],
                'max_epochs': config['model']['epochs'],
                'log_every_n_steps': config['training']['log_every_n_steps'],
                'monitor_metric': config['training']['monitor'],
                'monitor_mode': config['training']['mode'],
                'early_stopping_patience': config['training']['early_stopping_patience'],
                'checkpoint_dir': config['training']['checkpoint_dir'],
                'save_top_k': config['training']['save_top_k'],
                'distributed': config['training']['distributed'],
                'num_workers': config['training']['num_workers'],
                'use_gpu': config['training']['use_gpu'],
                'resources_per_worker': config['training']['resources_per_worker']
            }
            config = UnifiedTrainerConfig(**trainer_config)
        self.config = config
        
        self.model = model
        self.data_module = data_module
        self.trainer = None
    
    def _init_callbacks(self):
        """Initialize training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename=f"{self.config.task}_{self.config.model_name}_best",
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            save_top_k=self.config.save_top_k
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            patience=self.config.early_stopping_patience
        )
        callbacks.append(early_stopping)
        
        # Ray train report callback (only for distributed training)
        if self.config.distributed:
            ray_report_callback = RayTrainReportCallback()
            callbacks.append(ray_report_callback)
        
        return callbacks
    
    def _init_trainer(self):
        """Initialize the PyTorch Lightning trainer."""
        callbacks = self._init_callbacks()
        
        # Configure trainer based on training mode
        if self.config.distributed:
            # Distributed training with Ray
            self.trainer = pl.Trainer(
                max_epochs=self.config.max_epochs,
                accelerator="auto",
                devices="auto",
                strategy=RayDDPStrategy(),
                plugins=[RayLightningEnvironment()],
                callbacks=callbacks,
                log_every_n_steps=self.config.log_every_n_steps
            )
            # Validate trainer configuration
            self.trainer = prepare_trainer(self.trainer)
        else:
            # Local training with PyTorch Lightning
            self.trainer = pl.Trainer(
                max_epochs=self.config.max_epochs,
                accelerator="gpu" if self.config.use_gpu else "cpu", # Set accelerator based on config
                devices="auto",  # Let PL select available GPUs based on accelerator
                # PyTorch Lightning, with accelerator="gpu" and devices="auto",
                # will automatically switch to DDP if multiple GPUs are detected and available
                # and if a suitable strategy (like "ddp_notebook" or "ddp") is provided.
                # Remove the explicit torch.cuda.device_count() check here to prevent premature CUDA init.
                strategy="ddp_notebook" if self.config.use_gpu else None, # Use ddp_notebook if use_gpu is true, let PL handle device counting
                callbacks=callbacks,
                log_every_n_steps=self.config.log_every_n_steps,
            )
    
    def train(self):
        """Train the model using either local or distributed training."""
        if self.model is None or self.data_module is None:
            raise ValueError("Model and data module must be provided before training")
        
        self._init_trainer()
        
        if self.config.distributed:
            # Set up Ray cluster for distributed training
            try:
                from ray.util.spark import setup_ray_cluster
                setup_ray_cluster(
                    num_worker_nodes=self.config.num_workers,
                    num_cpus_per_node=self.config.resources_per_worker['CPU'],
                    num_gpus_per_node=self.config.resources_per_worker['GPU'] if self.config.use_gpu else 0
                )
            except ImportError:
                raise ImportError("Ray on Spark is not installed. Please install it using: pip install ray[spark]")
            
            # Configure Ray training
            scaling_config = ScalingConfig(
                num_workers=self.config.num_workers,
                use_gpu=self.config.use_gpu,
                resources_per_worker=self.config.resources_per_worker
            )
            
            run_config = RunConfig(
                storage_path=self.config.checkpoint_dir,
                name=f"{self.config.task}_{self.config.model_name}",
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute=self.config.monitor_metric,
                    checkpoint_score_order=self.config.monitor_mode
                )
            )
            
            # Create Ray trainer
            ray_trainer = TorchTrainer(
                lambda: self.trainer.fit(self.model, datamodule=self.data_module),
                scaling_config=scaling_config,
                run_config=run_config
            )
            
            # Start training
            result = ray_trainer.fit()
        else:
            # Local training
            self.trainer.fit(self.model, datamodule=self.data_module)
            result = type('Result', (), {'metrics': self.trainer.callback_metrics})
        
        # Log results to MLflow
        with mlflow.start_run(
            run_name=self.config.run_name or f"{self.config.task}_training",
            experiment_name=self.config.experiment_name
        ) as run:
            # Log metrics
            mlflow.log_metrics(result.metrics)
            
            # Log model
            mlflow.pytorch.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=f"{self.config.task}_{self.config.model_name}"
            )
            
            # Log checkpoint
            if self.config.distributed:
                checkpoint_path = result.checkpoint.path
            else:
                checkpoint_path = self.trainer.checkpoint_callback.best_model_path
            mlflow.log_artifact(checkpoint_path)
        
        return result
    
    def tune(self, search_space: dict, num_trials: int = 20):
        """Run hyperparameter tuning using Ray Tune.
        
        Args:
            search_space: Dictionary defining the hyperparameter search space
            num_trials: Number of trials to run
        """
        if not self.config.distributed:
            raise ValueError("Hyperparameter tuning requires distributed training mode")
        
        # Initialize Ray
        try:
            from ray.util.spark import setup_ray_cluster
            setup_ray_cluster(
                num_worker_nodes=self.config.num_workers,
                num_cpus_per_node=self.config.resources_per_worker['CPU'],
                num_gpus_per_node=self.config.resources_per_worker['GPU'] if self.config.use_gpu else 0
            )
        except ImportError:
            raise ImportError("Ray on Spark is not installed. Please install it using: pip install ray[spark]")
        
        # Configure scheduler
        scheduler = ASHAScheduler(
            metric=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            max_t=self.config.max_epochs,
            grace_period=10,
            reduction_factor=2
        )
        
        # Define training function for tuning
        def train_func(config):
            # Update model config with trial parameters
            trial_config = self.config.copy()
            for key, value in config.items():
                if key in trial_config:
                    trial_config[key].update(value)
                else:
                    trial_config[key] = value
            
            # Train model
            result = self.train()
            
            # Report metrics to Ray Tune
            train.report({
                'val_loss': result.metrics['val_loss'],
                'val_map': result.metrics['val_map'] if self.config.task == 'detection' else None,
                'val_iou': result.metrics['val_iou'] if self.config.task == 'segmentation' else None
            })
        
        # Configure MLflow logger
        mlflow_logger = MLflowLoggerCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            registry_uri=mlflow.get_registry_uri(),
            experiment_name=mlflow.active_run().info.experiment_id
        )
        
        # Run hyperparameter tuning
        from ray import tune
        analysis = tune.run(
            train_func,
            config=search_space,
            num_samples=num_trials,
            scheduler=scheduler,
            resources_per_trial={
                'cpu': self.config.resources_per_worker['CPU'],
                'gpu': self.config.resources_per_worker['GPU'] if self.config.use_gpu else 0
            },
            callbacks=[mlflow_logger],
            verbose=1
        )
        
        # Get best trial
        best_trial = analysis.get_best_trial(
            metric=self.config.monitor_metric,
            mode=self.config.monitor_mode
        )
        
        return best_trial.config
    
    def get_metrics(self):
        """Get training metrics from MLflow."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(mlflow.active_run().info.run_id)
        return run.data.metrics 
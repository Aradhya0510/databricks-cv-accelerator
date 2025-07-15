import os
import yaml
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
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
from ray.air.integrations.mlflow import MLflowLoggerCallback
from pathlib import Path
import sys
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field

# Import the simplified logging utilities
from utils.logging import VolumeCheckpoint, create_databricks_logger

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
    
    # Checkpoint settings
    checkpoint_dir: str
    # NEW: Optional path to a persistent volume for final checkpoints
    volume_checkpoint_dir: Optional[str] = None
    save_top_k: int = 3
    
    # Ray distributed training settings
    distributed: bool = False
    num_workers: int = 1
    use_gpu: bool = True
    resources_per_worker: Dict[str, int] = field(default_factory=lambda: {"CPU": 1, "GPU": 1})
    
    def __post_init__(self):
        """Validate and set default values."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.resources_per_worker.get("GPU", 0) == 0:
            self.use_gpu = False

class UnifiedTrainer:
    """Unified trainer for computer vision tasks using Ray on Databricks."""
    
    def __init__(
        self,
        config: Union[Dict[str, Any], UnifiedTrainerConfig],
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        logger: Optional[MLFlowLogger] = None  # Made optional for backward compatibility
    ):
        """Initialize the trainer with a config, model, data, and optionally a logger."""
        if isinstance(config, dict):
            # Simplified config parsing assuming keys match dataclass fields
            config = UnifiedTrainerConfig(**config)
        self.config = config
        
        self.model = model
        self.data_module = data_module
        self.trainer = None
        
        # Create logger if not provided
        if logger is None:
            logger = create_databricks_logger_for_task(
                task=self.config.task,
                model_name=self.config.model_name,
                log_model="all"  # Log all checkpoints automatically
            )
        
        self.logger = logger
        print("✅ UnifiedTrainer initialized with simplified MLflow integration.")

    def _init_callbacks(self) -> List[pl.Callback]:
        """
        Initialize all necessary training callbacks.
        This is now the single source of truth for callback configuration.
        """
        callbacks = []
        
        # 1. Model Checkpointing (This is watched by MLFlowLogger)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename=f"{self.config.model_name}-{{epoch}}-{{{self.config.monitor_metric}:.2f}}",
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            save_top_k=self.config.save_top_k,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        print(f"✅ ModelCheckpoint enabled (monitoring: '{self.config.monitor_metric}').")

        # 2. Early Stopping
        early_stopping = EarlyStopping(
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            patience=self.config.early_stopping_patience,
            verbose=True
        )
        callbacks.append(early_stopping)
        print(f"✅ EarlyStopping enabled (patience: {self.config.early_stopping_patience}).")

        # 3. Learning Rate Monitor
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        print("✅ LearningRateMonitor enabled.")

        # 4. (Optional) Copy checkpoints to a persistent volume
        if self.config.volume_checkpoint_dir:
            callbacks.append(VolumeCheckpoint(volume_dir=self.config.volume_checkpoint_dir))
            print(f"✅ VolumeCheckpoint enabled for: {self.config.volume_checkpoint_dir}")
        
        # 5. Ray Train Report Callback (for distributed training)
        if self.config.distributed:
            callbacks.append(RayTrainReportCallback())
            print("✅ RayTrainReportCallback enabled for distributed training.")
        
        print(f"✅ Total callbacks initialized: {len(callbacks)}")
        return callbacks
    
    def _init_trainer(self):
        """Initialize the PyTorch Lightning trainer."""
        callbacks = self._init_callbacks()
        
        trainer_params = {
            "max_epochs": self.config.max_epochs,
            "accelerator": "auto",
            "devices": "auto",
            "callbacks": callbacks,
            "log_every_n_steps": self.config.log_every_n_steps,
            "logger": self.logger
        }

        if self.config.distributed:
            trainer_params.update({
                "strategy": RayDDPStrategy(),
                "plugins": [RayLightningEnvironment()],
                "sync_batchnorm": True
            })
            self.trainer = pl.Trainer(**trainer_params)
            self.trainer = prepare_trainer(self.trainer) # Prepare for Ray
        else:
            self.trainer = pl.Trainer(**trainer_params)
    
    def train(self):
        """Train the model using either local or distributed training."""
        self._init_trainer()
        
        print("\n🚀 Starting training...")
        print(f"   Task: {self.config.task}")
        print(f"   Model: {self.config.model_name}")
        print(f"   Max epochs: {self.config.max_epochs}")
        print(f"   Monitor metric: {self.config.monitor_metric}")
        print(f"   Distributed: {self.config.distributed}")
        
        try:
            if self.config.distributed:
                # Use the LightningTrainer from Ray Train, which is the modern API
                from ray.train.lightning import LightningTrainer

                scaling_config = ScalingConfig(
                    num_workers=self.config.num_workers,
                    use_gpu=self.config.use_gpu,
                    resources_per_worker=self.config.resources_per_worker
                )
                
                run_config = RunConfig(
                    storage_path=self.config.checkpoint_dir,
                    name=self.logger.run_name, # Use the run name from the logger
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=self.config.save_top_k,
                        checkpoint_score_attribute=self.config.monitor_metric,
                        checkpoint_score_order=self.config.monitor_mode
                    )
                )

                # This is a higher-level API that wraps the pl.Trainer
                ray_trainer = LightningTrainer(
                    lightning_module=self.model,
                    lightning_datamodule=self.data_module,
                    trainer_init_config=self.trainer.init_kwargs, # Pass trainer config
                    scaling_config=scaling_config,
                    run_config=run_config
                )
                result = ray_trainer.fit()
            else:
                # Local training - MLFlowLogger handles all logging automatically
                self.trainer.fit(self.model, datamodule=self.data_module)
                result = self.trainer.callback_metrics

            # The MLFlowLogger's lifecycle is managed by the pl.Trainer.
            # It will automatically close the run on completion or failure.
            # No need for manual mlflow.log_param calls here.
            print("\n✅ Training finished. Logger has automatically handled run finalization.")
            return result
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
 
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
    
    def test(self, model=None, data_module=None):
        """Test the model using the underlying PyTorch Lightning trainer.
        
        Args:
            model: Optional model to test (uses self.model if not provided)
            data_module: Optional data module to test with (uses self.data_module if not provided)
            
        Returns:
            List of test results
        """
        if model is not None:
            self.model = model
        if data_module is not None:
            self.data_module = data_module
            
        if self.model is None or self.data_module is None:
            raise ValueError("Model and data module must be provided before testing")
        
        # Initialize trainer if not already done
        if self.trainer is None:
            self._init_trainer()
        
        # Run test
        results = self.trainer.test(self.model, datamodule=self.data_module)
        return results
    
    def get_metrics(self):
        """Get training metrics from MLflow."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(mlflow.active_run().info.run_id)
        return run.data.metrics 
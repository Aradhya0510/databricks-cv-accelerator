"""
Unified trainer for computer vision tasks using Databricks Serverless GPU compute.

This module provides a unified training interface that leverages Databricks Serverless GPU
compute for distributed training, allowing for interactive notebook-based multi-GPU training
without the limitations of traditional Lightning DDP strategies.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning as pl
import mlflow
import ray
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger
from ray import train
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our custom modules
from utils.logging import VolumeCheckpoint, create_databricks_logger

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
    volume_checkpoint_dir: Optional[str] = None
    save_top_k: int = 3
    
    # Distributed training settings
    distributed: bool = False
    use_ray: bool = False  # Whether to use Ray (multi-node) or Serverless GPU (single-node)
    use_serverless_gpu: bool = False  # Whether to use Serverless GPU compute
    num_workers: int = 1
    use_gpu: bool = True
    resources_per_worker: Dict[str, int] = field(default_factory=lambda: {"CPU": 1, "GPU": 1})
    master_port: Optional[str] = None  # Optional master port for traditional DDP
    
    # Serverless GPU specific settings
    serverless_gpu_type: str = "A10"  # A10 or H100
    serverless_gpu_count: int = 4  # Number of GPUs to use
    
    def __post_init__(self):
        """Validate and set default values."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.resources_per_worker.get("GPU", 0) == 0:
            self.use_gpu = False
        
        # Validate distributed settings
        if self.distributed and self.use_ray and not self.use_gpu:
            raise ValueError("Ray distributed training requires GPU support")
        
        # Validate serverless GPU settings
        if self.use_serverless_gpu and self.serverless_gpu_type not in ["A10", "H100"]:
            raise ValueError("Serverless GPU type must be 'A10' or 'H100'")
        
        if self.use_serverless_gpu and self.serverless_gpu_type == "H100" and self.serverless_gpu_count > 1:
            raise ValueError("H100 GPUs only support single-node workflows")

class UnifiedTrainerServerless:
    """Unified trainer for computer vision tasks using Databricks Serverless GPU compute."""
    
    def __init__(
        self,
        config: Union[Dict[str, Any], UnifiedTrainerConfig],
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        logger: Optional[MLFlowLogger] = None
    ):
        """Initialize the trainer with a config, model, data, and optionally a logger."""
        if isinstance(config, dict):
            config = UnifiedTrainerConfig(**config)
        self.config = config
        
        self.model = model
        self.data_module = data_module
        self.trainer = None
        
        self.logger = logger
        print("✅ UnifiedTrainerServerless initialized with Serverless GPU support.")
    
    def _init_callbacks(self) -> List[pl.Callback]:
        """Initialize all necessary training callbacks."""
        callbacks = []
        
        # 1. Model Checkpointing
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
        
        # 5. Ray Train Report Callback (for distributed training with Ray)
        if self.config.distributed and self.config.use_ray:
            callbacks.append(RayTrainReportCallback())
            print("✅ RayTrainReportCallback enabled for Ray distributed training.")
        
        print(f"✅ Total callbacks initialized: {len(callbacks)}")
        return callbacks
    
    def _setup_serverless_gpu_environment(self):
        """Set up environment for Serverless GPU compute."""
        try:
            # Check if serverless_gpu is available
            from serverless_gpu import distributed
            print("✅ Serverless GPU API available")
            
            # Set up environment variables for better performance
            os.environ.setdefault("NCCL_DEBUG", "WARN")
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
            
            print(f"🔧 Serverless GPU environment configured:")
            print(f"   GPU Type: {self.config.serverless_gpu_type}")
            print(f"   GPU Count: {self.config.serverless_gpu_count}")
            print(f"   NCCL_SOCKET_IFNAME: eth0")
            
        except ImportError:
            raise ImportError("Serverless GPU API not available. Please ensure you're running on Databricks Serverless GPU compute.")
        except Exception as e:
            print(f"⚠️  Serverless GPU setup failed: {e}")
            raise
    
    def _init_trainer(self):
        """Initialize the PyTorch Lightning trainer."""
        callbacks = self._init_callbacks()
        
        trainer_params = {
            "max_epochs": self.config.max_epochs,
            "accelerator": "gpu" if self.config.use_gpu else "cpu",
            "devices": "auto",
            "callbacks": callbacks,
            "log_every_n_steps": self.config.log_every_n_steps,
            "logger": self.logger
        }

        if self.config.distributed:
            if self.config.use_ray:
                # Use Ray strategy for multi-node training
                trainer_params.update({
                    "strategy": RayDDPStrategy(),
                    "plugins": [RayLightningEnvironment()],
                    "sync_batchnorm": True
                })
                self.trainer = pl.Trainer(**trainer_params)
                self.trainer = prepare_trainer(self.trainer)  # Prepare for Ray
                print("✅ Using Ray strategy for multi-node training.")
            elif self.config.use_serverless_gpu:
                # Use Serverless GPU for distributed training
                self._setup_serverless_gpu_environment()
                # For serverless GPU, we'll use standard DDP since the environment handles distribution
                trainer_params.update({
                    "strategy": "ddp",
                    "sync_batchnorm": True
                })
                self.trainer = pl.Trainer(**trainer_params)
                print("✅ Using DDP strategy with Serverless GPU compute.")
            else:
                # Fallback to traditional DDP (shouldn't be used in serverless context)
                trainer_params.update({
                    "strategy": "ddp",
                    "sync_batchnorm": True
                })
                self.trainer = pl.Trainer(**trainer_params)
                print("✅ Using traditional DDP strategy.")
        else:
            self.trainer = pl.Trainer(**trainer_params)
            print("✅ Using single-device training strategy.")
    
    def train(self):
        """Train the model using either local or distributed training."""
        print("\n🚀 Starting training...")
        print(f"   Task: {self.config.task}")
        print(f"   Model: {self.config.model_name}")
        print(f"   Max epochs: {self.config.max_epochs}")
        print(f"   Monitor metric: {self.config.monitor_metric}")
        print(f"   Distributed: {self.config.distributed}")
        if self.config.distributed:
            if self.config.use_ray:
                print(f"   Strategy: Ray (multi-node)")
            elif self.config.use_serverless_gpu:
                print(f"   Strategy: Serverless GPU ({self.config.serverless_gpu_type} x{self.config.serverless_gpu_count})")
            else:
                print(f"   Strategy: Traditional DDP")
        
        try:
            if self.config.distributed and self.config.use_ray:
                # Initialize trainer for Ray training
                self._init_trainer()
                # Use the LightningTrainer from Ray Train for multi-node
                from ray.train.lightning import LightningTrainer

                scaling_config = ScalingConfig(
                    num_workers=self.config.num_workers,
                    use_gpu=self.config.use_gpu,
                    resources_per_worker=self.config.resources_per_worker
                )
                
                run_config = RunConfig(
                    storage_path=self.config.checkpoint_dir,
                    name=self.logger.run_name,
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=self.config.save_top_k,
                        checkpoint_score_attribute=self.config.monitor_metric,
                        checkpoint_score_order=self.config.monitor_mode
                    )
                )

                ray_trainer = LightningTrainer(
                    lightning_module=self.model,
                    lightning_datamodule=self.data_module,
                    trainer_init_config=self.trainer.init_kwargs,
                    scaling_config=scaling_config,
                    run_config=run_config
                )
                result = ray_trainer.fit()
            elif self.config.distributed and self.config.use_serverless_gpu:
                # Use Serverless GPU distributed training
                from serverless_gpu import distributed
                
                # Serialize only the configuration, not the large objects
                config_dict = {
                    'task': self.config.task,
                    'model_name': self.config.model_name,
                    'max_epochs': self.config.max_epochs,
                    'log_every_n_steps': self.config.log_every_n_steps,
                    'monitor_metric': self.config.monitor_metric,
                    'checkpoint_dir': self.config.checkpoint_dir,
                    'early_stopping_patience': self.config.early_stopping_patience,
                    'save_top_k': self.config.save_top_k,
                    'learning_rate': self.config.learning_rate,
                    'weight_decay': self.config.weight_decay,
                    'gradient_clip_val': self.config.gradient_clip_val,
                    'accumulate_grad_batches': self.config.accumulate_grad_batches,
                    'use_gpu': self.config.use_gpu,
                    'distributed': self.config.distributed,
                    'use_serverless_gpu': self.config.use_serverless_gpu,
                    'serverless_gpu_type': self.config.serverless_gpu_type,
                    'serverless_gpu_count': self.config.serverless_gpu_count,
                    'use_ray': self.config.use_ray,
                    'num_workers': self.config.num_workers,
                    'master_port': self.config.master_port,
                    'mlflow_experiment_name': self.config.mlflow_experiment_name,
                    'mlflow_run_name': self.config.mlflow_run_name,
                    'mlflow_tags': self.config.mlflow_tags,
                    'mlflow_autolog': self.config.mlflow_autolog,
                    'output_dir': self.config.output_dir,
                    'save_predictions': self.config.save_predictions,
                    'save_metrics': self.config.save_metrics,
                    'save_model': self.config.save_model,
                    'model_save_path': self.config.model_save_path,
                    'predictions_save_path': self.config.predictions_save_path,
                    'metrics_save_path': self.config.metrics_save_path,
                    'volume_checkpoint_dir': self.config.volume_checkpoint_dir,
                    'use_volume_checkpoints': self.config.use_volume_checkpoints,
                    'data_config': self.data_config,
                    'model_config': self.model_config
                }
                
                @distributed(
                    gpus=self.config.serverless_gpu_count, 
                    gpu_type=self.config.serverless_gpu_type, 
                    remote=True
                )
                def distributed_train(config_dict):
                    """Distributed training function for Serverless GPU."""
                    import pytorch_lightning as pl
                    from src.tasks import get_task_module
                    from src.utils.logging import setup_mlflow_logger
                    
                    # Recreate the model and data module inside the distributed function
                    task_module = get_task_module(config_dict['task'])
                    
                    # Get the correct model and data module classes based on task
                    if config_dict['task'] == 'classification':
                        model = task_module.ClassificationModel(config_dict['model_config'])
                        data_module = task_module.ClassificationDataModule(config_dict['data_config'])
                    elif config_dict['task'] == 'detection':
                        model = task_module.DetectionModel(config_dict['model_config'])
                        data_module = task_module.DetectionDataModule(config_dict['data_config'])
                    elif config_dict['task'] == 'semantic_segmentation':
                        model = task_module.SemanticSegmentationModel(config_dict['model_config'])
                        data_module = task_module.SemanticSegmentationDataModule(config_dict['data_config'])
                    elif config_dict['task'] == 'instance_segmentation':
                        model = task_module.InstanceSegmentationModel(config_dict['model_config'])
                        data_module = task_module.InstanceSegmentationDataModule(config_dict['data_config'])
                    elif config_dict['task'] == 'universal_segmentation':
                        model = task_module.UniversalSegmentationModel(config_dict['model_config'])
                        data_module = task_module.UniversalSegmentationDataModule(config_dict['data_config'])
                    else:
                        raise ValueError(f"Unsupported task: {config_dict['task']}")
                    
                    # Setup logger
                    logger = setup_mlflow_logger(
                        experiment_name=config_dict['mlflow_experiment_name'],
                        run_name=config_dict['mlflow_run_name'],
                        tags=config_dict['mlflow_tags'],
                        autolog=config_dict['mlflow_autolog']
                    )
                    
                    # Initialize callbacks
                    callbacks = []
                    if config_dict['checkpoint_dir']:
                        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
                        callbacks.append(ModelCheckpoint(
                            dirpath=config_dict['checkpoint_dir'],
                            monitor=config_dict['monitor_metric'],
                            mode='max',
                            save_top_k=config_dict['save_top_k'],
                            save_last=True
                        ))
                        callbacks.append(EarlyStopping(
                            monitor=config_dict['monitor_metric'],
                            patience=config_dict['early_stopping_patience'],
                            mode='max'
                        ))
                        callbacks.append(LearningRateMonitor(logging_interval='step'))
                    
                    # Add volume checkpoint callback if enabled
                    if config_dict.get('use_volume_checkpoints', False) and config_dict.get('volume_checkpoint_dir'):
                        from src.utils.logging import VolumeCheckpoint
                        callbacks.append(VolumeCheckpoint(config_dict['volume_checkpoint_dir']))
                    
                    # Initialize trainer for this distributed process
                    trainer = pl.Trainer(
                        max_epochs=config_dict['max_epochs'],
                        accelerator="gpu",
                        devices="auto",
                        callbacks=callbacks,
                        log_every_n_steps=config_dict['log_every_n_steps'],
                        logger=logger,
                        strategy="ddp",
                        sync_batchnorm=True
                    )
                    
                    # Train the model
                    trainer.fit(model, datamodule=data_module)
                    return trainer.callback_metrics
                
                # Execute distributed training
                result = distributed_train.distributed(config_dict)
            else:
                # Local or traditional distributed training
                self._init_trainer()
                self.trainer.fit(self.model, datamodule=self.data_module)
                result = self.trainer.callback_metrics

            print("\n✅ Training finished. Logger has automatically handled run finalization.")
            return result
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def tune(self, search_space: dict, num_trials: int = 20):
        """Run hyperparameter tuning using Ray Tune."""
        if not self.config.distributed or not self.config.use_ray:
            raise ValueError("Hyperparameter tuning requires Ray distributed training mode")
        
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
        """Test the model using the underlying PyTorch Lightning trainer."""
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

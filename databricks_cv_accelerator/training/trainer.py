"""
Unified trainer for computer vision tasks using Ray or Databricks DDP.

This module provides a unified training interface that supports both
single-node distributed training (Databricks DDP) and multi-node
distributed training (Ray) on Databricks.

Key Features:
- Environment-aware strategy selection (Jobs vs Notebooks)
- Automatic multi-GPU DDP on Jobs compute
- Single GPU in notebooks to avoid experimental ddp_notebook
- Ray multi-node training with proper Spark cluster integration
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
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our custom modules
from utils.logging import VolumeCheckpoint, create_databricks_logger


# =============================================================================
# Environment Detection Helpers
# =============================================================================

def _is_databricks_job() -> bool:
    """Check if running in a Databricks Jobs environment (non-interactive)."""
    return os.getenv("DATABRICKS_JOB_RUN_ID") is not None


def _is_databricks_notebook() -> bool:
    """Check if running in a Databricks notebook (interactive)."""
    return (os.getenv("DATABRICKS_RUNTIME_VERSION") is not None) and not _is_databricks_job()


def _cuda_preinitialized() -> bool:
    """Check if CUDA has been initialized (can block ddp in notebooks)."""
    return torch.cuda.is_initialized()

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
    use_ray: bool = False  # Whether to use Ray (multi-node) or Databricks DDP (single-node)
    num_workers: int = 1
    use_gpu: bool = True
    resources_per_worker: Dict[str, int] = field(default_factory=lambda: {"CPU": 1, "GPU": 1})
    master_port: Optional[str] = None  # Optional master port for Databricks DDP
    
    # Strategy/device preferences (set by job scripts to override auto-detection)
    preferred_strategy: Optional[str] = None  # e.g., "ddp", "auto", "ddp_notebook"
    preferred_devices: Optional[Union[str, int]] = None  # e.g., "auto", 4, 1
    
    def __post_init__(self):
        """Validate and set default values."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.resources_per_worker.get("GPU", 0) == 0:
            self.use_gpu = False
        
        # Validate distributed settings
        if self.distributed and self.use_ray and not self.use_gpu:
            raise ValueError("Ray distributed training requires GPU support")

class UnifiedTrainer:
    def __init__(self, config, model, data_module, logger: Optional[MLFlowLogger] = None):
        if isinstance(config, dict):
            config = UnifiedTrainerConfig(**config)
        self.config = config
        self.model = model
        self.data_module = data_module
        self.trainer = None

        # If no logger provided, create a sensible default on Databricks
        if logger is None:
            # prefer experiment name = task/model for MLflow hygiene
            exp_name = f"cv/{self.config.task}/{self.config.model_name}"
            self.logger = create_databricks_logger(experiment_name=exp_name)
        else:
            self.logger = logger

        print("✅ UnifiedTrainer initialized with simplified MLflow integration.")

    def _choose_strategy_and_devices(self):
        """
        Environment-aware strategy and device selection.
        
        Returns:
            tuple: (strategy, devices) for pl.Trainer
            
        Rules:
        - If preferred_strategy/preferred_devices are set in config, use those (set by job script)
        - Interactive notebooks: use "auto" strategy with "auto" devices (single GPU, no CUDA probes)
        - Jobs (non-interactive): use DDP when >= 2 GPUs detected, else single device
        - Ray multi-node: handled separately (explicit RayDDPStrategy)
        """
        # Check for explicit preferences from job script (takes precedence)
        preferred_strategy = getattr(self.config, "preferred_strategy", None)
        preferred_devices = getattr(self.config, "preferred_devices", None)
        
        if preferred_strategy is not None and preferred_devices is not None:
            print(f"📌 Using explicit strategy/devices from config: {preferred_strategy}/{preferred_devices}")
            return preferred_strategy, preferred_devices
        
        # Fallback to environment-aware detection
        strategy = "auto"
        devices = "auto"
        
        if not (self.config.distributed and self.config.use_gpu):
            # Non-distributed or CPU-only: use defaults
            return strategy, devices
        
        # If using Ray, don't set strategy/devices here (Ray path handles it)
        if self.config.use_ray:
            return strategy, devices
        
        # Databricks Jobs: can use DDP safely
        if _is_databricks_job():
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 2:
                strategy = "ddp"
                devices = gpu_count
                print(f"📊 Jobs environment: using DDP with {devices} GPUs")
            elif gpu_count == 1:
                devices = 1
                print("📊 Jobs environment: single GPU detected, using single-device training")
            else:
                devices = "auto"
                print("⚠️  Jobs environment: no GPUs detected, falling back to CPU")
        
        # Databricks Notebook: avoid DDP, use single GPU
        elif _is_databricks_notebook():
            # Check if CUDA already initialized (would block ddp_notebook)
            if _cuda_preinitialized() and self.config.distributed:
                print("⚠️  CUDA pre-initialized in notebook; falling back to single-GPU to avoid ddp_notebook issues")
            
            # Let Lightning auto-select single GPU (don't probe CUDA ourselves)
            strategy = "auto"
            devices = "auto"
            print("📓 Notebook environment: using auto-selection (single GPU, no DDP)")
        
        return strategy, devices

    def _init_callbacks(self) -> List[pl.Callback]:
        callbacks = []
        ckpt = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename=f"{self.config.model_name}-{{epoch}}-{{{self.config.monitor_metric}:.4f}}",
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            save_top_k=self.config.save_top_k,
            save_last=True,
            auto_insert_metric_name=False,  # because we embed metric in filename explicitly
        )
        callbacks.append(ckpt)

        early = EarlyStopping(
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            patience=self.config.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early)

        callbacks.append(LearningRateMonitor(logging_interval="step"))

        if self.config.volume_checkpoint_dir:
            callbacks.append(VolumeCheckpoint(volume_dir=self.config.volume_checkpoint_dir))

        if self.config.distributed and self.config.use_ray:
            callbacks.append(RayTrainReportCallback())

        print(f"✅ Total callbacks initialized: {len(callbacks)}")
        return callbacks

    def _setup_ray_cluster(self):
        """
        Initialize Ray cluster on Databricks Spark cluster.
        
        This connects Ray to the Spark cluster workers, not local mode.
        Proper for multi-node distributed training.
        """
        if ray.is_initialized():
            print("✅ Ray already initialized")
            return
        
        try:
            from ray.util.spark import setup_ray_cluster
            
            # Setup Ray cluster on Spark workers (multi-node)
            address_info = setup_ray_cluster(
                num_worker_nodes=self.config.num_workers,
                num_cpus_per_worker=self.config.resources_per_worker.get("CPU", 1),
                num_gpus_per_worker=self.config.resources_per_worker.get("GPU", 1) if self.config.use_gpu else 0,
            )
            
            # Initialize Ray with the Spark cluster address
            ray.init(
                address=address_info["address"],
                ignore_reinit_error=True,
            )
            
            cpu_count = ray.available_resources().get("CPU", 0)
            gpu_count = ray.available_resources().get("GPU", 0)
            print(f"✅ Ray cluster on Spark: {cpu_count} CPUs, {gpu_count} GPUs across {self.config.num_workers} workers")
        except Exception as e:
            print(f"❌ Ray setup failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _setup_databricks_ddp_environment(self):
        """
        Configure environment for Databricks DDP (single-node multi-GPU).
        Sets NCCL parameters and distributed training environment variables.
        """
        try:
            from pyspark.sql import SparkSession
            import socket
            
            spark = SparkSession.getActiveSession()
            
            # NCCL configuration for Databricks
            os.environ.setdefault("NCCL_DEBUG", "WARN")
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
            os.environ.setdefault("NCCL_SHM_DISABLE", "1")  # Can help with shared memory issues

            # Get master address from Spark
            master_addr = None
            if spark is not None:
                master_addr = spark.conf.get("spark.driver.host")
            os.environ["MASTER_ADDR"] = master_addr or "127.0.0.1"

            # Set master port
            if self.config.master_port:
                os.environ["MASTER_PORT"] = str(self.config.master_port)
            else:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    os.environ["MASTER_PORT"] = str(s.getsockname()[1])

            print(f"🔧 DDP environment: MASTER_ADDR={os.environ['MASTER_ADDR']} PORT={os.environ['MASTER_PORT']}")
        except Exception as e:
            print(f"⚠️  Databricks DDP setup failed: {e}")
            import traceback
            traceback.print_exc()

    def _init_trainer(self):
        """
        Initialize PyTorch Lightning Trainer with environment-aware settings.
        
        Automatically selects strategy and devices based on:
        - Environment (Jobs vs Notebooks)
        - Available GPUs
        - Config settings
        """
        callbacks = self._init_callbacks()
        
        # Get environment-aware strategy and devices
        strategy, devices = self._choose_strategy_and_devices()

        trainer_params = {
            "max_epochs": self.config.max_epochs,
            "accelerator": "gpu" if self.config.use_gpu else "cpu",
            "devices": devices,
            "strategy": strategy,
            "callbacks": callbacks,
            "log_every_n_steps": self.config.log_every_n_steps,
            "logger": self.logger,
        }

        # Setup DDP environment only when using DDP strategy on Jobs
        if strategy == "ddp" and _is_databricks_job():
            self._setup_databricks_ddp_environment()
            trainer_params.update({"sync_batchnorm": True})

        self.trainer = pl.Trainer(**trainer_params)
        
        # Clear logging of final configuration
        env_type = "Jobs" if _is_databricks_job() else "Notebook" if _is_databricks_notebook() else "Unknown"
        print(f"✅ Trainer initialized [{env_type}]: strategy={strategy}, devices={devices}, accelerator={trainer_params['accelerator']}")

    def train(self):
        print("\n🚀 Starting training...")
        print(f"   Task={self.config.task} | Model={self.config.model_name} | epochs={self.config.max_epochs}")
        print(f"   Monitor={self.config.monitor_metric} ({self.config.monitor_mode})")
        print(f"   Distributed={self.config.distributed} | Ray={self.config.use_ray}")

        try:
            if self.config.distributed and self.config.use_ray:
                # Initialize Ray cluster on Databricks Spark
                self._setup_ray_cluster()
                
                # --- Ray-native path: LightningTrainer handles distributed training ---
                from ray.train.lightning import LightningTrainer
                
                scaling_config = ScalingConfig(
                    num_workers=self.config.num_workers,
                    use_gpu=self.config.use_gpu,
                    resources_per_worker=self.config.resources_per_worker,
                )

                # Trainer config for each Ray worker (mirrors pl.Trainer kwargs)
                trainer_init_config = {
                    "max_epochs": self.config.max_epochs,
                    "accelerator": "gpu" if self.config.use_gpu else "cpu",
                    "devices": 1,  # devices per worker
                    "callbacks": self._init_callbacks(),
                    "log_every_n_steps": self.config.log_every_n_steps,
                    "strategy": RayDDPStrategy(),
                    "plugins": [RayLightningEnvironment()],
                    "sync_batchnorm": True,
                }

                # Run name with fallback
                run_name = getattr(self.logger, "run_name", None) or f"{self.config.task}-{self.config.model_name}"

                run_config = RunConfig(
                    name=run_name,
                    storage_path=self.config.checkpoint_dir,  # Must be durable (UC Volumes or dbfs://)
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=self.config.save_top_k,
                        checkpoint_score_attribute=self.config.monitor_metric,
                        checkpoint_score_order=self.config.monitor_mode,
                    ),
                )

                ray_trainer = LightningTrainer(
                    lightning_module=self.model,
                    lightning_datamodule=self.data_module,
                    trainer_init_config=trainer_init_config,
                    scaling_config=scaling_config,
                    run_config=run_config,
                )
                
                print(f"🚀 Starting Ray distributed training across {self.config.num_workers} workers")
                result = ray_trainer.fit()
                
                # Return metrics from Ray result
                metrics = result.metrics if hasattr(result, "metrics") else {}
                print(f"✅ Ray training finished. Metrics: {list(metrics.keys())}")
                return metrics

            # --- non-Ray path ---
            self._init_trainer()
            self.trainer.fit(self.model, datamodule=self.data_module)
            
            # Convert callback_metrics (Tensors) to plain floats for JSON serialization
            out = {}
            for k, v in self.trainer.callback_metrics.items():
                try:
                    out[k] = float(v.item() if hasattr(v, "item") else v)
                except Exception:
                    pass  # Skip metrics that can't be converted
            
            print("\n✅ Training finished.")
            print(f"   Final metrics: {list(out.keys())}")
            return out

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def tune(self, param_space: dict, num_trials: int = 20):
        """
        Run hyperparameter tuning with Ray Tune.
        
        IMPORTANT: For production use, pass factory functions (callables) or classes
        for lightning_module and lightning_datamodule instead of instances.
        This method currently passes instances which may not serialize properly
        across workers.
        
        Args:
            param_space: Dictionary of hyperparameters to tune
            num_trials: Number of trials to run
            
        Returns:
            Best configuration found
        """
        if not (self.config.distributed and self.config.use_ray):
            raise ValueError("Hyperparameter tuning requires Ray distributed training mode.")

        # Initialize Ray cluster on Databricks
        self._setup_ray_cluster()

        from ray.train.lightning import LightningTrainer
        from ray import tune
        
        # Trainer init config for each Ray worker
        base_trainer_cfg = {
            "max_epochs": self.config.max_epochs,
            "accelerator": "gpu" if self.config.use_gpu else "cpu",
            "devices": 1,  # devices per worker
            "log_every_n_steps": self.config.log_every_n_steps,
            "strategy": RayDDPStrategy(),
            "plugins": [RayLightningEnvironment()],
            "sync_batchnorm": True,
            "callbacks": [
                ModelCheckpoint(
                    dirpath=self.config.checkpoint_dir,
                    filename=f"{self.config.model_name}-{{epoch}}-{{{self.config.monitor_metric}:.4f}}",
                    monitor=self.config.monitor_metric,
                    mode=self.config.monitor_mode,
                    save_top_k=self.config.save_top_k,
                    save_last=True,
                    auto_insert_metric_name=False,
                ),
                EarlyStopping(
                    monitor=self.config.monitor_metric,
                    mode=self.config.monitor_mode,
                    patience=self.config.early_stopping_patience,
                    verbose=True,
                ),
                LearningRateMonitor(logging_interval="step"),
                RayTrainReportCallback(),  # Required to report metrics to Tune
            ],
        }

        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            resources_per_worker=self.config.resources_per_worker,
        )

        # Run name with MLflow logger fallback
        run_name = getattr(self.logger, "run_name", None) or f"tune-{self.config.task}-{self.config.model_name}"

        trainable = LightningTrainer(
            lightning_module=self.model,  # TODO: Use factory function for production
            lightning_datamodule=self.data_module,  # TODO: Use factory function for production
            trainer_init_config=base_trainer_cfg,
            scaling_config=scaling_config,
            run_config=RunConfig(
                name=run_name,
                storage_path=self.config.checkpoint_dir,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=self.config.save_top_k,
                    checkpoint_score_attribute=self.config.monitor_metric,
                    checkpoint_score_order=self.config.monitor_mode,
                ),
            ),
        )

        scheduler = ASHAScheduler(
            metric=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            max_t=self.config.max_epochs,
            grace_period=max(1, min(10, self.config.max_epochs // 5)),
            reduction_factor=2,
        )

        tuner = tune.Tuner(
            trainable,
            tune_config=TuneConfig(
                num_samples=num_trials,
                scheduler=scheduler,
                metric=self.config.monitor_metric,
                mode=self.config.monitor_mode,
            ),
            param_space=param_space,
        )
        
        print(f"🔍 Starting hyperparameter tuning: {num_trials} trials")
        results = tuner.fit()
        best = results.get_best_result(metric=self.config.monitor_metric, mode=self.config.monitor_mode)
        
        print(f"✅ Best config found: {best.config}")
        print(f"   Best {self.config.monitor_metric}: {best.metrics.get(self.config.monitor_metric)}")
        
        return best.config

    def test(self, model=None, data_module=None):
        if model is not None: self.model = model
        if data_module is not None: self.data_module = data_module
        
        if self.model is None or self.data_module is None:
            raise ValueError("Model and data_module must be provided before testing")
        
        if self.trainer is None: self._init_trainer()
        return self.trainer.test(self.model, datamodule=self.data_module)

    def get_metrics(self):
        # Prefer the logger’s run_id over active_run()
        client = mlflow.tracking.MlflowClient()
        run_id = getattr(self.logger, "run_id", None)
        if run_id is None:
            raise RuntimeError("No MLflow run_id available on the logger.")
        run = client.get_run(run_id)
        return dict(run.data.metrics)
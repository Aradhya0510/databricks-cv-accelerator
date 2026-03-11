"""
Trainer for computer vision tasks on Databricks.

This module provides a training interface that supports both
single-GPU notebook training and multi-GPU DDP on Databricks Jobs.

Key Features:
- Environment-aware strategy selection (Jobs vs Notebooks)
- Automatic multi-GPU DDP on Jobs compute
- Single GPU in notebooks to avoid experimental ddp_notebook
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as pl
import mlflow
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy

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


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""
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

    # GPU
    use_gpu: bool = True

    def __post_init__(self):
        """Validate and set default values."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)


class Trainer:
    def __init__(self, config, model, data_module, logger: Optional[MLFlowLogger] = None):
        if isinstance(config, dict):
            config = TrainerConfig(**config)
        self.config = config
        self.model = model
        self.data_module = data_module
        self.trainer = None

        # If no logger provided, create a sensible default on Databricks
        if logger is None:
            exp_name = f"cv/{self.config.task}/{self.config.model_name}"
            self.logger = create_databricks_logger(experiment_name=exp_name)
        else:
            self.logger = logger

        print("Trainer initialized.")

    def _choose_strategy_and_devices(self):
        """
        Environment-aware strategy and device selection.

        Returns:
            tuple: (strategy, devices) for pl.Trainer

        Rules:
        - Jobs + >=2 GPUs: explicit DDPStrategy with NCCL + all GPUs
        - Jobs + 1 GPU: strategy="auto", devices=1
        - Notebook: strategy="auto", devices="auto" (single GPU)

        Note: We use explicit DDPStrategy() object instead of "ddp" string to bypass
        Lightning's interactive environment detection which incorrectly flags Databricks Jobs
        as interactive and blocks DDP training.
        """
        strategy = "auto"
        devices = "auto"

        if not self.config.use_gpu:
            return strategy, devices

        # Databricks Jobs: can use DDP safely with explicit DDPStrategy
        if _is_databricks_job():
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 2:
                strategy = DDPStrategy(
                    find_unused_parameters=False,
                    process_group_backend="nccl",
                )
                devices = gpu_count
                print(f"Jobs environment: using DDPStrategy with {devices} GPUs")
            elif gpu_count == 1:
                devices = 1
                print("Jobs environment: single GPU detected")
            else:
                devices = "auto"
                print("Jobs environment: no GPUs detected, falling back to CPU")

        # Databricks Notebook: avoid DDP, use single GPU
        elif _is_databricks_notebook():
            strategy = "auto"
            devices = "auto"
            print("Notebook environment: using auto-selection (single GPU)")

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
            auto_insert_metric_name=False,
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

        print(f"Total callbacks initialized: {len(callbacks)}")
        return callbacks

    def _setup_ddp_environment(self):
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
            os.environ.setdefault("NCCL_SHM_DISABLE", "1")

            # Get master address from Spark
            master_addr = None
            if spark is not None:
                master_addr = spark.conf.get("spark.driver.host")
            os.environ["MASTER_ADDR"] = master_addr or "127.0.0.1"

            # Set master port (find a free port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])

            print(f"DDP environment: MASTER_ADDR={os.environ['MASTER_ADDR']} PORT={os.environ['MASTER_PORT']}")
        except Exception as e:
            print(f"Databricks DDP setup failed: {e}")
            import traceback
            traceback.print_exc()

    def _init_trainer(self):
        """
        Initialize PyTorch Lightning Trainer with environment-aware settings.
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

        # Setup DDP environment only when using DDPStrategy
        is_ddp = isinstance(strategy, DDPStrategy)
        if is_ddp and _is_databricks_job():
            self._setup_ddp_environment()
            trainer_params.update({"sync_batchnorm": True})

        self.trainer = pl.Trainer(**trainer_params)

        env_type = "Jobs" if _is_databricks_job() else "Notebook" if _is_databricks_notebook() else "Unknown"
        strategy_name = strategy.__class__.__name__ if isinstance(strategy, DDPStrategy) else strategy
        print(f"Trainer initialized [{env_type}]: strategy={strategy_name}, devices={devices}, accelerator={trainer_params['accelerator']}")

    def train(self):
        print("\nStarting training...")
        print(f"   Task={self.config.task} | Model={self.config.model_name} | epochs={self.config.max_epochs}")
        print(f"   Monitor={self.config.monitor_metric} ({self.config.monitor_mode})")

        try:
            self._init_trainer()
            self.trainer.fit(self.model, datamodule=self.data_module)

            # Convert callback_metrics (Tensors) to plain floats for JSON serialization
            out = {}
            for k, v in self.trainer.callback_metrics.items():
                try:
                    out[k] = float(v.item() if hasattr(v, "item") else v)
                except Exception:
                    pass

            print("\nTraining finished.")
            print(f"   Final metrics: {list(out.keys())}")
            return out

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test(self, model=None, data_module=None):
        if model is not None: self.model = model
        if data_module is not None: self.data_module = data_module

        if self.model is None or self.data_module is None:
            raise ValueError("Model and data_module must be provided before testing")

        if self.trainer is None: self._init_trainer()
        return self.trainer.test(self.model, datamodule=self.data_module)

    def get_metrics(self):
        # Prefer the logger's run_id over active_run()
        client = mlflow.tracking.MlflowClient()
        run_id = getattr(self.logger, "run_id", None)
        if run_id is None:
            raise RuntimeError("No MLflow run_id available on the logger.")
        run = client.get_run(run_id)
        return dict(run.data.metrics)

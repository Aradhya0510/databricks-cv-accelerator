import os
import shutil
from typing import Optional, Dict, Any, Union, Literal

import mlflow
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

# --- 1. Simplified Logger Creation ---
def create_databricks_logger(
    experiment_name: str,
    run_name: Optional[str] = None,
    log_model: Union[bool, Literal["all"]] = "all",  # Use "all" to log every checkpoint, True for only the best
    tags: Optional[Dict[str, Any]] = None,
) -> MLFlowLogger:
    """
    Creates and configures an MLFlowLogger for Databricks.
    
    The tracking URI is automatically detected by MLflow when running in a
    Databricks notebook. Let the MLFlowLogger handle experiment setup on its own.
    
    Args:
        experiment_name: The MLflow experiment name
        run_name: Optional run name for this training session
        log_model: "all" to log every checkpoint, True for only the best, False for none
        tags: Optional dictionary of tags for the run
    """
    # Let MLFlowLogger handle experiment setup - no redundant mlflow.set_experiment() call
    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags,
        log_model=log_model,  # This is key for automatic checkpoint logging!
        # The 'mlflow-skinny' package is recommended for a lighter client
        save_dir='./mlruns'  # Optional: local save dir
    )
    print(f"✅ MLflowLogger created for experiment '{experiment_name}'")
    print(f"   Run name: {run_name}")
    print(f"   Log model: {log_model}")
    print(f"   Run ID: {logger.run_id}")
    return logger


# --- 2. A New, Focused Callback for Volume Copying ---
# This callback has a single responsibility: copying checkpoints to a volume.
# It doesn't interact with MLflow at all, preventing conflicts.

class VolumeCheckpoint(Callback):
    """
    A lightweight callback to copy saved checkpoints to a persistent volume.
    Relies on the ModelCheckpoint callback to have already saved the file.
    """
    def __init__(self, volume_dir: str):
        super().__init__()
        self.volume_dir = volume_dir
        os.makedirs(self.volume_dir, exist_ok=True)
        print(f"📁 Checkpoints will be copied to volume: {self.volume_dir}")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Check if ModelCheckpoint has saved a checkpoint in this epoch
        if trainer.checkpoint_callback and hasattr(trainer.checkpoint_callback, 'last_model_path'):
            last_ckpt_path = trainer.checkpoint_callback.last_model_path
            if os.path.exists(last_ckpt_path):
                try:
                    # Create a clear filename for the copy
                    dest_path = os.path.join(self.volume_dir, os.path.basename(last_ckpt_path))
                    shutil.copy2(last_ckpt_path, dest_path)
                    print(f"💾 Copied checkpoint to volume: {dest_path}")
                except Exception as e:
                    print(f"⚠️ Failed to copy checkpoint to volume: {e}")


# --- 3. Centralized Callback Setup ---
# A clean way to assemble all necessary callbacks.

def get_training_callbacks(
    checkpoint_dir: str,
    monitor_metric: str = "val_loss",
    monitor_mode: str = "min",
    save_top_k: int = 3,
    volume_checkpoint_dir: Optional[str] = None
) -> list:
    """Assembles all necessary callbacks for training."""
    
    # The main checkpoint callback that saves files locally
    # MLFlowLogger will watch this one!
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}-{"+monitor_metric+":.2f}",
        save_top_k=save_top_k,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True
    )
    
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Add the volume copy callback ONLY if a directory is provided
    if volume_checkpoint_dir:
        callbacks.append(VolumeCheckpoint(volume_dir=volume_checkpoint_dir))
        
    if torch.cuda.is_available():
        from lightning.pytorch.callbacks import DeviceStatsMonitor
        callbacks.append(DeviceStatsMonitor())

    print("✅ Assembled training callbacks.")
    return callbacks



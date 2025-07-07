from typing import Optional, Dict, Any
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback
import mlflow
import os
import torch


class MLflowCheckpointCallback(Callback):
    """Callback to log checkpoints to MLflow with volume path support."""
    
    def __init__(self, volume_checkpoint_dir: Optional[str] = None, log_every_n_epochs: int = 1):
        self.volume_checkpoint_dir = volume_checkpoint_dir
        self.log_every_n_epochs = log_every_n_epochs
        
        # Create volume checkpoint directory if provided
        if self.volume_checkpoint_dir:
            os.makedirs(self.volume_checkpoint_dir, exist_ok=True)
            print(f"📁 Volume checkpoint directory: {self.volume_checkpoint_dir}")
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Log checkpoint at the end of training epochs."""
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            # Get the best model path from the checkpoint callback
            checkpoint_path = None
            for callback in trainer.callbacks:
                if hasattr(callback, 'best_model_path') and callback.best_model_path:
                    checkpoint_path = callback.best_model_path
                    break
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Log to MLflow
                mlflow.log_artifact(checkpoint_path, f"checkpoints/epoch_{trainer.current_epoch}")
                
                # Also copy to volume directory if specified
                if self.volume_checkpoint_dir:
                    import shutil
                    volume_checkpoint_path = os.path.join(
                        self.volume_checkpoint_dir, 
                        f"epoch_{trainer.current_epoch}_{os.path.basename(checkpoint_path)}"
                    )
                    shutil.copy2(checkpoint_path, volume_checkpoint_path)
                    print(f"💾 Saved checkpoint to volume: {volume_checkpoint_path}")
    
    def on_validation_end(self, trainer, pl_module):
        """Log checkpoint after validation if it's the best so far."""
        checkpoint_path = None
        for callback in trainer.callbacks:
            if hasattr(callback, 'best_model_path') and callback.best_model_path:
                checkpoint_path = callback.best_model_path
                break
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Log to MLflow
            mlflow.log_artifact(checkpoint_path, "checkpoints/best_model")
            
            # Also copy to volume directory if specified
            if self.volume_checkpoint_dir:
                import shutil
                volume_best_path = os.path.join(
                    self.volume_checkpoint_dir, 
                    f"best_{os.path.basename(checkpoint_path)}"
                )
                shutil.copy2(checkpoint_path, volume_best_path)
                print(f"🏆 Saved best model to volume: {volume_best_path}")


def create_databricks_logger(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    log_model: bool = True
) -> MLFlowLogger:
    """Create an MLflow logger for Databricks environment.
    
    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking URI (auto-detected in Databricks)
        run_name: MLflow run name
        tags: MLflow tags
        log_model: Whether to log model artifacts
        
    Returns:
        MLFlowLogger instance
    """
    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=tags,
        log_model=log_model
    )


def create_enhanced_logging_callbacks(volume_checkpoint_dir: Optional[str] = None):
    """Create enhanced logging callbacks for better monitoring.
    
    Args:
        volume_checkpoint_dir: Optional volume path for storing checkpoints
    """
    from lightning.pytorch.callbacks import LearningRateMonitor, DeviceStatsMonitor
    
    callbacks = []
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Device stats monitoring (GPU memory, etc.)
    if torch.cuda.is_available():
        device_monitor = DeviceStatsMonitor()
        callbacks.append(device_monitor)
    
    # MLflow checkpoint logging with volume support
    mlflow_checkpoint_callback = MLflowCheckpointCallback(
        volume_checkpoint_dir=volume_checkpoint_dir,
        log_every_n_epochs=1
    )
    callbacks.append(mlflow_checkpoint_callback)
    
    return callbacks 
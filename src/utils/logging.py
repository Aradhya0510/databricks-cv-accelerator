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
                # Use Lightning's logger to log artifacts instead of direct MLflow calls
                if hasattr(trainer, 'loggers') and trainer.loggers:
                    for logger in trainer.loggers:
                        if isinstance(logger, MLFlowLogger):
                            try:
                                # Use the logger's experiment to log artifacts
                                logger.experiment.log_artifact(
                                    logger.run_id,
                                    checkpoint_path,
                                    f"checkpoints/epoch_{trainer.current_epoch}"
                                )
                                print(f"📊 Logged checkpoint to MLflow: epoch_{trainer.current_epoch}")
                            except Exception as e:
                                print(f"⚠️  Failed to log checkpoint to MLflow: {e}")
                            break
                
                # Also copy to volume directory if specified
                if self.volume_checkpoint_dir:
                    import shutil
                    volume_checkpoint_path = os.path.join(
                        self.volume_checkpoint_dir, 
                        f"epoch_{trainer.current_epoch}_{os.path.basename(checkpoint_path)}"
                    )
                    try:
                        shutil.copy2(checkpoint_path, volume_checkpoint_path)
                        print(f"💾 Saved checkpoint to volume: {volume_checkpoint_path}")
                    except Exception as e:
                        print(f"⚠️  Failed to save checkpoint to volume: {e}")
    
    def on_validation_end(self, trainer, pl_module):
        """Log checkpoint after validation if it's the best so far."""
        checkpoint_path = None
        for callback in trainer.callbacks:
            if hasattr(callback, 'best_model_path') and callback.best_model_path:
                checkpoint_path = callback.best_model_path
                break
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Use Lightning's logger to log artifacts instead of direct MLflow calls
            if hasattr(trainer, 'loggers') and trainer.loggers:
                for logger in trainer.loggers:
                    if isinstance(logger, MLFlowLogger):
                        try:
                            # Use the logger's experiment to log artifacts
                            logger.experiment.log_artifact(
                                logger.run_id,
                                checkpoint_path,
                                "checkpoints/best_model"
                            )
                            print(f"🏆 Logged best model to MLflow")
                        except Exception as e:
                            print(f"⚠️  Failed to log best model to MLflow: {e}")
                        break
            
            # Also copy to volume directory if specified
            if self.volume_checkpoint_dir:
                import shutil
                volume_best_path = os.path.join(
                    self.volume_checkpoint_dir, 
                    f"best_{os.path.basename(checkpoint_path)}"
                )
                try:
                    shutil.copy2(checkpoint_path, volume_best_path)
                    print(f"🏆 Saved best model to volume: {volume_best_path}")
                except Exception as e:
                    print(f"⚠️  Failed to save best model to volume: {e}")


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
    try:
        # Ensure experiment exists
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        # Create MLflowLogger with proper configuration
        logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_name=run_name,
            tags=tags,
            log_model=log_model
        )
        
        print(f"✅ MLflowLogger created successfully")
        print(f"   Experiment: {experiment_name}")
        print(f"   Run name: {run_name}")
        print(f"   Log model: {log_model}")
        
        return logger
        
    except Exception as e:
        print(f"❌ Failed to create MLflowLogger: {e}")
        # Return a dummy logger as fallback
        return None


def create_enhanced_logging_callbacks(volume_checkpoint_dir: Optional[str] = None):
    """Create enhanced logging callbacks for better monitoring.
    
    Args:
        volume_checkpoint_dir: Optional volume path for storing checkpoints
    """
    from lightning.pytorch.callbacks import LearningRateMonitor, DeviceStatsMonitor
    
    callbacks = []
    
    # Learning rate monitoring
    try:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        print("✅ Learning rate monitor added")
    except Exception as e:
        print(f"⚠️  Failed to add learning rate monitor: {e}")
    
    # Device stats monitoring (GPU memory, etc.)
    if torch.cuda.is_available():
        try:
            device_monitor = DeviceStatsMonitor()
            callbacks.append(device_monitor)
            print("✅ Device stats monitor added")
        except Exception as e:
            print(f"⚠️  Failed to add device stats monitor: {e}")
    
    # Note: MLflow checkpoint logging is now handled automatically by Lightning's MLflowLogger
    # when log_model=True is set in the logger configuration
    
    return callbacks


def log_training_parameters(config: Dict[str, Any]):
    """Log training parameters to MLflow if there's an active run."""
    if mlflow.active_run() is not None:
        try:
            # Log model parameters
            if 'model' in config:
                mlflow.log_params({
                    'model_name': config['model'].get('model_name', 'unknown'),
                    'task_type': config['model'].get('task_type', 'unknown'),
                    'num_classes': config['model'].get('num_classes', 0),
                    'pretrained': config['model'].get('pretrained', True)
                })
            
            # Log training parameters
            if 'training' in config:
                mlflow.log_params({
                    'max_epochs': config['training'].get('max_epochs', 0),
                    'learning_rate': config['training'].get('learning_rate', 0.0),
                    'weight_decay': config['training'].get('weight_decay', 0.0),
                    'monitor_metric': config['training'].get('monitor_metric', 'val_loss'),
                    'monitor_mode': config['training'].get('monitor_mode', 'min')
                })
            
            # Log data parameters
            if 'data' in config:
                mlflow.log_params({
                    'batch_size': config['data'].get('batch_size', 0),
                    'num_workers': config['data'].get('num_workers', 0),
                    'image_size': str(config['data'].get('image_size', [512, 512]))
                })
            
            print("✅ Training parameters logged to MLflow")
            
        except Exception as e:
            print(f"⚠️  Failed to log parameters to MLflow: {e}")


def log_final_metrics(metrics: Dict[str, float]):
    """Log final metrics to MLflow if there's an active run."""
    if mlflow.active_run() is not None and metrics:
        try:
            mlflow.log_metrics(metrics)
            print("✅ Final metrics logged to MLflow")
        except Exception as e:
            print(f"⚠️  Failed to log final metrics to MLflow: {e}") 
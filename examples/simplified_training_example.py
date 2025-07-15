#!/usr/bin/env python3
"""
Simplified Training Example with MLflow Integration

This example demonstrates the simplified MLflow integration approach
that removes redundant checkpoint logging and relies on the native
integration between MLFlowLogger and Lightning's ModelCheckpoint callback.
"""

import os
import sys
import torch
import lightning as pl
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.trainer import UnifiedTrainer, UnifiedTrainerConfig
from utils.logging import create_databricks_logger
from tasks.detection.model import DETRModel
from tasks.detection.data import DETRDataModule

def main():
    """Main training function demonstrating simplified MLflow integration."""
    
    print("🚀 Starting simplified DETR training example...")
    
    # 1. Define Configuration
    config = {
        "task": "detection",
        "model_name": "detr-resnet50",
        "max_epochs": 20,
        "log_every_n_steps": 50,
        "monitor_metric": "val_loss",
        "monitor_mode": "min",
        "early_stopping_patience": 5,
        "checkpoint_dir": "./checkpoints",
        "volume_checkpoint_dir": "/dbfs/mnt/my_volume/project_checkpoints",
        "save_top_k": 3,
        "distributed": False,  # Set to True for distributed training
    }
    trainer_config = UnifiedTrainerConfig(**config)

    # 2. ✨ Create the MLFlowLogger First ✨
    # This is the central piece for logging.
    
    # Create experiment name using Databricks user pattern
    try:
        import dbutils
        username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    except Exception:
        username = "unknown_user"
    
    experiment_name = f"/Users/{username}/{trainer_config.task}_pipeline"
    
    # Create default tags
    tags = {
        'framework': 'lightning',
        'model': trainer_config.model_name,
        'task': trainer_config.task,
        'dataset': 'coco',  # Default dataset
        'architecture': 'modular_cv_framework'
    }
    
    # Create run name
    run_name = f"{trainer_config.model_name}-run-{trainer_config.task}"
    
    mlf_logger = create_databricks_logger(
        experiment_name=experiment_name,
        run_name=run_name,
        log_model="all",  # Automatically log all ModelCheckpoint artifacts
        tags=tags
    )
    
    print(f"✅ MLflowLogger created:")
    print(f"   Experiment: {experiment_name}")
    print(f"   Run name: {run_name}")
    
    # 3. Initialize Model and Data
    # Remember to use self.save_hyperparameters() in your LightningModule
    # so the logger can automatically pick them up!
    model = DETRModel(
        num_classes=91,  # COCO classes
        learning_rate=0.0001,
        weight_decay=0.0001
    ) 
    
    data_module = DETRDataModule(
        data_dir="/dbfs/mnt/my_volume/coco_dataset",
        batch_size=4,
        num_workers=2
    )

    # 4. Initialize the UnifiedTrainer
    # Pass the config, model, data, and the logger to the trainer.
    unified_trainer = UnifiedTrainer(
        config=trainer_config,
        model=model,
        data_module=data_module,
        logger=mlf_logger
    )

    # 5. Start Training
    # All logging of params, metrics, and checkpoints is now handled automatically.
    print("\n🎯 Starting training...")
    result = unified_trainer.train()

    print("✅ Training completed!")
    print("📊 Final Metrics:", result)
    
    # 6. Evaluate the model
    print("\n🔍 Evaluating model...")
    evaluation_results = unified_trainer.test(model, data_module)
    print("📊 Evaluation Results:", evaluation_results)

if __name__ == "__main__":
    main() 
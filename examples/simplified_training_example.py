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
from utils.logging import create_databricks_logger_for_task
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
    mlf_logger = create_databricks_logger_for_task(
        task=trainer_config.task,
        model_name=trainer_config.model_name,
        run_name=f"{trainer_config.model_name}-run-{trainer_config.task}",
        log_model="all"  # Automatically log all ModelCheckpoint artifacts
    )
    
    print(f"✅ MLflowLogger created:")
    print(f"   Experiment: {mlf_logger.experiment}")
    print(f"   Run name: {mlf_logger.run_name}")
    print(f"   Run ID: {mlf_logger.run_id}")
    
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
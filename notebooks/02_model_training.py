# Databricks notebook source
# MAGIC %md
# MAGIC # 02. DETR Model Training
# MAGIC 
# MAGIC This notebook trains a DETR (DEtection TRansformer) model for object detection on the COCO dataset using our modular computer vision framework. We'll cover model initialization, training configuration, monitoring, and best practices for multi-GPU training.
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **DETR Training Process:**
# MAGIC 1. **Model Initialization**: Load pre-trained DETR with ResNet-50 backbone
# MAGIC 2. **Data Loading**: Efficient COCO data loading with preprocessing
# MAGIC 3. **Training Setup**: Multi-GPU training with Lightning
# MAGIC 4. **Monitoring**: Real-time metrics and MLflow tracking
# MAGIC 5. **Checkpointing**: Automatic model saving and early stopping
# MAGIC 
# MAGIC ### Key Training Concepts:
# MAGIC - **Bipartite Matching Loss**: Ensures unique predictions through Hungarian algorithm
# MAGIC - **Set Prediction**: Treats detection as a set prediction problem
# MAGIC - **Transformer Training**: Uses self-attention for global reasoning
# MAGIC - **Auxiliary Losses**: Help with convergence and stability
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 
# MAGIC 1. **Model Setup**: Initialize DETR model with proper configuration
# MAGIC 2. **Training Configuration**: Set up Lightning trainer with callbacks
# MAGIC 3. **Multi-GPU Training**: Configure distributed training on 4 GPUs
# MAGIC 4. **Monitoring**: Real-time metrics tracking and visualization
# MAGIC 5. **Model Saving**: Automatic checkpointing and model registration
# MAGIC 
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import Dependencies and Load Configuration

# COMMAND ----------

import sys
import os
import torch
import lightning
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')

from config import load_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from training.trainer import UnifiedTrainer
from utils.logging import setup_logger

# Load configuration from previous notebooks
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

# Load configuration
if os.path.exists(CONFIG_PATH):
    config = load_config(CONFIG_PATH)
else:
    print("⚠️  Config file not found. Using default detection config.")
    from config import get_default_config
    config = get_default_config("detection")

print("✅ Configuration loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model Initialization

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize DETR Model

# COMMAND ----------

def initialize_detr_model():
    """Initialize DETR model with proper configuration."""
    
    print("🔧 Initializing DETR model...")
    
    try:
        # Create detection model
        model = DetectionModel(config)
        
        print(f"✅ Model initialized successfully!")
        print(f"   Model name: {config['model']['model_name']}")
        print(f"   Number of classes: {config['model']['num_classes']}")
        print(f"   Task type: {config['model']['task_type']}")
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {total_params * 4 / 1e6:.1f} MB")
        
        return model
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return None

model = initialize_detr_model()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Architecture Analysis

# COMMAND ----------

def analyze_model_architecture(model):
    """Analyze the DETR model architecture."""
    
    if not model:
        return
    
    print("\n🏗️  DETR Architecture Analysis:")
    
    # Count layers by type
    layer_counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
    
    print("   Layer distribution:")
    for layer_type, count in sorted(layer_counts.items()):
        if count > 1:  # Only show layers that appear multiple times
            print(f"     {layer_type}: {count}")
    
    # Check for key DETR components
    has_backbone = any('backbone' in name for name, _ in model.named_modules())
    has_transformer = any('transformer' in name for name, _ in model.named_modules())
    has_query = any('query' in name for name, _ in model.named_modules())
    
    print(f"\n   Key components:")
    print(f"     Backbone (ResNet): {'✅' if has_backbone else '❌'}")
    print(f"     Transformer: {'✅' if has_transformer else '❌'}")
    print(f"     Object Queries: {'✅' if has_query else '❌'}")
    
    # Memory usage estimation
    if torch.cuda.is_available():
        model = model.cuda()
        torch.cuda.empty_cache()
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 800, 800).cuda()
        with torch.no_grad():
            _ = model(dummy_input)
        
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"\n   Estimated GPU memory per forward pass: {memory_used:.2f} GB")
        
        # Move back to CPU for training setup
        model = model.cpu()
        torch.cuda.empty_cache()

analyze_model_architecture(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Module Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Data Module

# COMMAND ----------

def setup_data_module():
    """Set up data module for training."""
    
    print("\n📊 Setting up data module...")
    
    try:
        # Create data module
        data_module = DetectionDataModule(config)
        
        # Setup for training
        data_module.setup('fit')
        
        print(f"✅ Data module setup complete!")
        print(f"   Train dataset: {len(data_module.train_dataset):,} samples")
        print(f"   Val dataset: {len(data_module.val_dataset):,} samples")
        print(f"   Test dataset: {len(data_module.test_dataset):,} samples")
        print(f"   Number of classes: {len(data_module.train_dataset.class_names)}")
        
        # Test data loading
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Batch size: {config['data']['batch_size']}")
        print(f"   Workers: {config['data']['num_workers']}")
        
        return data_module
        
    except Exception as e:
        print(f"❌ Data module setup failed: {e}")
        return None

data_module = setup_data_module()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Loading Performance Test

# COMMAND ----------

def test_data_loading_performance(data_module):
    """Test data loading performance and memory usage."""
    
    if not data_module:
        return
    
    print("\n⚡ Testing data loading performance...")
    
    import time
    
    # Test train loader performance
    train_loader = data_module.train_dataloader()
    
    # Time a few batches
    start_time = time.time()
    batch_times = []
    
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Test 5 batches
            break
        
        batch_start = time.time()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        print(f"   Batch {i+1}: {batch_time:.3f}s")
    
    total_time = time.time() - start_time
    avg_batch_time = np.mean(batch_times)
    
    print(f"\n📈 Performance Summary:")
    print(f"   Average batch loading time: {avg_batch_time:.3f}s")
    print(f"   Estimated batches per second: {1/avg_batch_time:.1f}")
    print(f"   Total test time: {total_time:.2f}s")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"   GPU memory used: {memory_used:.2f} GB")
        
        # Clear memory
        torch.cuda.empty_cache()
    
    return avg_batch_time

avg_batch_time = test_data_loading_performance(data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Training Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Trainer with Callbacks

# COMMAND ----------

def setup_trainer():
    """Set up Lightning trainer with callbacks and logging."""
    
    print("\n🚀 Setting up trainer...")
    
    try:
        # Create trainer
        trainer = UnifiedTrainer(config)
        
        print(f"✅ Trainer setup complete!")
        print(f"   Max epochs: {config['training']['max_epochs']}")
        print(f"   Learning rate: {config['training']['learning_rate']}")
        print(f"   Weight decay: {config['training']['weight_decay']}")
        print(f"   Scheduler: {config['training']['scheduler']}")
        print(f"   Early stopping patience: {config['training']['early_stopping_patience']}")
        print(f"   Monitor metric: {config['training']['monitor_metric']}")
        print(f"   Monitor mode: {config['training']['monitor_mode']}")
        
        # Check GPU configuration
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"   GPUs available: {num_gpus}")
            print(f"   Distributed training: {config['training']['distributed']}")
            
            if config['training']['distributed']:
                print(f"   Strategy: DDP")
                print(f"   Sync batch norm: Enabled")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        return None

trainer = setup_trainer()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Callbacks and Monitoring

# COMMAND ----------

def setup_monitoring():
    """Set up monitoring and logging."""
    
    print("\n📊 Setting up monitoring...")
    
    # Initialize logging
    logger = setup_logger(
        name="detr_training",
        experiment_name=config['mlflow']['experiment_name'],
        run_name=config['mlflow']['run_name'],
        log_dir=f"{BASE_VOLUME_PATH}/logs"
    )
    
    print(f"✅ Logging setup complete!")
    print(f"   Experiment: {config['mlflow']['experiment_name']}")
    print(f"   Run name: {config['mlflow']['run_name']}")
    print(f"   Log directory: {BASE_VOLUME_PATH}/logs")
    
    # Log configuration
    logger.info("Starting DETR training")
    logger.info(f"Configuration: {config}")
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"Dataset: {len(data_module.train_dataset)} train, {len(data_module.val_dataset)} val")
    
    return logger

logger = setup_monitoring()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Training Execution

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start Training

# COMMAND ----------

def start_training(model, data_module, trainer, logger):
    """Start the training process."""
    
    if not all([model, data_module, trainer, logger]):
        print("❌ Cannot start training - missing components")
        return False
    
    print("\n🎯 Starting DETR training...")
    print("=" * 60)
    
    try:
        # Log training start
        logger.info("Training started")
        
        # Start training
        trainer.fit(model, data_module)
        
        # Log training completion
        logger.info("Training completed successfully")
        
        print("✅ Training completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
        return False

# Start training
training_success = start_training(model, data_module, trainer, logger)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Progress Monitoring

# COMMAND ----------

def monitor_training_progress():
    """Monitor training progress and metrics."""
    
    print("\n📈 Training Progress Monitoring:")
    
    # Check if training logs exist
    log_dir = f"{BASE_VOLUME_PATH}/logs"
    if os.path.exists(log_dir):
        print(f"   Log directory: {log_dir}")
        
        # List log files
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        print(f"   Log files found: {len(log_files)}")
        
        for log_file in log_files:
            print(f"     - {log_file}")
    
    # Check MLflow experiments
    try:
        experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
        if experiment:
            print(f"   MLflow experiment: {experiment.name}")
            print(f"   Experiment ID: {experiment.experiment_id}")
            
            # Get latest run
            runs = mlflow.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
            if not runs.empty:
                latest_run = runs.iloc[0]
                print(f"   Latest run: {latest_run['tags.mlflow.runName']}")
                print(f"   Status: {latest_run['status']}")
                
                if latest_run['status'] == 'FINISHED':
                    print(f"   Final loss: {latest_run.get('metrics.train_loss', 'N/A')}")
                    print(f"   Final mAP: {latest_run.get('metrics.val_map', 'N/A')}")
        else:
            print("   MLflow experiment not found")
            
    except Exception as e:
        print(f"   MLflow monitoring error: {e}")

# Monitor progress (this would be called during training)
if training_success:
    monitor_training_progress()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Evaluation and Saving

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Trained Model

# COMMAND ----------

def evaluate_model(model, data_module, trainer):
    """Evaluate the trained model on validation set."""
    
    if not all([model, data_module, trainer]):
        print("❌ Cannot evaluate - missing components")
        return None
    
    print("\n🔍 Evaluating trained model...")
    
    try:
        # Run evaluation
        results = trainer.test(model, data_module)
        
        print("✅ Evaluation completed!")
        print("📊 Test Results:")
        
        for metric, value in results[0].items():
            print(f"   {metric}: {value:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

# Evaluate model
if training_success:
    evaluation_results = evaluate_model(model, data_module, trainer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save and Register Model

# COMMAND ----------

def save_and_register_model(model, trainer, evaluation_results):
    """Save the trained model and register it in MLflow."""
    
    if not model or not trainer:
        print("❌ Cannot save model - missing components")
        return False
    
    print("\n💾 Saving and registering model...")
    
    try:
        # Get best model checkpoint
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if hasattr(callback, 'best_model_path'):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path'):
            best_model_path = checkpoint_callback.best_model_path
            print(f"   Best model path: {best_model_path}")
            
            # Save model artifacts
            model_dir = f"{BASE_VOLUME_PATH}/models/detr_final"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model state dict
            torch.save(model.state_dict(), f"{model_dir}/model.pth")
            
            # Save configuration
            import yaml
            with open(f"{model_dir}/config.yaml", 'w') as f:
                yaml.dump(config, f)
            
            # Save evaluation results
            if evaluation_results:
                with open(f"{model_dir}/evaluation_results.yaml", 'w') as f:
                    yaml.dump(evaluation_results[0], f)
            
            print(f"✅ Model saved to: {model_dir}")
            
            # Register in MLflow
            try:
                with mlflow.start_run():
                    mlflow.log_artifact(model_dir, "model")
                    mlflow.log_metrics(evaluation_results[0] if evaluation_results else {})
                    print("✅ Model registered in MLflow")
                    
            except Exception as e:
                print(f"⚠️  MLflow registration failed: {e}")
            
            return True
        else:
            print("⚠️  No checkpoint callback found")
            return False
            
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        return False

# Save and register model
if training_success:
    model_saved = save_and_register_model(model, trainer, evaluation_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Training Summary and Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Summary

# COMMAND ----------

print("=" * 60)
print("DETR TRAINING SUMMARY")
print("=" * 60)

print(f"✅ Model Configuration:")
print(f"   Model: {config['model']['model_name']}")
print(f"   Task: {config['model']['task_type']}")
print(f"   Classes: {config['model']['num_classes']}")

print(f"\n✅ Training Configuration:")
print(f"   Max epochs: {config['training']['max_epochs']}")
print(f"   Learning rate: {config['training']['learning_rate']}")
print(f"   Batch size: {config['data']['batch_size']}")
print(f"   GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

print(f"\n✅ Data Configuration:")
print(f"   Train samples: {len(data_module.train_dataset) if data_module else 'N/A'}")
print(f"   Val samples: {len(data_module.val_dataset) if data_module else 'N/A'}")
print(f"   Test samples: {len(data_module.test_dataset) if data_module else 'N/A'}")

print(f"\n✅ Training Results:")
print(f"   Training success: {'Yes' if training_success else 'No'}")
print(f"   Evaluation completed: {'Yes' if evaluation_results else 'No'}")
print(f"   Model saved: {'Yes' if model_saved else 'No'}")

if evaluation_results:
    print(f"\n📊 Final Metrics:")
    for metric, value in evaluation_results[0].items():
        print(f"   {metric}: {value:.4f}")

print(f"\n📁 Output Paths:")
print(f"   Checkpoints: {config['training']['checkpoint_dir']}")
print(f"   Results: {config['output']['results_dir']}")
print(f"   Logs: {BASE_VOLUME_PATH}/logs")
print(f"   Models: {BASE_VOLUME_PATH}/models")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding DETR Training

# MAGIC 
# MAGIC ### Key Training Concepts:
# MAGIC 
# MAGIC **1. Bipartite Matching Loss:**
# MAGIC - DETR uses Hungarian algorithm to match predictions to ground truth
# MAGIC - Ensures unique predictions without NMS
# MAGIC - Combines classification and bounding box losses
# MAGIC 
# MAGIC **2. Set Prediction:**
# MAGIC - Treats detection as a set prediction problem
# MAGIC - All predictions made simultaneously (not autoregressive)
# MAGIC - Uses learned object queries to "query" the image
# MAGIC 
# MAGIC **3. Transformer Training:**
# MAGIC - Self-attention captures global relationships
# MAGIC - Cross-attention between queries and image features
# MAGIC - Position embeddings help with spatial understanding
# MAGIC 
# MAGIC **4. Training Strategy:**
# MAGIC - **Learning Rate**: Start with 1e-4, use cosine annealing
# MAGIC - **Batch Size**: 16 total (4 per GPU on 4-GPU setup)
# MAGIC - **Epochs**: 50-300 depending on dataset size
# MAGIC - **Auxiliary Losses**: Help with convergence
# MAGIC 
# MAGIC **5. Monitoring Metrics:**
# MAGIC - **mAP**: Mean Average Precision (primary metric)
# MAGIC - **Loss**: Total loss (classification + bbox)
# MAGIC - **Learning Rate**: Current learning rate
# MAGIC - **GPU Memory**: Memory usage per GPU
# MAGIC 
# MAGIC **6. Best Practices:**
# MAGIC - Use gradient clipping to prevent exploding gradients
# MAGIC - Monitor loss curves for convergence
# MAGIC - Use early stopping to prevent overfitting
# MAGIC - Save best model based on validation mAP
# MAGIC - Use mixed precision for memory efficiency
# MAGIC 
# MAGIC The training process leverages DETR's unique approach to object detection, using transformers and set prediction to achieve state-of-the-art results on COCO! 
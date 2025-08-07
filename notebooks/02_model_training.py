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

# (Databricks only) Install requirements and restart Python if running interactively
# %pip install -r "../requirements.txt"
# dbutils.library.restartPython()
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
from lightning.pytorch.loggers import MLFlowLogger

# Load configuration from previous notebooks
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

# Set up volume directories
CHECKPOINT_DIR = f"{BASE_VOLUME_PATH}/checkpoints"
RESULTS_DIR = f"{BASE_VOLUME_PATH}/results"
LOGS_DIR = f"{BASE_VOLUME_PATH}/logs"

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print(f"📁 Volume directories created:")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Results: {RESULTS_DIR}")
print(f"   Logs: {LOGS_DIR}")

# Load configuration
if os.path.exists(CONFIG_PATH):
    config = load_config(CONFIG_PATH)
    # Update checkpoint directory to use volume path
    config['training']['checkpoint_dir'] = CHECKPOINT_DIR
    
    # Fix config structure - move epochs from model to training if needed
    if 'model' in config and 'epochs' in config['model']:
        if 'training' not in config:
            config['training'] = {}
        config['training']['max_epochs'] = config['model']['epochs']
        print(f"✅ Fixed config: moved epochs ({config['model']['epochs']}) from model to training.max_epochs")
else:
    print("⚠️  Config file not found. Using default detection config.")
    from config import get_default_config
    config = get_default_config("detection")
    # Update checkpoint directory to use volume path
    config['training']['checkpoint_dir'] = CHECKPOINT_DIR

print("✅ Configuration loaded successfully!")
print(f"📁 Checkpoint directory: {config['training']['checkpoint_dir']}")

# Validate configuration for simplified MLflow integration
from utils.config_validator import validate_config_for_simplified_mlflow, print_config_compatibility_report

# Print compatibility report
print_config_compatibility_report(config)

# Validate and update config for simplified MLflow integration
try:
    validated_config = validate_config_for_simplified_mlflow(config)
    print("✅ Configuration validated for simplified MLflow integration!")
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
    print("Please check your configuration file and ensure all required fields are present.")
    raise

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
        # Prepare model config with num_workers from data config
        model_config = config["model"].copy()
        model_config["num_workers"] = config["data"]["num_workers"]
        
        # Create detection model
        model = DetectionModel(model_config)
        
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
        # Setup adapter first
        from tasks.detection.adapters import get_input_adapter
        
        # Fix image size handling - handle both list and scalar formats
        image_size = config["data"].get("image_size", 800)
        if isinstance(image_size, list):
            image_size = image_size[0]  # Use first value if it's a list
        elif isinstance(image_size, dict):
            image_size = image_size.get("height", 800)  # Use height if it's a dict
        
        adapter = get_input_adapter(config["model"]["model_name"], image_size=image_size)
        if adapter is None:
            print("❌ Failed to create adapter")
            return None
        
        # Create data module with data config only
        data_module = DetectionDataModule(config["data"])
        
        # Assign adapter to data module
        data_module.adapter = adapter
        
        # Setup for training
        data_module.setup('fit')
        
        print(f"✅ Data module setup complete!")
        print(f"   Train dataset: {len(data_module.train_dataset):,} samples")
        print(f"   Val dataset: {len(data_module.val_dataset):,} samples")
        # print(f"   Test dataset: {len(data_module.test_dataset):,} samples")
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
    
    # Setup adapter first
    from tasks.detection.adapters import get_input_adapter
    
    # Fix image size handling - handle both list and scalar formats
    image_size = config["data"].get("image_size", 800)
    if isinstance(image_size, list):
        image_size = image_size[0]  # Use first value if it's a list
    elif isinstance(image_size, dict):
        image_size = image_size.get("height", 800)  # Use height if it's a dict
    
    adapter = get_input_adapter(config["model"]["model_name"], image_size=image_size)
    if adapter is None:
        print("❌ Failed to create adapter")
        return False
    
    # Create data module with data config only
    data_module = DetectionDataModule(config["data"])
    
    # Assign adapter to data module
    data_module.adapter = adapter
    
    # Setup the data module to create datasets
    data_module.setup()
    
    # Test memory usage with a few batches
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

def setup_trainer_config():
    """Set up trainer configuration for the simplified MLflow integration."""
    
    print("\n🚀 Setting up trainer configuration...")
    
    try:
        # Create trainer config for the simplified approach
        trainer_config = {
            'task': config['model']['task_type'],
            'model_name': config['model']['model_name'],
            'max_epochs': config['training']['max_epochs'],
            'log_every_n_steps': config['training']['log_every_n_steps'],
            'monitor_metric': config['training']['monitor_metric'],
            'monitor_mode': config['training']['monitor_mode'],
            'early_stopping_patience': config['training']['early_stopping_patience'],
            'checkpoint_dir': f"{BASE_VOLUME_PATH}/checkpoints",
            'volume_checkpoint_dir': f"{BASE_VOLUME_PATH}/volume_checkpoints",
            'save_top_k': 3,
            'distributed': config['training']['distributed']
        }
        
        print(f"✅ Trainer configuration setup complete!")
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
        
        return trainer_config
        
    except Exception as e:
        print(f"❌ Trainer configuration setup failed: {e}")
        return None

trainer_config = setup_trainer_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Callbacks and Monitoring

# COMMAND ----------

# Set up MLflowLogger using the simplified approach
def setup_monitoring(data_module=None):
    """Set up monitoring and logging with simplified MLflowLogger integration."""
    
    print("\n📊 Setting up monitoring...")
    
    # Get task from config
    task = config['model']['task_type']
    model_name = config['model']['model_name']
    
    # Create experiment name using Databricks user pattern
    username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()    
    experiment_name = f"/Users/{username}/{task}_pipeline"
    
    # Create default tags
    tags = {
        'framework': 'lightning',
        'model': model_name,
        'task': task,
        'dataset': 'coco',  # Default dataset
        'architecture': 'modular_cv_framework',
        'training_config': str(config['training'])
    }
    
    # Create run name
    run_name = f"{model_name}-{task}-training"
    
    # Use the simplified logger creation function
    from utils.logging import create_databricks_logger
    
    mlf_logger = create_databricks_logger(
        experiment_name=experiment_name,
        run_name=run_name,
        log_model="all",  # Log all checkpoints automatically
        tags=tags
    )
    
    print(f"✅ MLflowLogger setup complete!")
    print(f"   Experiment: {experiment_name}")
    print(f"   Run name: {run_name}")
    print(f"   Log model: all")
    
    # Log configuration using print
    print("Starting DETR training")
    print(f"Model: {config['model']['model_name']}")
    
    # Only log dataset info if data_module is provided
    if data_module:
        print(f"Dataset: {len(data_module.train_dataset)} train, {len(data_module.val_dataset)} val")
    else:
        print("Dataset: Will be loaded during training")
    
    return mlf_logger

# Set up monitoring with data module
logger = setup_monitoring(data_module)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Training Execution

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start Training

# COMMAND ----------

def start_training(model, data_module, trainer_config, logger):
    """Start the training process with simplified logging integration."""
    
    if not all([model, data_module, trainer_config]):
        print("❌ Cannot start training - missing components")
        print(f"   Model: {'✅' if model else '❌'}")
        print(f"   Data module: {'✅' if data_module else '❌'}")
        print(f"   Trainer config: {'✅' if trainer_config else '❌'}")
        return False, None
    
    print("\n🎯 Starting DETR training...")
    print("=" * 60)
    
    # Memory cleanup before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"🧹 GPU memory cleared before training")
        print(f"   Initial allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"   Initial reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    try:
        # Initialize trainer with model, data module, and logger
        # The new UnifiedTrainer expects these as constructor parameters
        unified_trainer = UnifiedTrainer(
            config=trainer_config,
            model=model,
            data_module=data_module,
            logger=logger
        )
        
        print(f"✅ UnifiedTrainer initialized with:")
        print(f"   Model: {type(model).__name__}")
        print(f"   Data module: {type(data_module).__name__}")
        print(f"   Logger: {type(logger).__name__}")
        
        # Start training using the simplified approach
        print("🚀 Starting training...")
        result = unified_trainer.train()
        
        print("✅ Training completed successfully!")
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_allocated = torch.cuda.memory_allocated() / 1e9
            final_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"🧹 Final GPU memory - Allocated: {final_allocated:.2f} GB, Reserved: {final_reserved:.2f} GB")
        
        return True, unified_trainer
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

# Start training
training_success, unified_trainer = start_training(model, data_module, trainer_config, logger)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Status Check

# COMMAND ----------

def check_training_status():
    """Simple status check after training completion."""
    
    if training_success:
        print("\n✅ Training completed successfully!")
        print("📊 Check MLflow UI for detailed metrics and artifacts")
        print("🔗 You can find your experiment in the MLflow tracking UI")
    else:
        print("\n❌ Training failed!")
        print("📊 Check MLflow UI for error details and logs")
        print("🔗 Review the experiment run in MLflow tracking UI")

# Check training status
check_training_status()
    


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Evaluation and Saving

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Trained Model

# COMMAND ----------

def evaluate_model(model, data_module, unified_trainer):
    """Evaluate the trained model on validation set."""
    
    if not all([model, data_module, unified_trainer]):
        print("❌ Cannot evaluate - missing components")
        return None
    
    print("\n🔍 Evaluating trained model...")
    
    try:
        # Use the UnifiedTrainer's test method
        results = unified_trainer.test(model, data_module)
        
        print("✅ Evaluation completed!")
        print("📊 Test Results:")
        
        if results and len(results) > 0:
            for metric, value in results[0].items():
                print(f"   {metric}: {value:.4f}")
        else:
            print("   No results available")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

# Evaluate model
if training_success and unified_trainer is not None:
    evaluation_results = evaluate_model(model, data_module, unified_trainer)
else:
    evaluation_results = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save and Register Model

# COMMAND ----------

def save_and_register_model(model, unified_trainer, evaluation_results):
    """Save the trained model and register it in MLflow."""
    
    if not model or not unified_trainer:
        print("❌ Cannot save model - missing components")
        return False
    
    print("\n💾 Saving and registering model...")
    
    try:
        # Get best model checkpoint from the underlying PyTorch Lightning trainer
        checkpoint_callback = None
        if hasattr(unified_trainer, 'trainer') and unified_trainer.trainer is not None:
            for callback in unified_trainer.trainer.callbacks:
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
            
            # Register in MLflow - find the most recent run or create a new one
            try:
                # Try to find the most recent run from the same experiment
                task = config['model']['task_type']
                username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
                experiment_name = f"/Users/{username}/{task}_pipeline"
                
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    # Get the most recent run
                    runs = mlflow.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
                    if not runs.empty:
                        latest_run_id = runs.iloc[0]['run_id']
                        
                        # Log artifacts to the most recent run
                        with mlflow.start_run(run_id=latest_run_id):
                            mlflow.log_artifact(model_dir, "final_model")
                            if evaluation_results:
                                mlflow.log_metrics(evaluation_results[0])
                            print("✅ Model registered in MLflow (added to existing run)")
                    else:
                        # Create a new run for model registration
                        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="model_registration"):
                            mlflow.log_artifact(model_dir, "final_model")
                            if evaluation_results:
                                mlflow.log_metrics(evaluation_results[0])
                            print("✅ Model registered in MLflow (new run created)")
                else:
                    print("⚠️  Could not find MLflow experiment for model registration")
                    
            except Exception as e:
                print(f"⚠️  MLflow registration failed: {e}")
                print(f"   Model artifacts saved locally at: {model_dir}")
            
            return True
        else:
            print("⚠️  No checkpoint callback found")
            return False
            
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        return False

# Save and register model
if training_success and unified_trainer is not None:
    model_saved = save_and_register_model(model, unified_trainer, evaluation_results)
else:
    model_saved = False

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
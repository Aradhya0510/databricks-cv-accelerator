"""
Model Training Script for Databricks Jobs

Simplified training script designed to run on Databricks Jobs compute
with full multi-GPU DDP support. Configure via Lakeflow Jobs UI parameters.

Usage in Databricks Jobs UI:
    1. Select this file as Python script
    2. In Parameters field, add JSON array:
       ["--config_path", "/Volumes/.../config.yaml", 
        "--src_path", "/Workspace/.../src",
        "--force_strategy", "ddp",
        "--devices", "auto",
        "--force_jobs"]
    3. Select GPU cluster (e.g., g5.12xlarge for 4 GPUs)
    4. Run job

The script will automatically use all available GPUs with DDP strategy.
"""

# Environment setup BEFORE importing torch/lightning
# This prevents Lightning from incorrectly detecting as notebook environment
import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PL_DISABLE_FORK", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Silence TensorFlow warnings

# Remove IPython modules if preloaded by Lakeflow Jobs harness
for module_name in ("ipykernel", "IPython"):
    if module_name in sys.modules:
        del sys.modules[module_name]

# Now safe to import torch and other libraries
import argparse
from pathlib import Path
import warnings
from datetime import datetime

import torch
torch.set_float32_matmul_precision("high")  # Enable Tensor Cores for A10G/A100

import mlflow


def _is_jobs_env(args) -> bool:
    """
    Determine if running in Databricks Jobs (non-interactive) environment.
    
    Checks multiple environment variables since Lakeflow Jobs may set different ones.
    Can be explicitly forced via --force_jobs flag.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        True if running in Jobs environment, False otherwise
    """
    if getattr(args, "force_jobs", False):
        return True
    
    # Common signals for Databricks Jobs context
    env = os.environ
    job_flags = [
        "DATABRICKS_JOB_RUN_ID",      # Standard Jobs
        "DB_IS_JOB_CONTEXT",           # Sometimes "true" in jobs
        "DATABRICKS_LAKEFLOW_RUN_ID",  # Lakeflow Jobs
        "LAKEFLOW_RUN_ID",             # Alternative Lakeflow
        "DB_IS_PIPELINE_JOB",          # Pipeline jobs
    ]
    
    return any(flag in env for flag in job_flags)


def initialize_model_and_data(task, config):
    """Initialize model and data module based on task type."""
    print(f"\n🔧 Initializing {task} model and data module...")
    
    if task == "classification":
        from tasks.classification.model import ClassificationModel
        from tasks.classification.data import ClassificationDataModule
        from tasks.classification.adapters import get_input_adapter
        
        model = ClassificationModel(config["model"])
        data_module = ClassificationDataModule(config["data"])
        
        image_size = config["data"].get("image_size", 224)
        adapter = get_input_adapter(config["model"]["model_name"], image_size=image_size)
        
    elif task == "detection":
        from tasks.detection.model import DetectionModel
        from tasks.detection.data import DetectionDataModule
        from tasks.detection.adapters import get_input_adapter
        
        model_config = config["model"].copy()
        model_config["num_workers"] = config["data"]["num_workers"]
        model = DetectionModel(model_config)
        data_module = DetectionDataModule(config["data"])
        
        image_size = config["data"].get("image_size", 800)
        if isinstance(image_size, list):
            image_size = image_size[0]
        adapter = get_input_adapter(config["model"]["model_name"], image_size=image_size)
        
    elif task == "semantic_segmentation":
        from tasks.semantic_segmentation.model import SemanticSegmentationModel
        from tasks.semantic_segmentation.data import SemanticSegmentationDataModule
        from tasks.semantic_segmentation.adapters import get_input_adapter
        
        model = SemanticSegmentationModel(config["model"])
        data_module = SemanticSegmentationDataModule(config["data"])
        
        image_size = config["data"].get("image_size", 512)
        adapter = get_input_adapter(config["model"]["model_name"], image_size=image_size)
        
    elif task == "instance_segmentation":
        from tasks.instance_segmentation.model import InstanceSegmentationModel
        from tasks.instance_segmentation.data import InstanceSegmentationDataModule
        from tasks.instance_segmentation.adapters import get_input_adapter
        
        model = InstanceSegmentationModel(config["model"])
        data_module = InstanceSegmentationDataModule(config["data"])
        
        image_size = config["data"].get("image_size", 512)
        adapter = get_input_adapter(config["model"]["model_name"], image_size=image_size)
        
    elif task == "universal_segmentation":
        from tasks.universal_segmentation.model import UniversalSegmentationModel
        from tasks.universal_segmentation.data import UniversalSegmentationDataModule
        from tasks.universal_segmentation.adapters import get_input_adapter
        
        model = UniversalSegmentationModel(config["model"])
        data_module = UniversalSegmentationDataModule(config["data"])
        
        image_size = config["data"].get("image_size", 512)
        adapter = get_input_adapter(config["model"]["model_name"], image_size=image_size)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Assign adapter and setup data module
    data_module.adapter = adapter
    data_module.setup('fit')
    
    print(f"✅ Model and data initialized!")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   Train samples: {len(data_module.train_dataset):,}")
    print(f"   Val samples: {len(data_module.val_dataset):,}")
    
    return model, data_module


def main():
    """Main training execution."""
    # Suppress meta tensor warnings (harmless PyTorch warnings in Jobs environment)
    warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
    
    print("=" * 80)
    print("🚀 Databricks CV Model Training - Jobs Compute")
    print("=" * 80)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train CV models on Databricks Jobs compute")
    parser.add_argument("--config_path", required=True, help="Path to YAML configuration file")
    parser.add_argument("--src_path", required=True, help="Path to src directory")
    parser.add_argument(
        "--force_strategy",
        choices=["auto", "ddp", "ddp_notebook"],
        default="auto",
        help="Force Lightning strategy (use 'ddp' on Jobs, 'auto' for notebooks)",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Number of devices per node (int) or 'auto'",
    )
    parser.add_argument(
        "--force_jobs",
        action="store_true",
        help="Treat environment as non-interactive Jobs, even if IPython kernel is present",
    )
    args = parser.parse_args()
    
    print(f"📦 Using src path: {args.src_path}")
    print(f"📋 Config path: {args.config_path}")
    
    # Add src to Python path
    if args.src_path not in sys.path:
        sys.path.insert(0, args.src_path)
    
    # Import after adding to path
    from config import load_config
    from training.trainer import UnifiedTrainer
    from utils.logging import create_databricks_logger
    
    # Get config path from arguments
    config_path = args.config_path
    
    print(f"\n📋 Loading configuration from: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Determine environment and strategy
    # CRITICAL: Do NOT probe torch.cuda.* unless we're certain we're in Jobs mode
    # CUDA probing pre-initializes CUDA and prevents multi-GPU in notebook-like environments
    is_jobs = _is_jobs_env(args)
    force_strategy = args.force_strategy
    force_devices = args.devices
    
    if is_jobs:
        # Safe to probe CUDA in non-interactive Jobs environment
        if force_devices == "auto":
            visible_gpus = torch.cuda.device_count()
        else:
            visible_gpus = int(force_devices)
        
        distributed = visible_gpus >= 2 and (force_strategy in ("auto", "ddp"))
        devices = visible_gpus if visible_gpus > 0 else "auto"
        strategy = "ddp" if distributed else "auto"
        
        print(f"\n🧭 Environment: Jobs (non-interactive)")
        print(f"   Strategy: {strategy}")
        print(f"   Devices: {devices}")
        print(f"   Distributed: {distributed}")
    else:
        # Interactive/notebook mode: DO NOT probe CUDA here
        # Let Lightning auto-select safely
        distributed = False  # Default to single-GPU for safety
        devices = force_devices  # Usually "auto"
        strategy = "auto" if force_strategy == "auto" else force_strategy
        
        print(f"\n🧭 Environment: Interactive-like (notebook/REPL)")
        print(f"   Strategy: {strategy} (defaulting to single-GPU)")
        print(f"   Devices: {devices}")
        print(f"   Distributed: {distributed}")
        print(f"⚠️  Multi-GPU DDP not recommended in interactive environments")
    
    config["training"]["distributed"] = distributed
    config["training"]["use_gpu"] = (devices != 0 and devices != "cpu")
    
    # Print configuration summary
    print(f"\n📊 Training Configuration:")
    print(f"   Task: {config['model']['task_type']}")
    print(f"   Model: {config['model']['model_name']}")
    print(f"   Max epochs: {config['training']['max_epochs']}")
    print(f"   Batch size: {config['data']['batch_size']}")
    print(f"   Distributed: {config['training']['distributed']}")
    print(f"   Checkpoint dir: {config['training']['checkpoint_dir']}")
    
    # Initialize model and data
    task = config['model']['task_type']
    model, data_module = initialize_model_and_data(task, config)
    
    # Setup trainer configuration
    trainer_config = {
        'task': config['model']['task_type'],
        'model_name': config['model']['model_name'],
        'max_epochs': config['training']['max_epochs'],
        'log_every_n_steps': config['training']['log_every_n_steps'],
        'monitor_metric': config['training']['monitor_metric'],
        'monitor_mode': config['training']['monitor_mode'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'checkpoint_dir': config['training']['checkpoint_dir'],
        'volume_checkpoint_dir': config['training'].get('volume_checkpoint_dir'),
        'save_top_k': config['training'].get('save_top_k', 3),
        'distributed': config['training']['distributed'],
        'use_ray': False,
        'num_workers': 1,
        'use_gpu': config['training']['use_gpu'],
        # NEW: Explicit strategy/device preferences from job script
        # These take precedence in UnifiedTrainer to avoid re-inference
        'preferred_strategy': strategy,
        'preferred_devices': devices,
    }
    
    # Create MLflow logger with experiment name from config
    # Experiment name must be an absolute workspace path on Databricks
    # Example: "/Users/user@email.com/my_experiment" or "/Shared/experiment_name"
    experiment_name = config['training'].get('experiment_name')
    if not experiment_name:
        raise ValueError(
            "Missing 'experiment_name' in config['training'].\n"
            "Experiment name must be an absolute workspace path.\n"
            "Examples:\n"
            "  /Users/your.email@databricks.com/cv_detection_experiment\n"
            "  /Shared/cv_experiments/detection\n"
            "  /Workspace/Users/your.email@databricks.com/experiments/detection"
        )
    
    print(f"📊 MLflow Experiment: {experiment_name}")
    logger = create_databricks_logger(experiment_name=experiment_name)
    
    # Create trainer
    trainer = UnifiedTrainer(
        config=trainer_config,
        model=model,
        data_module=data_module,
        logger=logger
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("🎯 Starting Training")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        metrics = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("✅ Training Completed Successfully!")
        print("=" * 80)
        print(f"   Duration: {duration/60:.2f} minutes")
        print(f"   Final metrics:")
        for key, value in metrics.items():
            print(f"     {key}: {value:.4f}")
        
        print(f"\n📁 Checkpoints saved to: {config['training']['checkpoint_dir']}")
        print(f"📊 View metrics in MLflow: Experiment '{experiment_name}'")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Training Failed!")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()


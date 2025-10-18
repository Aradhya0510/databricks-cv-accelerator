"""
Configuration management for the Databricks Computer Vision Pipeline with Serverless GPU support.

This module provides dataclasses and utilities for managing configuration
across different computer vision tasks and training scenarios with Serverless GPU compute.
"""

import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

def tuple_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> tuple:
    """Custom constructor for YAML tuples."""
    value = loader.construct_sequence(node)
    return tuple(value)

# Register the tuple constructor
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, yaml.SafeLoader)

@dataclass
class ModelConfig:
    """Model configuration."""
    # Model architecture
    model_name: str
    num_classes: int
    pretrained: bool = True
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    scheduler_params: Optional[Dict[str, Any]] = None
    epochs: int = 100
    
    # Task-specific settings
    task_type: str = "detection"
    class_names: Optional[List[str]] = None
    segmentation_type: Optional[str] = None
    
    # Detection-specific settings
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_detections: int = 100
    
    # Classification-specific settings
    dropout: float = 0.2
    mixup_alpha: float = 0.2
    
    # Segmentation-specific settings
    num_queries: int = 100
    aux_loss: bool = True
    
    # Image processing
    image_size: int = 800
    normalize: bool = True
    
    # Additional model parameters
    num_workers: int = 4

@dataclass
class DataConfig:
    """Data configuration."""
    # Data paths
    train_data_path: str = ""
    train_annotation_file: str = ""
    val_data_path: str = ""
    val_annotation_file: str = ""
    test_data_path: str = ""
    test_annotation_file: str = ""
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Data augmentation
    image_size: int = 800
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    
    # Data preprocessing
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Dataset-specific settings
    cache_images: bool = False
    mosaic: bool = False
    mixup: bool = False

@dataclass
class TrainingConfig:
    """Training configuration with Serverless GPU support."""
    # Basic training parameters
    max_epochs: int = 100
    
    # Learning rate and optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    scheduler_params: Optional[Dict[str, Any]] = None
    
    # Early stopping
    early_stopping_patience: int = 10
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    
    # Checkpointing
    checkpoint_dir: str = "/Volumes/<catalog>/<schema>/<volume>/<path>/checkpoints"
    volume_checkpoint_dir: Optional[str] = None  # Persistent storage for checkpoints
    save_top_k: int = 3
    
    # Logging
    log_every_n_steps: int = 50
    
    # Distributed training
    distributed: bool = False
    use_ray: bool = False  # Whether to use Ray (multi-node) or Databricks DDP (single-node)
    num_workers: int = 1  # Number of distributed training workers (not DataLoader workers)
    use_serverless_gpu: bool = False  # Whether to use Serverless GPU compute
    use_gpu: bool = True
    resources_per_worker: Dict[str, int] = field(default_factory=lambda: {
        "CPU": 4,
        "GPU": 1
    })
    master_port: Optional[int] = None  # Port for DDP communication
    
    # Serverless GPU specific settings
    serverless_gpu_type: str = "A10"  # A10 or H100
    serverless_gpu_count: int = 4  # Number of GPUs to use
    
    # Strategy overrides (typically set by job scripts)
    preferred_strategy: Optional[str] = None  # e.g., "ddp", "auto", "ddp_notebook"
    preferred_devices: Optional[Union[str, int]] = None  # e.g., "auto", 4, 1
    
    # Additional training parameters
    precision: str = "16-mixed"  # 16-mixed, 32, 16
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

@dataclass
class MLflowConfig:
    """MLflow configuration."""
    experiment_name: str = "cv_training"
    run_name: str = "default_run"
    tags: Dict[str, str] = field(default_factory=dict)
    log_model: bool = True
    log_artifacts: bool = True
    artifact_path: str = "model"

@dataclass
class OutputConfig:
    """Output configuration."""
    results_dir: str = "/Volumes/<catalog>/<schema>/<volume>/<path>/results"
    save_predictions: bool = True
    save_visualizations: bool = True
    save_metrics: bool = True
    export_format: str = "json"  # json, csv, parquet

@dataclass
class UnifiedConfig:
    """Unified configuration combining all components."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    mlflow: MLflowConfig
    output: OutputConfig

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with Serverless GPU support.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure all required sections exist
    required_sections = ['model', 'data', 'training', 'mlflow', 'output']
    for section in required_sections:
        if section not in config:
            config[section] = {}
    
    # Convert numeric fields to appropriate types
    if 'model' in config:
        # Integer fields
        for key in ['num_classes', 'epochs', 'max_detections', 'num_queries', 'image_size', 'num_workers']:
            if key in config['model'] and not isinstance(config['model'][key], list):
                config['model'][key] = int(config['model'][key])
        
        # Float fields
        for key in ['learning_rate', 'weight_decay', 'confidence_threshold', 'iou_threshold', 'dropout', 'mixup_alpha', 'rotation', 'brightness', 'contrast', 'saturation', 'hue']:
            if key in config['model']:
                config['model'][key] = float(config['model'][key])
        
        # Boolean fields
        for key in ['pretrained', 'aux_loss', 'normalize']:
            if key in config['model']:
                config['model'][key] = bool(config['model'][key])
    
    if 'data' in config:
        # Integer fields
        for key in ['batch_size', 'num_workers']:
            if key in config['data'] and not isinstance(config['data'][key], list):
                config['data'][key] = int(config['data'][key])
        
        # Float fields
        for key in ['rotation', 'brightness', 'contrast', 'saturation', 'hue']:
            if key in config['data'] and not isinstance(config['data'][key], list):
                config['data'][key] = float(config['data'][key])
        
        # Boolean fields
        for key in ['pin_memory', 'shuffle', 'horizontal_flip', 'vertical_flip', 'normalize', 'cache_images', 'mosaic', 'mixup']:
            if key in config['data']:
                config['data'][key] = bool(config['data'][key])
    
    if 'training' in config:
        # Integer fields
        for key in ['max_epochs', 'early_stopping_patience', 'save_top_k', 'log_every_n_steps', 'serverless_gpu_count', 'master_port', 'accumulate_grad_batches']:
            if key in config['training'] and not isinstance(config['training'][key], list):
                config['training'][key] = int(config['training'][key])
        
        # Float fields
        for key in ['learning_rate', 'weight_decay', 'gradient_clip_val']:
            if key in config['training'] and not isinstance(config['training'][key], list):
                config['training'][key] = float(config['training'][key])
        
        # Boolean fields
        for key in ['distributed', 'use_ray', 'use_serverless_gpu', 'use_gpu']:
            if key in config['training']:
                config['training'][key] = bool(config['training'][key])
    
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def get_default_config(task_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific task type with Serverless GPU support.
    
    Args:
        task_type: Type of task ('detection', 'classification', 'segmentation')
        
    Returns:
        Default configuration dictionary
    """
    if task_type == "detection":
        return {
            'model': {
                'model_name': 'facebook/detr-resnet-50',
                'task_type': 'detection',
                'num_classes': 91,
                'pretrained': True,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'epochs': 100,
                'confidence_threshold': 0.5,
                'iou_threshold': 0.5,
                'max_detections': 100,
                'image_size': 800,
                'normalize': True,
                'num_workers': 4
            },
            'data': {
                'batch_size': 2,
                'num_workers': 4,
                'pin_memory': True,
                'shuffle': True,
                'image_size': 800,
                'horizontal_flip': True,
                'vertical_flip': False,
                'rotation': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'saturation': 0.0,
                'hue': 0.0,
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'cache_images': False,
                'mosaic': False,
                'mixup': False
            },
            'training': {
                'max_epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'early_stopping_patience': 10,
                'monitor_metric': 'val_map',
                'monitor_mode': 'max',
                'checkpoint_dir': '/Volumes/<catalog>/<schema>/<volume>/<path>/checkpoints',
                'save_top_k': 3,
                'log_every_n_steps': 50,
                'distributed': True,
                'use_ray': False,
                'use_serverless_gpu': True,
                'use_gpu': True,
                'resources_per_worker': {
                    'CPU': 4,
                    'GPU': 1
                },
                'serverless_gpu_type': 'A10',
                'serverless_gpu_count': 4,
                'precision': '16-mixed',
                'gradient_clip_val': 1.0,
                'accumulate_grad_batches': 1
            },
            'mlflow': {
                'experiment_name': 'cv_detection_training',
                'run_name': 'detection_run',
                'tags': {'task': 'detection'},
                'log_model': True,
                'log_artifacts': True,
                'artifact_path': 'model'
            },
            'output': {
                'results_dir': '/Volumes/<catalog>/<schema>/<volume>/<path>/results',
                'save_predictions': True,
                'save_visualizations': True,
                'save_metrics': True,
                'export_format': 'json'
            }
        }
    
    elif task_type == "classification":
        return {
            'model': {
                'model_name': 'resnet50',
                'task_type': 'classification',
                'num_classes': 1000,
                'pretrained': True,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'epochs': 100,
                'dropout': 0.2,
                'mixup_alpha': 0.2,
                'image_size': 224,
                'normalize': True,
                'num_workers': 4
            },
            'data': {
                'batch_size': 32,
                'num_workers': 4,
                'pin_memory': True,
                'shuffle': True,
                'image_size': 224,
                'horizontal_flip': True,
                'vertical_flip': False,
                'rotation': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'saturation': 0.0,
                'hue': 0.0,
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'cache_images': False,
                'mosaic': False,
                'mixup': False
            },
            'training': {
                'max_epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'early_stopping_patience': 10,
                'monitor_metric': 'val_accuracy',
                'monitor_mode': 'max',
                'checkpoint_dir': '/Volumes/<catalog>/<schema>/<volume>/<path>/checkpoints',
                'save_top_k': 3,
                'log_every_n_steps': 50,
                'distributed': True,
                'use_ray': False,
                'use_serverless_gpu': True,
                'use_gpu': True,
                'resources_per_worker': {
                    'CPU': 4,
                    'GPU': 1
                },
                'serverless_gpu_type': 'A10',
                'serverless_gpu_count': 4,
                'precision': '16-mixed',
                'gradient_clip_val': 1.0,
                'accumulate_grad_batches': 1
            },
            'mlflow': {
                'experiment_name': 'cv_classification_training',
                'run_name': 'classification_run',
                'tags': {'task': 'classification'},
                'log_model': True,
                'log_artifacts': True,
                'artifact_path': 'model'
            },
            'output': {
                'results_dir': '/Volumes/<catalog>/<schema>/<volume>/<path>/results',
                'save_predictions': True,
                'save_visualizations': True,
                'save_metrics': True,
                'export_format': 'json'
            }
        }
    
    elif task_type == "segmentation":
        return {
            'model': {
                'model_name': 'facebook/mask2former-swin-small-coco-instance',
                'task_type': 'segmentation',
                'num_classes': 91,
                'pretrained': True,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'epochs': 100,
                'num_queries': 100,
                'aux_loss': True,
                'image_size': 800,
                'normalize': True,
                'num_workers': 4
            },
            'data': {
                'batch_size': 2,
                'num_workers': 4,
                'pin_memory': True,
                'shuffle': True,
                'image_size': 800,
                'horizontal_flip': True,
                'vertical_flip': False,
                'rotation': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'saturation': 0.0,
                'hue': 0.0,
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'cache_images': False,
                'mosaic': False,
                'mixup': False
            },
            'training': {
                'max_epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'early_stopping_patience': 10,
                'monitor_metric': 'val_iou',
                'monitor_mode': 'max',
                'checkpoint_dir': '/Volumes/<catalog>/<schema>/<volume>/<path>/checkpoints',
                'save_top_k': 3,
                'log_every_n_steps': 50,
                'distributed': True,
                'use_ray': False,
                'use_serverless_gpu': True,
                'use_gpu': True,
                'resources_per_worker': {
                    'CPU': 4,
                    'GPU': 1
                },
                'serverless_gpu_type': 'A10',
                'serverless_gpu_count': 4,
                'precision': '16-mixed',
                'gradient_clip_val': 1.0,
                'accumulate_grad_batches': 1
            },
            'mlflow': {
                'experiment_name': 'cv_segmentation_training',
                'run_name': 'segmentation_run',
                'tags': {'task': 'segmentation'},
                'log_model': True,
                'log_artifacts': True,
                'artifact_path': 'model'
            },
            'output': {
                'results_dir': '/Volumes/<catalog>/<schema>/<volume>/<path>/results',
                'save_predictions': True,
                'save_visualizations': True,
                'save_metrics': True,
                'export_format': 'json'
            }
        }
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for Serverless GPU compatibility.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    # Check required sections
    required_sections = ['model', 'data', 'training', 'mlflow', 'output']
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required section: {section}")
            return False
    
    # Check serverless GPU configuration
    if config['training'].get('use_serverless_gpu', False):
        gpu_type = config['training'].get('serverless_gpu_type', 'A10')
        gpu_count = config['training'].get('serverless_gpu_count', 4)
        
        if gpu_type not in ['A10', 'H100']:
            print(f"❌ Invalid serverless_gpu_type: {gpu_type}. Must be 'A10' or 'H100'")
            return False
        
        if gpu_type == 'H100' and gpu_count > 1:
            print("⚠️  H100 GPUs only support single-node workflows")
            config['training']['serverless_gpu_count'] = 1
        
        if not config['training'].get('distributed', False):
            print("⚠️  Serverless GPU requires distributed training. Setting distributed=True")
            config['training']['distributed'] = True
        
        if config['training'].get('use_ray', False):
            print("⚠️  Cannot use both Ray and Serverless GPU. Setting use_ray=False")
            config['training']['use_ray'] = False
    
    print("✅ Configuration validation passed")
    return True

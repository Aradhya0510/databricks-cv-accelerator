"""
Configuration management for the Databricks Computer Vision Pipeline.

This module provides dataclasses and utilities for managing configuration
across different computer vision tasks and training scenarios.
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
    aux_loss_weight: float = 0.4
    mask_threshold: float = 0.5

@dataclass
class DataConfig:
    """Data configuration."""
    # Dataset paths
    train_data_path: str
    val_data_path: str
    train_annotation_file: Optional[str] = None
    val_annotation_file: Optional[str] = None
    test_data_path: Optional[str] = None
    test_annotation_file: Optional[str] = None
    
    # Data loading parameters
    batch_size: int = 16
    num_workers: int = 4
    model_name: str = ""  # For adapter initialization
    
    # Image processing
    image_size: Union[List[int], Tuple[int, int]] = (512, 512)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Augmentation
    augment: bool = True
    augmentations: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation": 15,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1
        },
        "random_crop": True,
        "random_resize": [0.8, 1.2]
    })

@dataclass
class TrainingConfig:
    """Training configuration."""
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
    save_top_k: int = 3
    
    # Logging
    log_every_n_steps: int = 50
    
    # Distributed training
    distributed: bool = False
    use_gpu: bool = True
    resources_per_worker: Dict[str, int] = field(default_factory=lambda: {
        "CPU": 4,
        "GPU": 1
    })

@dataclass
class MLflowConfig:
    """MLflow configuration."""
    experiment_name: str = "cv_training"
    run_name: str = "default_run"
    log_model: bool = True
    tags: Dict[str, str] = field(default_factory=lambda: {
        "framework": "lightning",
        "model": "default",
        "dataset": "default"
    })

@dataclass
class OutputConfig:
    """Output configuration."""
    results_dir: str = "/Volumes/<catalog>/<schema>/<volume>/<path>/results"
    save_predictions: bool = True
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        "save_images": True,
        "confidence_threshold": 0.5,
        "max_boxes": 20,
        "max_images": 10
    })

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with proper type conversion."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert types at the source
    if 'model' in config:
        # Integer fields
        for key in ['epochs', 'num_classes', 'max_detections']:
            if key in config['model']:
                config['model'][key] = int(config['model'][key])
        
        # Float fields
        for key in ['learning_rate', 'weight_decay', 'confidence_threshold', 'iou_threshold']:
            if key in config['model']:
                config['model'][key] = float(config['model'][key])
        
        # Boolean fields
        for key in ['pretrained']:
            if key in config['model']:
                config['model'][key] = bool(config['model'][key])
    
    if 'data' in config:
        # Integer fields
        for key in ['batch_size', 'num_workers']:
            if key in config['data']:
                config['data'][key] = int(config['data'][key])
    
    if 'training' in config:
        # Integer fields
        for key in ['max_epochs', 'early_stopping_patience', 'save_top_k', 'log_every_n_steps']:
            if key in config['training']:
                config['training'][key] = int(config['training'][key])
        
        # Float fields
        for key in ['learning_rate', 'weight_decay']:
            if key in config['training']:
                config['training'][key] = float(config['training'][key])
        
        # Boolean fields
        for key in ['distributed', 'use_gpu']:
            if key in config['training']:
                config['training'][key] = bool(config['training'][key])
    
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_default_config(task: str) -> Dict[str, Any]:
    """Get default configuration for the specified task."""
    # Base configuration
    config = {
        'model': asdict(ModelConfig(
            model_name="nvidia/mit-b0",
            num_classes=19,
            task_type=task
        )),
        'data': asdict(DataConfig(
            train_data_path="/Volumes/<catalog>/<schema>/<volume>/<path>/data/train",
            val_data_path="/Volumes/<catalog>/<schema>/<volume>/<path>/data/val"
        )),
        'training': asdict(TrainingConfig()),
        'mlflow': asdict(MLflowConfig()),
        'output': asdict(OutputConfig())
    }
    
    # Task-specific configurations
    if task == "detection":
        config['model'].update({
            'model_name': "facebook/detr-resnet-50",
            'num_classes': 80,
            'confidence_threshold': 0.5,
            'iou_threshold': 0.5,
            'max_detections': 100
        })
        config['data'].update({
            'model_name': "facebook/detr-resnet-50",
            'image_size': [800, 800],
            'batch_size': 16
        })
        config['training'].update({
            'max_epochs': 50,
            'monitor_metric': "val_map",
            'monitor_mode': "max"
        })
        config['mlflow'].update({
            'experiment_name': "detection_training",
            'run_name': "detr_resnet50",
            'tags': {
                "framework": "lightning",
                "model": "detr",
                "dataset": "coco"
            }
        })
        config['output'].update({
            'results_dir': "/Volumes/<catalog>/<schema>/<volume>/<path>/results/detection"
        })
        
    elif task == "classification":
        config['model'].update({
            'model_name': "microsoft/resnet-50",
            'num_classes': 1000,
            'dropout': 0.2,
            'mixup_alpha': 0.2
        })
        config['data'].update({
            'model_name': "microsoft/resnet-50",
            'image_size': [224, 224],
            'batch_size': 32
        })
        config['training'].update({
            'max_epochs': 100,
            'monitor_metric': "val_loss",
            'monitor_mode': "min"
        })
        config['mlflow'].update({
            'experiment_name': "classification_training",
            'run_name': "resnet50",
            'tags': {
                "framework": "lightning",
                "model": "resnet",
                "dataset": "imagenet"
            }
        })
        config['output'].update({
            'results_dir': "/Volumes/<catalog>/<schema>/<volume>/<path>/results/classification"
        })
        
    elif task == "semantic_segmentation":
        config['model'].update({
            'model_name': "nvidia/mit-b0",
            'num_classes': 19,
            'segmentation_type': "semantic",
            'aux_loss_weight': 0.4,
            'mask_threshold': 0.5
        })
        config['data'].update({
            'model_name': "nvidia/mit-b0",
            'image_size': [512, 512],
            'batch_size': 8
        })
        config['training'].update({
            'max_epochs': 200,
            'monitor_metric': "val_miou",
            'monitor_mode': "max"
        })
        config['mlflow'].update({
            'experiment_name': "semantic_segmentation_training",
            'run_name': "segformer_mit_b0",
            'tags': {
                "framework": "lightning",
                "model": "segformer",
                "dataset": "cityscapes"
            }
        })
        config['output'].update({
            'results_dir': "/Volumes/<catalog>/<schema>/<volume>/<path>/results/semantic_segmentation"
        })
        
    elif task == "universal_segmentation":
        config['model'].update({
            'model_name': "facebook/mask2former-swin-base-coco-panoptic",
            'num_classes': 133,
            'segmentation_type': "universal",
            'aux_loss_weight': 0.4,
            'mask_threshold': 0.5
        })
        config['data'].update({
            'model_name': "facebook/mask2former-swin-base-coco-panoptic",
            'image_size': [512, 512],
            'batch_size': 4
        })
        config['training'].update({
            'max_epochs': 200,
            'monitor_metric': "val_miou",
            'monitor_mode': "max"
        })
        config['mlflow'].update({
            'experiment_name': "universal_segmentation_training",
            'run_name': "mask2former_swin_base",
            'tags': {
                "framework": "lightning",
                "model": "mask2former",
                "dataset": "coco"
            }
        })
        config['output'].update({
            'results_dir': "/Volumes/<catalog>/<schema>/<volume>/<path>/results/universal_segmentation"
        })
    
    return config 
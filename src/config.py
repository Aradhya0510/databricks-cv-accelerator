import os
import yaml
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Union, Dict, Any, Tuple

def tuple_constructor(loader, node):
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
class TrainingConfig:
    """Training configuration."""
    # Batch and optimization settings
    batch_size: int = 32
    num_workers: int = 4
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    accumulate_grad_batches: int = 1
    
    # Early stopping and checkpointing
    early_stopping_patience: int = 10
    checkpoint_dir: str = "/Volumes/main/cv_ref/checkpoints"
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"
    
    # Logging settings
    log_every_n_steps: int = 50
    log_metrics: bool = True
    log_artifacts: bool = True
    
    # Ray distributed training settings
    distributed: bool = False
    num_workers: int = 2
    use_gpu: bool = True
    resources_per_worker: Dict[str, int] = field(default_factory=lambda: {
        "CPU": 2,
        "GPU": 1
    })

@dataclass
class DataConfig:
    """Data configuration."""
    # Dataset paths
    train_path: str
    val_path: str
    test_path: Optional[str] = None
    
    # Data processing
    image_size: Union[List[int], Tuple[int, int]] = (512, 512)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data loading
    augment: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation settings
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

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
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
        'training': asdict(TrainingConfig()),
        'data': {
            'train': {
                'root_dir': "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train",
                'annotation_file': "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train/annotations.json"
            },
            'val': {
                'root_dir': "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val",
                'annotation_file': "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val/annotations.json"
            },
            'test': {
                'root_dir': "/Volumes/<catalog>/<schema>/<volume>/<path>/data/test",
                'annotation_file': "/Volumes/<catalog>/<schema>/<volume>/<path>/data/test/annotations.json"
            },
            'image_size': (512, 512),
            'normalize_mean': (0.485, 0.456, 0.406),
            'normalize_std': (0.229, 0.224, 0.225),
            'augment': True,
            'num_workers': 4,
            'pin_memory': True,
            'augmentations': {
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
            }
        },
        'output': {
            'results_dir': "/Volumes/<catalog>/<schema>/<volume>/<path>/results",
            'save_predictions': True,
            'visualization': {
                'save_images': True,
                'max_images': 10
            }
        }
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
    elif task == "classification":
        config['model'].update({
            'model_name': "microsoft/resnet-50",
            'num_classes': 1000,
            'dropout': 0.2,
            'mixup_alpha': 0.2
        })
    elif task == "segmentation":
        config['model'].update({
            'model_name': "nvidia/mit-b0",
            'num_classes': 19,
            'segmentation_type': "semantic",
            'aux_loss_weight': 0.4,
            'mask_threshold': 0.5
        })
    
    return config 
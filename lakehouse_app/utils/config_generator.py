"""
Configuration Generator for CV Training Pipeline
Generates YAML configurations from UI inputs
"""

from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path
from datetime import datetime


class ConfigGenerator:
    """Generate YAML configurations from user inputs."""
    
    # Model mappings for each task
    TASK_MODELS = {
        "classification": [
            {"name": "microsoft/resnet-50", "display": "ResNet-50 (Microsoft)", "size": "Medium"},
            {"name": "google/vit-base-patch16-224", "display": "ViT-Base (Google)", "size": "Large"},
            {"name": "facebook/convnext-base-224", "display": "ConvNeXT-Base (Facebook)", "size": "Large"},
            {"name": "microsoft/swin-base-patch4-window7-224", "display": "Swin-Base (Microsoft)", "size": "Large"},
        ],
        "detection": [
            {"name": "facebook/detr-resnet-50", "display": "DETR ResNet-50 (Facebook)", "size": "Medium"},
            {"name": "facebook/detr-resnet-101", "display": "DETR ResNet-101 (Facebook)", "size": "Large"},
            {"name": "hustvl/yolos-tiny", "display": "YOLOS-Tiny (HUST-VL)", "size": "Small"},
            {"name": "hustvl/yolos-small", "display": "YOLOS-Small (HUST-VL)", "size": "Medium"},
        ],
        "semantic_segmentation": [
            {"name": "nvidia/segformer-b0-finetuned-ade-512-512", "display": "SegFormer-B0 (NVIDIA)", "size": "Small"},
            {"name": "nvidia/segformer-b1-finetuned-ade-512-512", "display": "SegFormer-B1 (NVIDIA)", "size": "Medium"},
            {"name": "nvidia/mit-b0", "display": "MiT-B0 (NVIDIA)", "size": "Small"},
            {"name": "facebook/mask2former-swin-base-ade-semantic", "display": "Mask2Former-Swin (Facebook)", "size": "Large"},
        ],
        "instance_segmentation": [
            {"name": "facebook/mask2former-swin-base-coco-instance", "display": "Mask2Former-Swin-Base (Facebook)", "size": "Large"},
            {"name": "facebook/mask2former-swin-small-coco-instance", "display": "Mask2Former-Swin-Small (Facebook)", "size": "Medium"},
        ],
        "universal_segmentation": [
            {"name": "facebook/mask2former-swin-base-coco-panoptic", "display": "Mask2Former-Swin-Base Panoptic (Facebook)", "size": "Large"},
            {"name": "facebook/mask2former-swin-small-coco-panoptic", "display": "Mask2Former-Swin-Small Panoptic (Facebook)", "size": "Medium"},
        ],
    }
    
    # Default image sizes for each model family
    DEFAULT_IMAGE_SIZES = {
        "resnet": 224,
        "vit": 224,
        "convnext": 224,
        "swin": 224,
        "detr": 800,
        "yolos": 512,
        "segformer": 512,
        "mit": 512,
        "mask2former": 512,
    }
    
    # Default hyperparameters by model size
    DEFAULT_HYPERPARAMS = {
        "Small": {"batch_size": 32, "learning_rate": 2e-4, "epochs": 100},
        "Medium": {"batch_size": 16, "learning_rate": 1e-4, "epochs": 100},
        "Large": {"batch_size": 8, "learning_rate": 5e-5, "epochs": 150},
    }
    
    @classmethod
    def get_models_for_task(cls, task: str) -> List[Dict[str, str]]:
        """Get available models for a specific task."""
        return cls.TASK_MODELS.get(task, [])
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, str]]:
        """Get information about a specific model."""
        for task_models in cls.TASK_MODELS.values():
            for model in task_models:
                if model["name"] == model_name:
                    return model
        return None
    
    @classmethod
    def get_default_image_size(cls, model_name: str) -> int:
        """Get default image size for a model."""
        for model_family, size in cls.DEFAULT_IMAGE_SIZES.items():
            if model_family in model_name.lower():
                return size
        return 512  # Default fallback
    
    @classmethod
    def get_default_batch_size(cls, model_name: str) -> int:
        """Get recommended batch size based on model."""
        model_info = cls.get_model_info(model_name)
        if model_info:
            size = model_info.get("size", "Medium")
            return cls.DEFAULT_HYPERPARAMS[size]["batch_size"]
        return 16
    
    @classmethod
    def generate_config(
        cls,
        task: str,
        model_name: str,
        data_config: Dict[str, Any],
        training_config: Dict[str, Any],
        mlflow_config: Dict[str, Any],
        output_config: Dict[str, Any],
        model_specific_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete configuration dictionary.
        
        Args:
            task: Task type (classification, detection, etc.)
            model_name: HuggingFace model name
            data_config: Data configuration parameters
            training_config: Training configuration parameters
            mlflow_config: MLflow configuration parameters
            output_config: Output configuration parameters
            model_specific_config: Task-specific model parameters
            
        Returns:
            Complete configuration dictionary
        """
        # Get image size
        image_size = data_config.get("image_size", cls.get_default_image_size(model_name))
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        
        # Base model config
        model_config = {
            "model_name": model_name,
            "task_type": task,
            "num_classes": data_config.get("num_classes", 80),
            "pretrained": True,
            "learning_rate": training_config.get("learning_rate", 1e-4),
            "weight_decay": training_config.get("weight_decay", 1e-4),
            "scheduler": training_config.get("scheduler", "cosine"),
            "scheduler_params": training_config.get("scheduler_params", {
                "T_max": training_config.get("epochs", 100),
                "eta_min": 1e-6
            }),
            "epochs": training_config.get("epochs", 100),
        }
        
        # Add task-specific model parameters
        if task == "detection":
            model_config.update({
                "confidence_threshold": model_specific_config.get("confidence_threshold", 0.5) if model_specific_config else 0.5,
                "iou_threshold": model_specific_config.get("iou_threshold", 0.5) if model_specific_config else 0.5,
                "max_detections": model_specific_config.get("max_detections", 100) if model_specific_config else 100,
                "image_size": image_size[0] if isinstance(image_size, list) else image_size,
            })
        elif task == "classification":
            model_config.update({
                "dropout": model_specific_config.get("dropout", 0.2) if model_specific_config else 0.2,
                "mixup_alpha": model_specific_config.get("mixup_alpha", 0.2) if model_specific_config else 0.2,
            })
        elif task in ["semantic_segmentation", "instance_segmentation", "universal_segmentation"]:
            model_config.update({
                "aux_loss_weight": model_specific_config.get("aux_loss_weight", 0.4) if model_specific_config else 0.4,
                "mask_threshold": model_specific_config.get("mask_threshold", 0.5) if model_specific_config else 0.5,
            })
        
        # Data configuration
        data_dict = {
            "batch_size": data_config.get("batch_size", 16),
            "num_workers": data_config.get("num_workers", 4),
            "model_name": model_name,
            "image_size": image_size,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "augment": data_config.get("augment", True),
        }
        
        # Add data paths
        if task in ["detection", "instance_segmentation", "universal_segmentation"]:
            data_dict.update({
                "train_data_path": data_config.get("train_data_path", ""),
                "train_annotation_file": data_config.get("train_annotation_file", ""),
                "val_data_path": data_config.get("val_data_path", ""),
                "val_annotation_file": data_config.get("val_annotation_file", ""),
            })
            if data_config.get("test_data_path"):
                data_dict["test_data_path"] = data_config["test_data_path"]
                data_dict["test_annotation_file"] = data_config.get("test_annotation_file", "")
        else:
            data_dict.update({
                "train_data_path": data_config.get("train_data_path", ""),
                "val_data_path": data_config.get("val_data_path", ""),
            })
            if data_config.get("test_data_path"):
                data_dict["test_data_path"] = data_config["test_data_path"]
        
        # Augmentation settings
        if data_config.get("augment", True):
            data_dict["augmentations"] = data_config.get("augmentations", {
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
        
        # Training configuration
        monitor_metrics = {
            "classification": ("val_loss", "min"),
            "detection": ("val_map", "max"),
            "semantic_segmentation": ("val_miou", "max"),
            "instance_segmentation": ("val_map", "max"),
            "universal_segmentation": ("val_miou", "max"),
        }
        monitor_metric, monitor_mode = monitor_metrics.get(task, ("val_loss", "min"))
        
        training_dict = {
            "max_epochs": training_config.get("epochs", 100),
            "experiment_name": mlflow_config.get("experiment_name", f"/Users/default/cv_{task}"),
            "learning_rate": training_config.get("learning_rate", 1e-4),
            "weight_decay": training_config.get("weight_decay", 1e-4),
            "scheduler": training_config.get("scheduler", "cosine"),
            "scheduler_params": {
                "T_max": training_config.get("epochs", 100),
                "eta_min": 1e-6
            },
            "early_stopping_patience": training_config.get("early_stopping_patience", 20),
            "monitor_metric": monitor_metric,
            "monitor_mode": monitor_mode,
            "checkpoint_dir": training_config.get("checkpoint_dir", f"/Volumes/<catalog>/<schema>/<volume>/checkpoints/{task}"),
            "volume_checkpoint_dir": training_config.get("volume_checkpoint_dir", f"/Volumes/<catalog>/<schema>/<volume>/volume_checkpoints/{task}"),
            "save_top_k": training_config.get("save_top_k", 3),
            "log_every_n_steps": training_config.get("log_every_n_steps", 50),
            "distributed": training_config.get("distributed", False),
            "use_ray": training_config.get("use_ray", False),
            "use_gpu": training_config.get("use_gpu", True),
            "num_workers": training_config.get("num_workers", 1),
            "resources_per_worker": {
                "CPU": training_config.get("cpu_per_worker", 4),
                "GPU": training_config.get("gpu_per_worker", 1),
            }
        }
        
        # Output configuration
        output_dict = {
            "results_dir": output_config.get("results_dir", f"/Volumes/<catalog>/<schema>/<volume>/results/{task}"),
            "save_predictions": output_config.get("save_predictions", True),
            "visualization": {
                "save_images": output_config.get("save_images", True),
                "max_images": output_config.get("max_images", 10),
            }
        }
        
        if task == "detection":
            output_dict["visualization"].update({
                "confidence_threshold": model_specific_config.get("confidence_threshold", 0.5) if model_specific_config else 0.5,
                "max_boxes": 20,
            })
        
        # Complete configuration
        config = {
            "model": model_config,
            "data": data_dict,
            "training": training_dict,
            "output": output_dict,
        }
        
        return config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], file_path: str) -> str:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            file_path: Path to save the YAML file
            
        Returns:
            Path where the config was saved
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return str(path)
    
    @classmethod
    def load_config(cls, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Configuration dictionary
        """
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required sections
        required_sections = ["model", "data", "training", "output"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate model section
        if "model" in config:
            required_model_fields = ["model_name", "task_type", "num_classes"]
            for field in required_model_fields:
                if field not in config["model"]:
                    errors.append(f"Missing required field: model.{field}")
        
        # Validate data section
        if "data" in config:
            required_data_fields = ["train_data_path", "val_data_path", "batch_size"]
            for field in required_data_fields:
                if field not in config["data"]:
                    errors.append(f"Missing required field: data.{field}")
        
        # Validate training section
        if "training" in config:
            required_training_fields = ["max_epochs", "learning_rate", "checkpoint_dir"]
            for field in required_training_fields:
                if field not in config["training"]:
                    errors.append(f"Missing required field: training.{field}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @classmethod
    def get_config_preview(cls, config: Dict[str, Any]) -> str:
        """
        Generate a YAML preview string of the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            YAML string representation
        """
        return yaml.dump(config, default_flow_style=False, sort_keys=False)


#!/usr/bin/env python3
"""
Example script demonstrating the data configuration format for train/validation/test splits.

This script shows how to configure data modules with separate paths for training,
validation, and test datasets, which is required for proper dataset splitting.
"""

import yaml
from pathlib import Path
from src.tasks.detection.data import DetectionDataModule, DetectionDataConfig
from src.tasks.classification.data import ClassificationDataModule, ClassificationDataConfig
from src.tasks.semantic_segmentation.data import SemanticSegmentationDataModule, SemanticSegmentationDataConfig

def example_detection_config():
    """Example configuration for object detection with separate train/val/test paths."""
    
    detection_config = {
        "train_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train2017/",
        "train_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_train2017.json",
        "val_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val2017/",
        "val_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_val2017.json",
        "test_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/test2017/",
        "test_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_test2017.json",
        "batch_size": 8,
        "num_workers": 4,
        "model_name": "facebook/detr-resnet-50",
        "image_size": [800, 800]
    }
    
    print("=== Detection Configuration Example ===")
    print("Configuration with separate train/val/test paths:")
    print(yaml.dump(detection_config, default_flow_style=False))
    
    # Create data module
    data_module = DetectionDataModule(detection_config)
    data_module.setup()
    
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Val dataset size: {len(data_module.val_dataset)}")
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    print()

def example_classification_config():
    """Example configuration for classification with separate train/val/test paths."""
    
    classification_config = {
        "train_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train/",
        "val_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val/",
        "test_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/test/",
        "batch_size": 32,
        "num_workers": 4,
        "model_name": "microsoft/resnet-50",
        "image_size": 224
    }
    
    print("=== Classification Configuration Example ===")
    print("Configuration with separate train/val/test paths:")
    print(yaml.dump(classification_config, default_flow_style=False))
    
    # Create data module
    data_module = ClassificationDataModule(classification_config)
    data_module.setup()
    
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Val dataset size: {len(data_module.val_dataset)}")
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    print()

def example_optional_test_data():
    """Example showing optional test data configuration."""
    
    print("=== Optional Test Data Example ===")
    
    # Configuration without test data (will use validation data for testing)
    config_without_test = {
        "train_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train2017/",
        "train_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_train2017.json",
        "val_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val2017/",
        "val_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_val2017.json",
        "batch_size": 8,
        "model_name": "facebook/detr-resnet-50"
    }
    
    print("Configuration without test data (test_data_path and test_annotation_file omitted):")
    print(yaml.dump(config_without_test, default_flow_style=False))
    
    # Create data module
    data_module = DetectionDataModule(config_without_test)
    data_module.setup()
    
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Val dataset size: {len(data_module.val_dataset)}")
    print(f"Test dataset size: {len(data_module.test_dataset)} (uses validation data)")
    print()

def example_yaml_config():
    """Example of how to structure YAML configuration files."""
    
    yaml_config = {
        "data": {
            "train_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train2017/",
            "train_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_train2017.json",
            "val_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val2017/",
            "val_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_val2017.json",
            "test_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/test2017/",
            "test_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_test2017.json",
            "batch_size": 16,
            "model_name": "facebook/detr-resnet-50",
            "image_size": [800, 800]
        },
        "training": {
            "max_epochs": 50,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4
        }
    }
    
    print("=== YAML Configuration Example ===")
    print("Recommended YAML structure:")
    print(yaml.dump(yaml_config, default_flow_style=False, sort_keys=False))
    
    # Save example config
    config_path = Path("example_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Example config saved to: {config_path}")
    print()

def validate_config_example():
    """Example of how to validate data configuration."""
    
    print("=== Configuration Validation Example ===")
    
    def validate_data_config(config):
        """Validate that all required paths exist."""
        from pathlib import Path
        
        # Check required paths
        required_paths = [
            ("train_data_path", config.get("train_data_path")),
            ("val_data_path", config.get("val_data_path"))
        ]
        
        for name, path in required_paths:
            if not path:
                raise ValueError(f"Required field '{name}' is missing")
            if not Path(path).exists():
                raise ValueError(f"Path does not exist: {name} = {path}")
        
        # Check optional test path
        test_path = config.get("test_data_path")
        if test_path and not Path(test_path).exists():
            raise ValueError(f"Test path does not exist: {test_path}")
        
        # Check annotation files for detection tasks
        if "train_annotation_file" in config:
            annotation_files = [
                ("train_annotation_file", config.get("train_annotation_file")),
                ("val_annotation_file", config.get("val_annotation_file")),
                ("test_annotation_file", config.get("test_annotation_file"))
            ]
            
            for name, ann_file in annotation_files:
                if ann_file and not Path(ann_file).exists():
                    raise ValueError(f"Annotation file does not exist: {name} = {ann_file}")
        
        print("✅ Configuration validation passed!")
    
    # Example valid configuration
    valid_config = {
        "train_data_path": "/path/to/coco/train2017/",
        "train_annotation_file": "/path/to/coco/annotations/instances_train2017.json",
        "val_data_path": "/path/to/coco/val2017/",
        "val_annotation_file": "/path/to/coco/annotations/instances_val2017.json",
        "batch_size": 8,
        "model_name": "facebook/detr-resnet-50"
    }
    
    print("Validating configuration...")
    try:
        validate_data_config(valid_config)
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
    
    print()

def main():
    """Run all examples."""
    print("Data Configuration Examples")
    print("=" * 50)
    print()
    
    example_detection_config()
    example_classification_config()
    example_optional_test_data()
    example_yaml_config()
    validate_config_example()
    
    print("Summary:")
    print("- Separate train/val/test paths are required for proper dataset splitting")
    print("- Test data is optional and will fall back to validation data if not provided")
    print("- All data modules now use the same consistent format")
    print("- Configuration validation helps catch path errors early")

if __name__ == "__main__":
    main() 
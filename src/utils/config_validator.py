"""
Configuration validator for the simplified MLflow integration approach.

This utility helps ensure config files are compatible with the new UnifiedTrainer
and provides validation for the simplified MLflow integration approach.
"""

import os
from dataclasses import fields
from typing import Any, Dict, List, Optional

from training.trainer import UnifiedTrainerConfig


def validate_config_for_simplified_mlflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and update configuration for compatibility with simplified MLflow integration.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        Updated configuration with any necessary fixes applied
        
    Raises:
        ValueError: If required fields are missing and cannot be inferred
    """
    print("🔍 Validating configuration for simplified MLflow integration...")
    
    # Create a copy to avoid modifying the original
    validated_config = config.copy()
    
    # 1. Check for required UnifiedTrainerConfig fields
    required_fields = {
        'task': 'model.task_type',
        'model_name': 'model.model_name', 
        'max_epochs': 'training.max_epochs',
        'log_every_n_steps': 'training.log_every_n_steps',
        'monitor_metric': 'training.monitor_metric',
        'monitor_mode': 'training.monitor_mode',
        'early_stopping_patience': 'training.early_stopping_patience',
        'checkpoint_dir': 'training.checkpoint_dir',
        'distributed': 'training.distributed'
    }
    
    missing_fields = []
    field_mappings = {}
    
    for required_field, config_path in required_fields.items():
        if required_field not in validated_config:
            # Try to extract from nested config
            keys = config_path.split('.')
            value = validated_config
            for key in keys:
                if key in value:
                    value = value[key]
                else:
                    value = None
                    break
            
            if value is not None:
                field_mappings[required_field] = value
                print(f"✅ Mapped {config_path} -> {required_field}: {value}")
            else:
                missing_fields.append(required_field)
                print(f"❌ Missing required field: {required_field} (expected at {config_path})")
    
    # 2. Add missing fields with defaults if possible
    if 'save_top_k' not in validated_config:
        validated_config['save_top_k'] = 3
        print("✅ Added default save_top_k: 3")
    
    if 'volume_checkpoint_dir' not in validated_config:
        # Try to infer from checkpoint_dir
        if 'checkpoint_dir' in field_mappings:
            checkpoint_dir = field_mappings['checkpoint_dir']
            volume_dir = checkpoint_dir.replace('/checkpoints/', '/volume_checkpoints/')
            validated_config['volume_checkpoint_dir'] = volume_dir
            print(f"✅ Inferred volume_checkpoint_dir: {volume_dir}")
        else:
            validated_config['volume_checkpoint_dir'] = None
            print("⚠️  No volume_checkpoint_dir specified (will be None)")
    
    # 3. Add the mapped fields to the top level
    for field, value in field_mappings.items():
        validated_config[field] = value
    
    # 4. Check for deprecated MLflow configuration
    if 'mlflow' in validated_config:
        print("⚠️  Found deprecated 'mlflow' section - this is no longer needed")
        print("   The simplified approach uses create_databricks_logger() directly")
        print("   You can remove the mlflow section from your config")
        # Don't remove it automatically to avoid breaking existing code
    
    # 5. Validate field types
    try:
        # Test if the config can be used to create UnifiedTrainerConfig
        test_config = {k: v for k, v in validated_config.items() 
                      if k in [f.name for f in fields(UnifiedTrainerConfig)]}
        UnifiedTrainerConfig(**test_config)
        print("✅ Configuration validation passed!")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        raise ValueError(f"Configuration validation failed: {e}")
    
    return validated_config


def get_config_compatibility_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a compatibility report for the configuration.
    
    Args:
        config: The configuration dictionary to analyze
        
    Returns:
        Dictionary containing compatibility information
    """
    report = {
        'compatible': True,
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check for required sections
    required_sections = ['model', 'data', 'training']
    for section in required_sections:
        if section not in config:
            report['issues'].append(f"Missing required section: {section}")
            report['compatible'] = False
    
    # Check for required model fields
    if 'model' in config:
        model_required = ['model_name', 'task_type', 'num_classes']
        for field in model_required:
            if field not in config['model']:
                report['issues'].append(f"Missing model.{field}")
                report['compatible'] = False
    
    # Check for required training fields
    if 'training' in config:
        training_required = ['max_epochs', 'learning_rate', 'monitor_metric', 'monitor_mode']
        for field in training_required:
            if field not in config['training']:
                report['issues'].append(f"Missing training.{field}")
                report['compatible'] = False
    
    # Check for deprecated MLflow configuration
    if 'mlflow' in config:
        report['warnings'].append("Deprecated 'mlflow' section found - no longer needed")
        report['recommendations'].append("Remove the mlflow section and use create_databricks_logger() directly")
    
    # Check for missing volume checkpoint directory
    if 'training' in config and 'checkpoint_dir' in config['training']:
        if 'volume_checkpoint_dir' not in config['training']:
            report['warnings'].append("No volume_checkpoint_dir specified")
            report['recommendations'].append("Add volume_checkpoint_dir for persistent storage")
    
    return report


def print_config_compatibility_report(config: Dict[str, Any]) -> None:
    """
    Print a formatted compatibility report for the configuration.
    
    Args:
        config: The configuration dictionary to analyze
    """
    report = get_config_compatibility_report(config)
    
    print("\n📋 Configuration Compatibility Report")
    print("=" * 50)
    
    if report['compatible']:
        print("✅ Configuration is compatible with simplified MLflow integration")
    else:
        print("❌ Configuration has compatibility issues")
    
    if report['issues']:
        print("\n🚨 Issues:")
        for issue in report['issues']:
            print(f"   • {issue}")
    
    if report['warnings']:
        print("\n⚠️  Warnings:")
        for warning in report['warnings']:
            print(f"   • {warning}")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    print("=" * 50)


def create_simplified_config_template() -> Dict[str, Any]:
    """
    Create a template configuration for the simplified MLflow integration.
    
    Returns:
        Template configuration dictionary
    """
    template = {
        "model": {
            "model_name": "facebook/detr-resnet-50",
            "task_type": "detection",
            "num_classes": 80,
            "pretrained": True,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4
        },
        "data": {
            "train_data_path": "/Volumes/<catalog>/<schema>/<volume>/data/train/",
            "train_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/data/train/annotations.json",
            "val_data_path": "/Volumes/<catalog>/<schema>/<volume>/data/val/",
            "val_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/data/val/annotations.json",
            "batch_size": 16,
            "num_workers": 4,
            "image_size": [800, 800]
        },
        "training": {
            "max_epochs": 50,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "early_stopping_patience": 20,
            "monitor_metric": "val_map",
            "monitor_mode": "max",
            "checkpoint_dir": "/Volumes/<catalog>/<schema>/<volume>/checkpoints",
            "volume_checkpoint_dir": "/Volumes/<catalog>/<schema>/<volume>/volume_checkpoints",
            "save_top_k": 3,
            "log_every_n_steps": 50,
            "distributed": False
        },
        "output": {
            "results_dir": "/Volumes/<catalog>/<schema>/<volume>/results",
            "save_predictions": True
        }
    }
    
    return template 
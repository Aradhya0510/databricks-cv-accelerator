"""
Configuration validator for the Serverless GPU training approach.

This utility helps ensure config files are compatible with serverless GPU training
and provides validation for the Serverless GPU training approach.
"""

import os
from dataclasses import fields
from typing import Any, Dict, List, Optional

from training.trainer import UnifiedTrainer


def validate_config_for_serverless_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and update configuration for compatibility with Serverless GPU training.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        Updated configuration with any necessary fixes applied
        
    Raises:
        ValueError: If required fields are missing and cannot be inferred
    """
    print("🔍 Validating configuration for Serverless GPU training...")
    
    # Create a copy to avoid modifying the original
    validated_config = config.copy()
    
    # 1. Check for required serverless GPU fields
    required_fields = {
        'task': 'model.task_type',
        'model_name': 'model.model_name', 
        'max_epochs': 'training.max_epochs',
        'log_every_n_steps': 'training.log_every_n_steps',
        'monitor_metric': 'training.monitor_metric',
        'monitor_mode': 'training.monitor_mode',
        'early_stopping_patience': 'training.early_stopping_patience',
        'checkpoint_dir': 'training.checkpoint_dir',
        'distributed': 'training.distributed',
        'use_serverless_gpu': 'training.use_serverless_gpu',
        'serverless_gpu_type': 'training.serverless_gpu_type',
        'serverless_gpu_count': 'training.serverless_gpu_count'
    }
    
    missing_fields = []
    field_mappings = {}
    
    for required_field, config_path in required_fields.items():
        if required_field not in validated_config:
            # Try to extract from nested config path
            section, key = config_path.split('.')
            if section in validated_config and key in validated_config[section]:
                validated_config[required_field] = validated_config[section][key]
                field_mappings[required_field] = config_path
            else:
                missing_fields.append(required_field)
    
    # 2. Set default values for missing fields
    defaults = {
        'task': 'detection',
        'model_name': 'facebook/detr-resnet-50',
        'max_epochs': 100,
        'log_every_n_steps': 50,
        'monitor_metric': 'val_loss',
        'monitor_mode': 'min',
        'early_stopping_patience': 10,
        'checkpoint_dir': '/tmp/checkpoints',
        'distributed': True,
        'use_serverless_gpu': True,
        'serverless_gpu_type': 'A10',
        'serverless_gpu_count': 4
    }
    
    for field in missing_fields:
        if field in defaults:
            validated_config[field] = defaults[field]
            print(f"ℹ️  Set default value for {field}: {defaults[field]}")
        else:
            raise ValueError(f"Required field '{field}' is missing and no default is available")
    
    # 3. Validate Serverless GPU specific settings
    if validated_config.get('use_serverless_gpu', False):
        gpu_type = validated_config.get('serverless_gpu_type', 'A10')
        gpu_count = validated_config.get('serverless_gpu_count', 4)
        
        # Validate GPU type
        if gpu_type not in ['A10', 'H100']:
            raise ValueError(f"Invalid serverless_gpu_type: {gpu_type}. Must be 'A10' or 'H100'")
        
        # Validate H100 limitations
        if gpu_type == 'H100' and gpu_count > 1:
            print("⚠️  H100 GPUs only support single-node workflows, setting serverless_gpu_count to 1")
            validated_config['serverless_gpu_count'] = 1
        
        # Ensure distributed is enabled for serverless GPU
        if not validated_config.get('distributed', False):
            validated_config['distributed'] = True
            print("ℹ️  Enabled distributed training for Serverless GPU")
        
        # Ensure use_ray is disabled when using serverless GPU
        if validated_config.get('use_ray', False):
            validated_config['use_ray'] = False
            print("ℹ️  Disabled Ray when using Serverless GPU")
    
    # 4. Validate task-specific settings
    task = validated_config.get('task', 'detection')
    if task == 'detection':
        if validated_config.get('monitor_metric') == 'val_loss':
            validated_config['monitor_metric'] = 'val_map'
            validated_config['monitor_mode'] = 'max'
            print("ℹ️  Updated monitor_metric to 'val_map' for detection task")
    elif task == 'classification':
        if validated_config.get('monitor_metric') == 'val_loss':
            validated_config['monitor_metric'] = 'val_accuracy'
            validated_config['monitor_mode'] = 'max'
            print("ℹ️  Updated monitor_metric to 'val_accuracy' for classification task")
    elif task == 'segmentation':
        if validated_config.get('monitor_metric') == 'val_loss':
            validated_config['monitor_metric'] = 'val_iou'
            validated_config['monitor_mode'] = 'max'
            print("ℹ️  Updated monitor_metric to 'val_iou' for segmentation task")
    
    # 5. Validate checkpoint directory
    checkpoint_dir = validated_config.get('checkpoint_dir', '/tmp/checkpoints')
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"ℹ️  Created checkpoint directory: {checkpoint_dir}")
        except Exception as e:
            print(f"⚠️  Could not create checkpoint directory {checkpoint_dir}: {e}")
    
    # 6. Print validation summary
    print("✅ Serverless GPU configuration validation completed!")
    print(f"   Task: {validated_config.get('task')}")
    print(f"   Model: {validated_config.get('model_name')}")
    print(f"   Max epochs: {validated_config.get('max_epochs')}")
    print(f"   Monitor metric: {validated_config.get('monitor_metric')}")
    print(f"   Distributed: {validated_config.get('distributed')}")
    print(f"   Serverless GPU: {validated_config.get('use_serverless_gpu')}")
    if validated_config.get('use_serverless_gpu'):
        print(f"   GPU Type: {validated_config.get('serverless_gpu_type')}")
        print(f"   GPU Count: {validated_config.get('serverless_gpu_count')}")
    
    return validated_config


def validate_config_for_serverless_gpu_compatibility(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration for Serverless GPU compatibility and fix common issues.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        Updated configuration with compatibility fixes applied
    """
    print("🔍 Validating Serverless GPU compatibility...")
    
    # Create a copy to avoid modifying the original
    validated_config = config.copy()
    
    # Check if serverless GPU is enabled
    use_serverless_gpu = validated_config.get('training', {}).get('use_serverless_gpu', False)
    
    if not use_serverless_gpu:
        print("ℹ️  Serverless GPU not enabled, skipping validation")
        return validated_config
    
    # Ensure training section exists
    if 'training' not in validated_config:
        validated_config['training'] = {}
    
    # Set default serverless GPU settings if not provided
    training_config = validated_config['training']
    
    if 'serverless_gpu_type' not in training_config:
        training_config['serverless_gpu_type'] = 'A10'
        print("ℹ️  Set default serverless_gpu_type to 'A10'")
    
    if 'serverless_gpu_count' not in training_config:
        training_config['serverless_gpu_count'] = 4
        print("ℹ️  Set default serverless_gpu_count to 4")
    
    # Validate serverless GPU type
    gpu_type = training_config['serverless_gpu_type']
    if gpu_type not in ['A10', 'H100']:
        raise ValueError(f"Invalid serverless_gpu_type: {gpu_type}. Must be 'A10' or 'H100'")
    
    # Validate H100 limitations
    if gpu_type == 'H100' and training_config['serverless_gpu_count'] > 1:
        print("⚠️  H100 GPUs only support single-node workflows, setting serverless_gpu_count to 1")
        training_config['serverless_gpu_count'] = 1
    
    # Ensure distributed is enabled for serverless GPU
    if not training_config.get('distributed', False):
        training_config['distributed'] = True
        print("ℹ️  Enabled distributed training for Serverless GPU")
    
    # Ensure use_ray is disabled when using serverless GPU
    if training_config.get('use_ray', False):
        training_config['use_ray'] = False
        print("ℹ️  Disabled Ray when using Serverless GPU")
    
    print("✅ Serverless GPU compatibility validation completed")
    return validated_config


def get_serverless_gpu_compatibility_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a compatibility report for Serverless GPU configuration.
    
    Args:
        config: The configuration dictionary to analyze
        
    Returns:
        Dictionary containing compatibility report
    """
    report = {
        'serverless_gpu_enabled': False,
        'compatibility_issues': [],
        'recommendations': [],
        'configuration_summary': {}
    }
    
    # Check if serverless GPU is enabled
    use_serverless_gpu = config.get('training', {}).get('use_serverless_gpu', False)
    report['serverless_gpu_enabled'] = use_serverless_gpu
    
    if not use_serverless_gpu:
        report['recommendations'].append("Consider enabling Serverless GPU for better performance and cost optimization")
        return report
    
    # Analyze serverless GPU configuration
    training_config = config.get('training', {})
    gpu_type = training_config.get('serverless_gpu_type', 'A10')
    gpu_count = training_config.get('serverless_gpu_count', 4)
    distributed = training_config.get('distributed', False)
    use_ray = training_config.get('use_ray', False)
    
    report['configuration_summary'] = {
        'gpu_type': gpu_type,
        'gpu_count': gpu_count,
        'distributed': distributed,
        'use_ray': use_ray
    }
    
    # Check for compatibility issues
    if gpu_type not in ['A10', 'H100']:
        report['compatibility_issues'].append(f"Invalid GPU type: {gpu_type}. Must be 'A10' or 'H100'")
    
    if gpu_type == 'H100' and gpu_count > 1:
        report['compatibility_issues'].append("H100 GPUs only support single-node workflows")
        report['recommendations'].append("Set serverless_gpu_count to 1 for H100 GPUs")
    
    if not distributed:
        report['compatibility_issues'].append("Serverless GPU requires distributed training")
        report['recommendations'].append("Set distributed=True for Serverless GPU training")
    
    if use_ray:
        report['compatibility_issues'].append("Cannot use both Ray and Serverless GPU")
        report['recommendations'].append("Set use_ray=False when using Serverless GPU")
    
    # Add performance recommendations
    if gpu_count > 8:
        report['recommendations'].append("Consider reducing GPU count for better cost efficiency")
    
    if gpu_type == 'A10' and gpu_count < 2:
        report['recommendations'].append("Consider increasing GPU count for better performance with A10 GPUs")
    
    return report


def validate_serverless_gpu_environment() -> bool:
    """
    Validate that the Serverless GPU environment is properly configured.
    
    Returns:
        True if environment is valid, False otherwise
    """
    print("🔍 Validating Serverless GPU environment...")
    
    # Check for serverless_gpu module
    try:
        from serverless_gpu import distributed
        print("✅ serverless_gpu module available")
        serverless_available = True
    except ImportError:
        print("❌ serverless_gpu module not available")
        print("   This requires Serverless GPU compute environment")
        serverless_available = False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available with {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                print(f"   GPU {i}: {props.name} - {memory_gb:.1f} GB")
        else:
            print("❌ CUDA not available")
            serverless_available = False
    except ImportError:
        print("❌ PyTorch not available")
        serverless_available = False
    
    if serverless_available:
        print("✅ Serverless GPU environment validation passed")
    else:
        print("❌ Serverless GPU environment validation failed")
    
    return serverless_available


def create_serverless_gpu_config_template(task_type: str = "detection") -> Dict[str, Any]:
    """
    Create a template configuration for Serverless GPU training.
    
    Args:
        task_type: Type of task ('detection', 'classification', 'segmentation')
        
    Returns:
        Template configuration dictionary
    """
    template = {
        'model': {
            'model_name': 'facebook/detr-resnet-50' if task_type == 'detection' else 'resnet50',
            'task_type': task_type,
            'num_classes': 91 if task_type == 'detection' else 1000,
            'pretrained': True,
            'image_size': 800 if task_type == 'detection' else 224
        },
        'data': {
            'batch_size': 2 if task_type == 'detection' else 32,
            'num_workers': 4,
            'image_size': 800 if task_type == 'detection' else 224,
            'train_data_path': '/Volumes/<catalog>/<schema>/<volume>/<path>/data/train',
            'train_annotation_file': '/Volumes/<catalog>/<schema>/<volume>/<path>/data/train_annotations.json',
            'val_data_path': '/Volumes/<catalog>/<schema>/<volume>/<path>/data/val',
            'val_annotation_file': '/Volumes/<catalog>/<schema>/<volume>/<path>/data/val_annotations.json'
        },
        'training': {
            'max_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'early_stopping_patience': 10,
            'monitor_metric': 'val_map' if task_type == 'detection' else 'val_accuracy',
            'monitor_mode': 'max',
            'checkpoint_dir': '/Volumes/<catalog>/<schema>/<volume>/<path>/checkpoints',
            'save_top_k': 3,
            'log_every_n_steps': 50,
            'distributed': True,
            'use_ray': False,
            'use_serverless_gpu': True,
            'use_gpu': True,
            'serverless_gpu_type': 'A10',
            'serverless_gpu_count': 4,
            'precision': '16-mixed',
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1
        },
        'mlflow': {
            'experiment_name': f'cv_{task_type}_serverless_training',
            'run_name': f'{task_type}_serverless_run',
            'tags': {'task': task_type, 'compute': 'serverless_gpu'},
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
    
    print(f"✅ Created Serverless GPU configuration template for {task_type}")
    return template

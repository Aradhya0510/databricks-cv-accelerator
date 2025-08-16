#!/usr/bin/env python3
"""
Validation script for standardized config files.

This script tests that all config files can be loaded and have the required
structure for the Databricks Computer Vision Pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

def validate_config_structure(config: Dict[str, Any], config_path: str) -> bool:
    """Validate that a config has the required structure."""
    required_sections = ['model', 'data', 'training', 'mlflow', 'output']
    
    # Check for required sections
    for section in required_sections:
        if section not in config:
            print(f"❌ {config_path}: Missing required section '{section}'")
            return False
    
    # Check model section
    model = config['model']
    required_model_fields = ['model_name', 'task_type', 'num_classes']
    for field in required_model_fields:
        if field not in model:
            print(f"❌ {config_path}: Missing required model field '{field}'")
            return False
    
    # Check data section
    data = config['data']
    required_data_fields = ['train_data_path', 'val_data_path', 'batch_size', 'num_workers']
    for field in required_data_fields:
        if field not in data:
            print(f"❌ {config_path}: Missing required data field '{field}'")
            return False
    
    # Check training section
    training = config['training']
    required_training_fields = ['max_epochs', 'learning_rate', 'checkpoint_dir']
    for field in required_training_fields:
        if field not in training:
            print(f"❌ {config_path}: Missing required training field '{field}'")
            return False
    
    # Check mlflow section
    mlflow = config['mlflow']
    required_mlflow_fields = ['experiment_name', 'run_name']
    for field in required_mlflow_fields:
        if field not in mlflow:
            print(f"❌ {config_path}: Missing required mlflow field '{field}'")
            return False
    
    # Check output section
    output = config['output']
    required_output_fields = ['results_dir', 'save_predictions']
    for field in required_output_fields:
        if field not in output:
            print(f"❌ {config_path}: Missing required output field '{field}'")
            return False
    
    return True

def validate_naming_convention(filename: str) -> bool:
    """Validate that filename follows the naming convention."""
    # Expected pattern: {task}_{model_type}_config.yaml
    if not filename.endswith('_config.yaml'):
        return False
    
    # Remove _config.yaml suffix
    base_name = filename.replace('_config.yaml', '')
    
    # Valid task names (including those with underscores)
    valid_tasks = [
        'detection', 'classification', 'semantic_segmentation', 
        'instance_segmentation', 'universal_segmentation'
    ]
    
    # Check if the filename starts with any valid task
    for task in valid_tasks:
        if base_name.startswith(task + '_'):
            return True
    
    return False

def main():
    """Main validation function."""
    configs_dir = Path(__file__).parent
    config_files = list(configs_dir.glob('*_config.yaml'))
    
    print("🔍 Validating config files...")
    print(f"Found {len(config_files)} config files")
    print()
    
    all_valid = True
    
    for config_file in config_files:
        filename = config_file.name
        print(f"📄 Validating {filename}...")
        
        # Check naming convention
        if not validate_naming_convention(filename):
            print(f"❌ {filename}: Does not follow naming convention")
            all_valid = False
            continue
        
        # Try to load the config
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ {filename}: Failed to load YAML - {e}")
            all_valid = False
            continue
        
        # Validate structure
        if validate_config_structure(config, filename):
            print(f"✅ {filename}: Valid")
        else:
            all_valid = False
    
    print()
    if all_valid:
        print("🎉 All config files are valid!")
        return 0
    else:
        print("❌ Some config files have issues. Please fix them.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
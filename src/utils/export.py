import os
import torch
import torch.onnx
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

def export_detection_model(
    model_path: str,
    config_path: str,
    output_dir: str,
    input_shape: tuple = (1, 3, 640, 640),
    format: str = 'onnx'
) -> None:
    """Export detection model to ONNX or TorchScript format.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Directory to save exported model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        format: Export format ('onnx' or 'torchscript')
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    from src.tasks.detection.model import DetectionModel
    model = DetectionModel.load_from_checkpoint(model_path, config=config)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'onnx':
        # Export to ONNX
        output_path = os.path.join(output_dir, 'model.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11
        )
    elif format == 'torchscript':
        # Export to TorchScript
        output_path = os.path.join(output_dir, 'model.pt')
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Model exported to {output_path}")

def export_segmentation_model(
    model_path: str,
    config_path: str,
    output_dir: str,
    input_shape: tuple = (1, 3, 512, 512),
    format: str = 'onnx'
) -> None:
    """Export segmentation model to ONNX or TorchScript format.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Directory to save exported model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        format: Export format ('onnx' or 'torchscript')
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    from src.tasks.semantic_segmentation.model import SemanticSegmentationModel
    model = SemanticSegmentationModel.load_from_checkpoint(model_path, config=config)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'onnx':
        # Export to ONNX
        output_path = os.path.join(output_dir, 'model.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11
        )
    elif format == 'torchscript':
        # Export to TorchScript
        output_path = os.path.join(output_dir, 'model.pt')
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Model exported to {output_path}")

def export_model_to_mlflow(
    model_path: str,
    config_path: str,
    task: str,
    run_id: Optional[str] = None
) -> None:
    """Export model to MLflow.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to config file
        task: Task type ('detection' or 'segmentation')
        run_id: MLflow run ID to log model to
    """
    import mlflow
    import mlflow.pytorch
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    if task == 'detection':
        from src.tasks.detection.model import DetectionModel
        model = DetectionModel.load_from_checkpoint(model_path, config=config)
    elif task == 'segmentation':
        from src.tasks.semantic_segmentation.model import SemanticSegmentationModel
        model = SemanticSegmentationModel.load_from_checkpoint(model_path, config=config)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Log model to MLflow
    with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run() as run:
        # Log model
        mlflow.pytorch.log_model(
            model,
            f"{task}_model",
            registered_model_name=f"{task}_model"
        )
        
        # Log configuration
        mlflow.log_dict(config, "config.yaml")
        
        print(f"Model logged to MLflow run: {run.info.run_id}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--task', type=str, required=True, choices=['detection', 'segmentation'], help='Task type')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'torchscript'], help='Export format')
    parser.add_argument('--mlflow', action='store_true', help='Export to MLflow')
    parser.add_argument('--run-id', type=str, help='MLflow run ID')
    
    args = parser.parse_args()
    
    if args.mlflow:
        export_model_to_mlflow(args.model, args.config, args.task, args.run_id)
    else:
        if args.task == 'detection':
            export_detection_model(args.model, args.config, args.output, format=args.format)
        else:
            export_segmentation_model(args.model, args.config, args.output, format=args.format) 
"""
Model Registration and Deployment Script for Databricks Jobs

Simplified script to register trained models to Unity Catalog and optionally
deploy to Model Serving endpoints. Configure via Lakeflow Jobs UI parameters.

Usage in Databricks Jobs UI:
    1. Select this file as Python script
    2. In Parameters field, add JSON array:
       ["--config_path", "/Volumes/.../config.yaml", 
        "--src_path", "/Workspace/.../src",
        "--checkpoint_path", "/Volumes/.../checkpoint.ckpt",
        "--model_name", "my_model",
        "--deploy", "true"]
    3. Select CPU cluster (no GPU needed for registration)
    4. Run job
"""

import argparse
import os
import sys
from pathlib import Path
import time
from datetime import datetime
from typing import Optional

import torch
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from databricks.sdk import WorkspaceClient


def load_model_from_checkpoint(task: str, checkpoint_path: str, config: dict):
    """Load Lightning model from checkpoint."""
    print(f"\n📦 Loading model from checkpoint...")
    print(f"   Path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if task == "classification":
        from tasks.classification.model import ClassificationModel
        model = ClassificationModel.load_from_checkpoint(checkpoint_path)
        
    elif task == "detection":
        from tasks.detection.model import DetectionModel
        model_config = config["model"].copy()
        model_config["num_workers"] = config["data"].get("num_workers", 4)
        model = DetectionModel.load_from_checkpoint(checkpoint_path, config=model_config)
        
    elif task == "semantic_segmentation":
        from tasks.semantic_segmentation.model import SemanticSegmentationModel
        model = SemanticSegmentationModel.load_from_checkpoint(checkpoint_path)
        
    elif task == "instance_segmentation":
        from tasks.instance_segmentation.model import InstanceSegmentationModel
        model = InstanceSegmentationModel.load_from_checkpoint(checkpoint_path)
        
    elif task == "universal_segmentation":
        from tasks.universal_segmentation.model import UniversalSegmentationModel
        model = UniversalSegmentationModel.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Task: {task}")
    print(f"   Model type: {type(model).__name__}")
    
    return model


class CVModelWrapper(mlflow.pyfunc.PythonModel):
    """PyFunc wrapper for CV models to enable MLflow serving."""
    
    def __init__(self, model, task, config):
        self.model = model
        self.task = task
        self.config = config
    
    def predict(self, context, model_input):
        """Run inference on input data."""
        import torch
        
        # Handle different input formats
        if isinstance(model_input, dict):
            if "pixel_values" in model_input:
                pixel_values = model_input["pixel_values"]
            else:
                raise ValueError("Input must contain 'pixel_values' key")
        else:
            raise ValueError("Input must be a dictionary with 'pixel_values'")
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                pixel_values = pixel_values.cuda()
            
            outputs = self.model(pixel_values)
        
        # Format outputs based on task
        if self.task == "detection":
            results = []
            for i in range(len(outputs["pred_boxes"])):
                results.append({
                    "boxes": outputs["pred_boxes"][i].cpu().numpy().tolist(),
                    "scores": outputs["scores"][i].cpu().numpy().tolist(),
                    "labels": outputs["pred_labels"][i].cpu().numpy().tolist()
                })
            return results
            
        elif self.task == "classification":
            probs = torch.softmax(outputs["logits"], dim=-1)
            return probs.cpu().numpy()
            
        else:
            # Segmentation tasks
            return outputs["pred_masks"].cpu().numpy()


def register_model(model, config, model_name: str, checkpoint_path: str, catalog: str, schema: str):
    """Register model to Unity Catalog."""
    print(f"\n📝 Registering model to Unity Catalog...")
    
    task = config['model']['task_type']
    full_model_name = f"{catalog}.{schema}.{model_name}"
    
    # Create sample input for signature
    print("   Creating model signature...")
    image_size = config["data"].get("image_size", 800)
    if isinstance(image_size, list):
        image_size = image_size[0]
    
    sample_input = {
        "pixel_values": torch.randn(1, 3, image_size, image_size)
    }
    
    # Wrap model
    wrapped_model = CVModelWrapper(model, task, config)
    
    # Get sample output for signature
    model.eval()
    with torch.no_grad():
        sample_output = wrapped_model.predict(None, sample_input)
    
    # Infer signature
    signature = infer_signature(sample_input, sample_output)
    
    # Register model
    print(f"   Registering to: {full_model_name}")
    
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped_model,
            signature=signature,
            pip_requirements=[
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "transformers>=4.30.0",
                "lightning>=2.0.0",
                "Pillow>=9.0.0",
                "numpy>=1.20.0",
            ],
            registered_model_name=full_model_name,
        )
        
        # Log metadata
        mlflow.log_params({
            "task": task,
            "checkpoint": checkpoint_path,
            "model_name": config["model"]["model_name"],
            "registered_at": datetime.now().isoformat()
        })
        
        # Set tags
        mlflow.set_tag("task", task)
        mlflow.set_tag("framework", "pytorch_lightning")
        
        run_id = run.info.run_id
    
    print(f"✅ Model registered successfully!")
    print(f"   Model name: {full_model_name}")
    print(f"   Version: {model_info.registered_model_version}")
    print(f"   Run ID: {run_id}")
    
    return full_model_name, model_info.registered_model_version


def deploy_model(model_name: str, model_version: str, endpoint_name: str, workload_size: str = "Small"):
    """Deploy model to serving endpoint."""
    print(f"\n🚀 Deploying model to serving endpoint...")
    print(f"   Endpoint: {endpoint_name}")
    
    w = WorkspaceClient()
    
    # Check if endpoint exists
    try:
        existing_endpoint = w.serving_endpoints.get(endpoint_name)
        endpoint_exists = True
        print(f"   Updating existing endpoint...")
    except Exception:
        endpoint_exists = False
        print(f"   Creating new endpoint...")
    
    # Prepare endpoint configuration
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )
    
    served_entity = ServedEntityInput(
        entity_name=model_name,
        entity_version=model_version,
        workload_size=workload_size,
        scale_to_zero_enabled=True
    )
    
    if endpoint_exists:
        # Update existing endpoint
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=[served_entity]
        )
    else:
        # Create new endpoint
        config = EndpointCoreConfigInput(served_entities=[served_entity])
        w.serving_endpoints.create(name=endpoint_name, config=config)
    
    # Wait for endpoint to be ready
    print("   Waiting for endpoint to be ready...")
    max_wait = 1200  # 20 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state.config_update if endpoint.state else None
        
        if state == "NOT_UPDATING":
            print(f"✅ Endpoint deployed successfully!")
            print(f"   Endpoint: {endpoint_name}")
            print(f"   Status: Ready")
            return endpoint_name
        elif state == "UPDATE_FAILED":
            raise RuntimeError(f"Endpoint deployment failed")
        
        elapsed = int(time.time() - start_time)
        print(f"   Status: {state}... ({elapsed}s elapsed)")
        time.sleep(30)
    
    raise TimeoutError(f"Endpoint deployment timed out after {max_wait}s")


def main():
    """Main execution."""
    print("=" * 80)
    print("📦 Databricks CV Model Registration & Deployment")
    print("=" * 80)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Register and deploy CV models")
    parser.add_argument("--config_path", required=True, help="Path to YAML configuration file")
    parser.add_argument("--src_path", required=True, help="Path to src directory")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--model_name", required=True, help="Model name in Unity Catalog")
    parser.add_argument("--deploy", default="false", help="Deploy to endpoint (true/false)")
    parser.add_argument("--endpoint_name", default=None, help="Endpoint name (default: {model_name}-endpoint)")
    parser.add_argument("--workload_size", default="Small", help="Endpoint size (Small/Medium/Large)")
    parser.add_argument("--catalog", default="main", help="Unity Catalog name")
    parser.add_argument("--schema", default="cv_models", help="Unity Catalog schema")
    args = parser.parse_args()
    
    print(f"📦 Using src path: {args.src_path}")
    
    # Add src to Python path
    if args.src_path not in sys.path:
        sys.path.insert(0, args.src_path)
    
    # Import after adding to path
    from config import load_config
    
    # Get parameters from arguments
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    model_name = args.model_name
    should_deploy = args.deploy.lower() == "true"
    
    # Optional deployment parameters
    endpoint_name = args.endpoint_name if args.endpoint_name else f"{model_name}-endpoint"
    workload_size = args.workload_size
    catalog = args.catalog
    schema = args.schema
    
    print(f"\n📋 Configuration:")
    print(f"   Config: {config_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Model name: {model_name}")
    print(f"   Deploy: {should_deploy}")
    if should_deploy:
        print(f"   Endpoint: {endpoint_name}")
        print(f"   Workload size: {workload_size}")
    
    # Load configuration
    config = load_config(config_path)
    task = config['model']['task_type']
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(task, checkpoint_path, config)
    
    # Register model
    full_model_name, version = register_model(
        model, config, model_name, checkpoint_path, catalog, schema
    )
    
    # Deploy if requested
    if should_deploy:
        deploy_model(full_model_name, version, endpoint_name, workload_size)
    else:
        print("\n📝 Model registered but not deployed")
        print(f"   To deploy, add job parameter: deploy=true")
    
    print("\n" + "=" * 80)
    print("✅ Operation Completed Successfully!")
    print("=" * 80)
    print(f"\n📦 Model: {full_model_name} (version {version})")
    if should_deploy:
        print(f"🚀 Endpoint: {endpoint_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Operation Failed!")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


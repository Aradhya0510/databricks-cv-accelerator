# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation and Prediction
# MAGIC 
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Load a trained model and evaluate its performance
# MAGIC 2. Make predictions on test images
# MAGIC 3. Convert predictions to standardized format
# MAGIC 4. Evaluate model performance with task-specific metrics
# MAGIC 5. Visualize predictions

# COMMAND ----------

%pip install -r "../requirements.txt"
dbutils.library.restartPython()

# COMMAND ----------

%run ./01_data_preparation
%run ./02_model_training

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import yaml
import torch
import mlflow
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Any

# Add the project root to Python path
project_root = "/Volumes/<catalog>/<schema>/<volume>/<path>"
sys.path.append(project_root)

# Import project modules
from src.utils.logging import setup_logger
from src.utils.coco_handler import COCOHandler
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule
from src.tasks.detection.evaluate import DetectionEvaluator
from src.tasks.classification.model import ClassificationModel
from src.tasks.classification.data import ClassificationDataModule
from src.tasks.classification.evaluate import ClassificationEvaluator
from src.tasks.semantic_segmentation.model import SemanticSegmentationModel
from src.tasks.semantic_segmentation.data import SemanticSegmentationDataModule
from src.tasks.semantic_segmentation.evaluate import SemanticSegmentationEvaluator
from src.tasks.instance_segmentation.model import InstanceSegmentationModel
from src.tasks.instance_segmentation.data import InstanceSegmentationDataModule
from src.tasks.instance_segmentation.evaluate import InstanceSegmentationEvaluator
from src.tasks.panoptic_segmentation.model import PanopticSegmentationModel
from src.tasks.panoptic_segmentation.data import PanopticSegmentationDataModule
from src.tasks.panoptic_segmentation.evaluate import PanopticSegmentationEvaluator

# COMMAND ----------

# DBTITLE 1,Initialize Logging
volume_path = os.getenv("UNITY_CATALOG_VOLUME", project_root)
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="model_evaluation",
    log_file=f"{log_dir}/evaluation.log"
)

# COMMAND ----------

# DBTITLE 1,Load Configuration and Model
def load_task_config(task: str):
    """Load task-specific configuration."""
    config_path = f"{volume_path}/configs/{task}_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_model_class(task: str):
    """Get the appropriate model class based on task."""
    model_classes = {
        'detection': DetectionModel,
        'classification': ClassificationModel,
        'semantic_segmentation': SemanticSegmentationModel,
        'instance_segmentation': InstanceSegmentationModel,
        'panoptic_segmentation': PanopticSegmentationModel
    }
    
    if task not in model_classes:
        raise ValueError(f"Unsupported task: {task}")
    
    return model_classes[task]

def get_data_module_class(task: str):
    """Get the appropriate data module class based on task."""
    data_module_classes = {
        'detection': DetectionDataModule,
        'classification': ClassificationDataModule,
        'semantic_segmentation': SemanticSegmentationDataModule,
        'instance_segmentation': InstanceSegmentationDataModule,
        'panoptic_segmentation': PanopticSegmentationDataModule
    }
    
    if task not in data_module_classes:
        raise ValueError(f"Unsupported task: {task}")
    
    return data_module_classes[task]

def get_evaluator_class(task: str):
    """Get the appropriate evaluator class based on task."""
    evaluator_classes = {
        'detection': DetectionEvaluator,
        'classification': ClassificationEvaluator,
        'semantic_segmentation': SemanticSegmentationEvaluator,
        'instance_segmentation': InstanceSegmentationEvaluator,
        'panoptic_segmentation': PanopticSegmentationEvaluator
    }
    
    if task not in evaluator_classes:
        raise ValueError(f"Unsupported task: {task}")
    
    return evaluator_classes[task]

def load_model(task: str, model_path: str = None):
    """Load a trained model."""
    config = load_task_config(task)
    
    # Load model from MLflow if no specific path provided
    if model_path is None:
        model_name = config['model']['model_name']
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pytorch.load_model(model_uri)
    else:
        model = torch.load(model_path, map_location='cpu')
    
    model.eval()
    return model, config

# COMMAND ----------

# DBTITLE 1,Make Predictions
def predict_image(model, image_path: str, task: str, config: dict) -> Dict[str, Any]:
    """Make predictions on a single image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Get model-specific prediction function
    if task == 'detection':
        return predict_detection(model, image, config)
    elif task == 'classification':
        return predict_classification(model, image, config)
    elif task == 'semantic_segmentation':
        return predict_semantic_segmentation(model, image, config)
    elif task == 'instance_segmentation':
        return predict_instance_segmentation(model, image, config)
    elif task == 'panoptic_segmentation':
        return predict_panoptic_segmentation(model, image, config)
    else:
        raise ValueError(f"Unsupported task: {task}")

def predict_detection(model, image: Image.Image, config: dict) -> Dict[str, Any]:
    """Make detection predictions."""
    # This would use the model's inference method
    # For now, return a placeholder structure
    return {
        "boxes": np.array([]),
        "scores": np.array([]),
        "labels": np.array([])
    }

def predict_classification(model, image: Image.Image, config: dict) -> Dict[str, Any]:
    """Make classification predictions."""
    # This would use the model's inference method
    # For now, return a placeholder structure
    return {
        "logits": np.array([]),
        "probabilities": np.array([]),
        "predicted_class": 0
    }

def predict_semantic_segmentation(model, image: Image.Image, config: dict) -> Dict[str, Any]:
    """Make semantic segmentation predictions."""
    # This would use the model's inference method
    # For now, return a placeholder structure
    return {
        "logits": np.array([]),
        "segmentation": np.array([])
    }

def predict_instance_segmentation(model, image: Image.Image, config: dict) -> Dict[str, Any]:
    """Make instance segmentation predictions."""
    # This would use the model's inference method
    # For now, return a placeholder structure
    return {
        "boxes": np.array([]),
        "scores": np.array([]),
        "labels": np.array([]),
        "masks": np.array([])
    }

def predict_panoptic_segmentation(model, image: Image.Image, config: dict) -> Dict[str, Any]:
    """Make panoptic segmentation predictions."""
    # This would use the model's inference method
    # For now, return a placeholder structure
    return {
        "segmentation": np.array([]),
        "segments_info": []
    }

# COMMAND ----------

# DBTITLE 1,Evaluate Model Performance
def evaluate_model(task: str, model, test_loader, config: dict):
    """Evaluate model performance on test set."""
    evaluator_class = get_evaluator_class(task)
    evaluator = evaluator_class(model=model, config=config)
    
    # Run evaluation
    metrics = evaluator.evaluate(test_loader)
    
    # Log metrics
    logger.info(f"Evaluation metrics for {task}: {metrics}")
    
    return metrics

def save_evaluation_results(task: str, metrics: dict):
    """Save evaluation results to file."""
    results_path = f"{volume_path}/results/{task}_evaluation_results.yaml"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        yaml.dump(metrics, f)
    
    logger.info(f"Evaluation results saved to {results_path}")

# COMMAND ----------

# DBTITLE 1,Visualize Predictions
def visualize_prediction(image_path: str, prediction: dict, task: str, config: dict):
    """Visualize model predictions on an image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Visualize based on task type
    if task == 'detection':
        visualize_detection_prediction(prediction, config)
    elif task == 'classification':
        visualize_classification_prediction(prediction, config)
    elif task == 'semantic_segmentation':
        visualize_semantic_segmentation_prediction(prediction, config)
    elif task == 'instance_segmentation':
        visualize_instance_segmentation_prediction(prediction, config)
    elif task == 'panoptic_segmentation':
        visualize_panoptic_segmentation_prediction(prediction, config)
    
    plt.axis("off")
    plt.show()

def visualize_detection_prediction(prediction: dict, config: dict):
    """Visualize detection predictions."""
    boxes = prediction.get("boxes", [])
    scores = prediction.get("scores", [])
    labels = prediction.get("labels", [])
    
    for box, score, label in zip(boxes, scores, labels):
        if score > config["model"].get("confidence_threshold", 0.5):
            # Convert box to [x, y, width, height] for plotting
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
            plt.gca().add_patch(rect)
            
            # Add label with score
            plt.text(
                x1,
                y1 - 5,
                f"Class {label}: {score:.2f}",
                color="red",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.7)
            )

def visualize_classification_prediction(prediction: dict, config: dict):
    """Visualize classification predictions."""
    probabilities = prediction.get("probabilities", [])
    predicted_class = prediction.get("predicted_class", 0)
    
    # Add text with prediction
    plt.text(
        10,
        10,
        f"Predicted Class: {predicted_class}\nConfidence: {probabilities[predicted_class]:.2f}",
        color="white",
        fontsize=14,
        bbox=dict(facecolor="black", alpha=0.7)
    )

def visualize_semantic_segmentation_prediction(prediction: dict, config: dict):
    """Visualize semantic segmentation predictions."""
    segmentation = prediction.get("segmentation", np.array([]))
    
    if len(segmentation) > 0:
        # Overlay segmentation mask
        plt.imshow(segmentation, alpha=0.5, cmap='tab20')

def visualize_instance_segmentation_prediction(prediction: dict, config: dict):
    """Visualize instance segmentation predictions."""
    boxes = prediction.get("boxes", [])
    masks = prediction.get("masks", [])
    
    for box, mask in zip(boxes, masks):
        if len(mask) > 0:
            # Overlay instance mask
            plt.imshow(mask, alpha=0.3, cmap='Set1')

def visualize_panoptic_segmentation_prediction(prediction: dict, config: dict):
    """Visualize panoptic segmentation predictions."""
    segmentation = prediction.get("segmentation", np.array([]))
    
    if len(segmentation) > 0:
        # Overlay panoptic segmentation
        plt.imshow(segmentation, alpha=0.5, cmap='tab20')

# COMMAND ----------

# DBTITLE 1,Main Evaluation Function
def evaluate_task_model(task: str, model_path: str = None):
    """Main function to evaluate a trained model."""
    # Load model and configuration
    model, config = load_model(task, model_path)
    
    # Prepare test data
    test_loader, _ = prepare_data(task)
    
    # Evaluate model
    metrics = evaluate_model(task, model, test_loader, config)
    
    # Save results
    save_evaluation_results(task, metrics)
    
    return model, config, metrics

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Evaluate detection model
task = "detection"

# Evaluate model
model, config, metrics = evaluate_task_model(task)

# Visualize predictions on sample images
test_images = [
    f"{volume_path}/data/test/images/sample1.jpg",
    f"{volume_path}/data/test/images/sample2.jpg",
    f"{volume_path}/data/test/images/sample3.jpg"
]

for image_path in test_images:
    if os.path.exists(image_path):
        # Make prediction
        prediction = predict_image(model, image_path, task, config)
        
        # Visualize prediction
        visualize_prediction(image_path, prediction, task, config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review evaluation metrics
# MAGIC 2. Analyze model performance
# MAGIC 3. Identify areas for improvement
# MAGIC 4. Proceed to model registration and deployment notebook 
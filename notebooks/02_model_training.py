# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training
# MAGIC 
# MAGIC This notebook handles model initialization and training for computer vision tasks.
# MAGIC 
# MAGIC ## Unity Catalog Setup
# MAGIC 
# MAGIC The notebook uses the following Unity Catalog volume structure:
# MAGIC ```
# MAGIC /Volumes/cv_ref/datasets/coco_mini/
# MAGIC ├── configs/
# MAGIC │   └── {task}_config.yaml
# MAGIC ├── checkpoints/
# MAGIC │   └── {task}_model/
# MAGIC ├── logs/
# MAGIC │   └── training.log
# MAGIC └── results/
# MAGIC     └── {task}_test_results.yaml
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import mlflow
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2

# Add the project root to Python path
project_root = "/Workspace/Repos/Databricks_CV_ref"
sys.path.append(project_root)

# Import project modules
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule
from src.tasks.classification.model import ClassificationModel
from src.tasks.classification.data import ClassificationDataModule
from src.tasks.semantic_segmentation.model import SemanticSegmentationModel
from src.tasks.semantic_segmentation.data import SemanticSegmentationDataModule
from src.training.trainer import UnifiedTrainer
from src.utils.logging import setup_logger, get_mlflow_logger
from src.tasks.detection.adapters import DETROutputAdapter, get_adapter

# COMMAND ----------

# DBTITLE 1,Initialize Logging
# Get the Unity Catalog volume path from environment or use default
volume_path = os.getenv("UNITY_CATALOG_VOLUME", "/Volumes/cv_ref/datasets/coco_mini")
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="model_training",
    log_file=f"{log_dir}/training.log"
)

# COMMAND ----------

# DBTITLE 1,Setup MLflow
def setup_mlflow(experiment_name: str):
    """Setup MLflow experiment and logger."""
    # Get MLflow logger
    mlflow_logger = get_mlflow_logger(
        experiment_name=experiment_name,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        run_name=f"training_run_{os.getenv('USER')}",
        log_model=True,
        tags={"framework": "pytorch_lightning"}
    )
    
    return mlflow_logger

# COMMAND ----------

# DBTITLE 1,Load Configuration
def load_task_config(task: str):
    """Load task-specific configuration."""
    config_path = f"{volume_path}/configs/{task}_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# COMMAND ----------

# DBTITLE 1,Get Model Class
def get_model_class(task: str):
    """Get the appropriate model class based on task."""
    model_classes = {
        'detection': DetectionModel,
        'classification': ClassificationModel,
        'semantic_segmentation': SemanticSegmentationModel
    }
    
    if task not in model_classes:
        raise ValueError(f"Unsupported task: {task}")
    
    return model_classes[task]

# COMMAND ----------

# DBTITLE 1,Initialize Model
def initialize_model(task: str, config: dict):
    """Initialize model for the specified task."""
    model_class = get_model_class(task)
    
    # Extract model-specific configuration
    model_config = config['model']
    model = model_class(config=model_config)
    
    # For detection task, set the output adapter
    if task == 'detection':
        output_adapter = DETROutputAdapter(
            model_name=config['model']['model_name'],
            image_size=config['data']['image_size'][0]
        )
        model.output_adapter = output_adapter
    
    return model

# COMMAND ----------

# DBTITLE 1,Setup Trainer
def setup_trainer(task: str, model, config: dict, mlflow_logger):
    """Setup trainer for the specified task."""
    trainer = UnifiedTrainer(
        task=task,
        model=model,
        config=config,
        data_module_class=get_data_module_class(task)
    )
    
    return trainer

# COMMAND ----------

# DBTITLE 1,Training Function
def train_model(
    task: str,
    train_loader,
    val_loader,
    test_loader=None,
    experiment_name: str = None
):
    """Main function to train the model."""
    # Load configuration
    config = load_task_config(task)
    
    # Initialize model
    model = initialize_model(task, config)
    
    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    
    # Create data module with config
    data_config = {
        'data_path': config['data']['data_path'],
        'annotation_file': config['data']['annotation_file'],
        'image_size': config['data'].get('image_size', 640),
        'batch_size': config['data'].get('batch_size', 8),
        'num_workers': config['data'].get('num_workers', 4),
        'model_name': config['data'].get('model_name')
    }
    
    # Initialize data module
    data_module_class = get_data_module_class(task)
    data_module = data_module_class(config=data_config)
    
    # Set the data loaders directly
    data_module.train_dataloader = lambda: train_loader
    data_module.val_dataloader = lambda: val_loader
    if test_loader:
        data_module.test_dataloader = lambda: test_loader
    
    # Setup MLflow logger
    mlflow_logger = setup_mlflow(experiment_name)
    
    # Setup trainer with initialized data module
    trainer = UnifiedTrainer(
        config=config,
        model=model,
        data_module=data_module,
        logger=mlflow_logger  # Pass the MLflow logger to the trainer
    )
    
    # Train model
    trainer.train()
    
    # Test model if test loader is available
    if test_loader:
        test_results = trainer.trainer.test(model, datamodule=trainer.data_module)
        
        # Save test results
        results_path = f"{volume_path}/results/{task}_test_results.yaml"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            yaml.dump(test_results, f)
        
        logger.info(f"Test results: {test_results}")
    
    return trainer

def get_data_module_class(task: str):
    """Get the appropriate data module class based on task."""
    data_modules = {
        'detection': DetectionDataModule,
        'classification': ClassificationDataModule,
        'semantic_segmentation': SemanticSegmentationDataModule
    }
    
    if task not in data_modules:
        raise ValueError(f"Unsupported task: {task}")
    
    return data_modules[task]

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Train detection model
task = "detection"
experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/{task}_pipeline"

# Setup MLflow experiment and autologging
mlflow_logger = setup_mlflow(experiment_name)
print(f"Using MLflow experiment: {experiment_name}")

# Prepare data loaders (from previous notebook)
from notebook_01_data_preparation import prepare_data
train_loader, val_loader = prepare_data(task)

# Train model
trainer = train_model(
    task=task,
    train_loader=train_loader,
    val_loader=val_loader,
    experiment_name=experiment_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review training metrics in MLflow
# MAGIC 2. Check model performance on test set
# MAGIC 3. Proceed to hyperparameter tuning notebook 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Model Predictions
# MAGIC 
# MAGIC Let's load the best checkpoint and visualize predictions on a sample image.

# COMMAND ----------

# DBTITLE 1,Load Best Checkpoint and Visualize Predictions
def visualize_predictions(model, image, target, prediction, class_names, original_height, original_width, save_path=None):
    """Visualize model predictions against ground truth.
    
    Args:
        model: The model used for prediction
        image: Input image tensor
        target: Ground truth target dictionary
        prediction: Model prediction dictionary
        class_names: List of class names
        original_height: Original image height in pixels
        original_width: Original image width in pixels
        save_path: Optional path to save the annotated image
    """
    # Convert image to numpy array if it's a tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        # Denormalize image using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    
    # Resize image to a more manageable size for visualization
    display_size = (800, 800)  # Increased from 400x400 to 800x800
    image = cv2.resize(image, display_size)
    
    # Calculate scaling factors
    scale_x = display_size[0] / original_width
    scale_y = display_size[1] / original_height
    
    # Create figure and axis with larger size
    dpi = 72  # Standard screen DPI
    fig, ax = plt.subplots(1, figsize=(12, 12), dpi=dpi)  # Increased from 5x5 to 12x12
    ax.imshow(image)
    
    # Plot ground truth boxes
    if "boxes" in target and len(target["boxes"]) > 0:
        # Handle different label field names
        labels = target.get("class_labels", target.get("labels", []))
        
        # DETR target boxes are in [cx, cy, w, h] normalized format
        # Convert to [x1, y1, x2, y2] absolute pixel coordinates
        target_boxes = target["boxes"].clone()
        
        # First, denormalize by multiplying by original image size
        target_boxes[:, [0, 2]] *= original_width  # cx, w
        target_boxes[:, [1, 3]] *= original_height  # cy, h
        
        # Then convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        cx, cy, w, h = target_boxes.unbind(1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        target_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        
        for box, label in zip(target_boxes_xyxy, labels):
            x1, y1, x2, y2 = box.cpu().numpy()
            # Scale boxes to display size
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='g', facecolor='none'  # Increased linewidth from 1 to 3
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, class_names[label], color='g', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7))  # Increased fontsize and added background
    
    # Plot predicted boxes
    if "boxes" in prediction and len(prediction["boxes"]) > 0:
        # Handle different field names for predictions
        boxes = prediction["boxes"]
        labels = prediction.get("labels", [])
        scores = prediction.get("scores", [])
        
        # Prediction boxes are already in [x1, y1, x2, y2] absolute pixel format
        # from the output adapter's post-processing
        
        for i, box in enumerate(boxes):
            if i < len(scores) and scores[i] > 0.5:  # Only show predictions with confidence > 0.5
                x1, y1, x2, y2 = box.cpu().numpy()
                # Scale boxes to display size
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
                
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=3, edgecolor='r', facecolor='none'  # Increased linewidth from 1 to 3
                )
                ax.add_patch(rect)
                
                # Get label and score
                label = labels[i] if i < len(labels) else 0
                score = scores[i] if i < len(scores) else 0.0
                
                ax.text(
                    x1, y1-5,
                    f"{class_names[label]} ({score:.2f})",
                    color='r', fontsize=14,  # Increased fontsize from 6 to 14
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7)  # Added background
                )
    
    plt.axis('off')
    
    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(
            save_path,
            bbox_inches='tight',
            pad_inches=0,
            dpi=dpi,
            format='png'
        )
        print(f"Saved annotated image to {save_path}")
    
    plt.show()

# Load best checkpoint
checkpoint_path = f"{volume_path}/checkpoints/{task}_model/best.ckpt"
if os.path.exists(checkpoint_path):
    # Load model from checkpoint
    model = initialize_model(task, config)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()
    
    # Initialize data module with proper configuration
    data_config = {
        'data_path': config['data']['data_path'],
        'annotation_file': config['data']['annotation_file'],
        'batch_size': config['data'].get('batch_size', 2),
        'num_workers': config['data'].get('num_workers', 4),
        'model_name': config['data'].get('model_name')
    }
    
    data_module = get_data_module_class(task)(data_config)
    
    # Set up the adapter before initializing datasets
    adapter = get_adapter(
        model_name=config['data'].get('model_name', 'facebook/detr-resnet-50'),
        image_size=config['data'].get('image_size', [800])[0]
    )
    data_module.adapter = adapter
    
    data_module.setup('fit')  # Initialize datasets
    
    # Get a sample batch from validation dataloader
    val_loader = data_module.val_dataloader()
    sample_batch = next(iter(val_loader))
    images = sample_batch["pixel_values"]
    targets = sample_batch["labels"]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
    
    # Format predictions using model's output adapter
    predictions = model._format_predictions(outputs, sample_batch)
    
    # Get class names from data module
    class_names = data_module.class_names
    
    # Create target dictionary for first image
    target = {
        "boxes": targets[0]["boxes"],
        "class_labels": targets[0]["class_labels"]
    }
    
    # Get prediction for first image
    prediction = predictions[0]
    
    # Get original image size for coordinate conversion
    # Check if we have size information in the target
    if "size" in targets[0]:
        original_size = targets[0]["size"]
        original_height, original_width = original_size[0].item(), original_size[1].item()
    else:
        # Fallback: use the processed image size
        original_height, original_width = images[0].shape[1], images[0].shape[2]
    
    # Debug: Print coordinate information
    print(f"Original image size: {original_width}x{original_height}")
    print(f"Target boxes (DETR format [cx,cy,w,h] normalized): {target['boxes'].min():.3f} to {target['boxes'].max():.3f}")
    print(f"Prediction boxes ([x1,y1,x2,y2] absolute): {prediction['boxes'].min():.3f} to {prediction['boxes'].max():.3f}")
    
    # Debug: Show a few converted target boxes
    if len(target['boxes']) > 0:
        target_boxes = target["boxes"].clone()
        target_boxes[:, [0, 2]] *= original_width
        target_boxes[:, [1, 3]] *= original_height
        cx, cy, w, h = target_boxes.unbind(1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        target_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        print(f"Converted target boxes (first 2): {target_boxes_xyxy[:2]}")
    
    # Create save path for annotated image
    save_path = f"{volume_path}/results/{task}_predictions/annotated_image.png"
    
    # Visualize first image in batch
    visualize_predictions(
        model=model,
        image=images[0],
        target=target,
        prediction=prediction,
        class_names=class_names,
        original_height=original_height,
        original_width=original_width,
        save_path=save_path
    )
else:
    print(f"No checkpoint found at {checkpoint_path}") 
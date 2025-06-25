#!/usr/bin/env python3
"""
Example script demonstrating how to use the YOLOS adapter for object detection.

This script shows:
1. How to load and configure a YOLOS model
2. How to use the YOLOS adapter for data preprocessing
3. How to perform inference with the model
4. How to post-process and visualize results
"""

import torch
import yaml
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Import our custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tasks.detection.adapters import YOLOSAdapter, YOLOSOutputAdapter
from tasks.detection.model import DetectionModel, DetectionModelConfig

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_sample_image() -> Image.Image:
    """Download a sample image for testing."""
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def create_sample_target() -> dict:
    """Create a sample target for demonstration."""
    return {
        "boxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
        "labels": torch.tensor([1, 2]),  # Example class labels
        "image_id": torch.tensor([0])
    }

def visualize_predictions(image: Image.Image, predictions: dict, title: str = "YOLOS Predictions"):
    """Visualize model predictions on the image."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]
    
    # COCO class names (first 10 for brevity)
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light"
    ]
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        class_name = coco_classes[label] if label < len(coco_classes) else f"class_{label}"
        ax.text(x1, y1-5, f"{class_name}: {score:.2f}", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                fontsize=8, color='white')
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """Main function demonstrating YOLOS usage."""
    print("🚀 YOLOS Object Detection Example")
    print("=" * 50)
    
    # Load configuration
    config_path = "../configs/detection_yolos_config.yaml"
    config = load_config(config_path)
    
    # Initialize YOLOS model
    model_name = config["model"]["model_name"]
    print(f"📦 Loading YOLOS model: {model_name}")
    
    # Create model configuration
    model_config = DetectionModelConfig(
        model_name=model_name,
        num_classes=config["model"]["num_classes"],
        image_size=config["model"]["image_size"],
        confidence_threshold=config["model"]["confidence_threshold"]
    )
    
    # Initialize the detection model
    model = DetectionModel(model_config)
    model.eval()
    
    # Initialize adapters
    print("🔧 Initializing YOLOS adapters...")
    input_adapter = YOLOSAdapter(model_name=model_name, image_size=config["model"]["image_size"])
    output_adapter = YOLOSOutputAdapter(model_name=model_name, image_size=config["model"]["image_size"])
    
    # Download sample image
    print("📸 Downloading sample image...")
    image = download_sample_image()
    print(f"   Image size: {image.size}")
    
    # Create sample target (for demonstration)
    target = create_sample_target()
    
    # Process image and target using YOLOS adapter
    print("⚙️ Processing image with YOLOS adapter...")
    processed_image, adapted_target = input_adapter(image, target)
    print(f"   Processed image shape: {processed_image.shape}")
    print(f"   Adapted target keys: {list(adapted_target.keys())}")
    
    # Perform inference
    print("🔮 Running inference...")
    with torch.no_grad():
        # Add batch dimension
        batch_image = processed_image.unsqueeze(0)
        
        # Forward pass
        outputs = model(pixel_values=batch_image)
        
        # Format predictions
        predictions = output_adapter.format_predictions(outputs)
        
        if predictions:
            prediction = predictions[0]  # Get first (and only) prediction
            print(f"   Detected {len(prediction['boxes'])} objects")
            
            # Show top 5 detections
            for i, (box, score, label) in enumerate(zip(
                prediction['boxes'][:5], 
                prediction['scores'][:5], 
                prediction['labels'][:5]
            )):
                print(f"   Object {i+1}: Class {label.item()}, Score {score.item():.3f}, Box {box.tolist()}")
        else:
            print("   No objects detected")
    
    # Visualize results
    print("🎨 Visualizing results...")
    if predictions:
        visualize_predictions(image, predictions[0])
    
    print("✅ YOLOS example completed successfully!")

if __name__ == "__main__":
    main() 
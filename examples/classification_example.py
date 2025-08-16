#!/usr/bin/env python3
"""
Example script demonstrating how to use the improved classification adapters.

This script shows:
1. How to load and configure different classification models (ViT, ConvNeXT, Swin)
2. How to use the classification adapters for data preprocessing
3. How to perform inference with the models
4. How to visualize results
"""

import os
import sys
from io import BytesIO
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import requests
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Add src to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tasks.classification.adapters import (
    ConvNeXTAdapter,
    SwinAdapter,
    ViTAdapter,
    get_adapter,
    get_output_adapter,
)
from tasks.classification.model import ClassificationModel, ClassificationModelConfig

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_sample_image() -> Image.Image:
    """Download a sample image for testing."""
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def create_sample_target() -> dict:
    """Create a sample target for demonstration."""
    return {
        "class_labels": torch.tensor([283]),  # Cat class in ImageNet
        "image_id": torch.tensor([0])
    }

def visualize_predictions(image: Image.Image, predictions: dict, class_names: list, title: str = "Classification Predictions"):
    """Visualize model predictions on the image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show image
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Show predictions
    scores = predictions["scores"]
    labels = predictions["labels"]
    
    # Get top 5 predictions
    top_scores, top_indices = torch.topk(scores, 5)
    
    # Create bar chart
    y_pos = range(len(top_indices))
    ax2.barh(y_pos, top_scores.numpy())
    ax2.set_yticks(y_pos)
    
    # Get class names (use ImageNet class names for demo)
    imagenet_classes = [
        "cat", "dog", "bird", "car", "truck",
        "person", "bicycle", "motorcycle", "airplane", "boat"
    ]
    
    class_labels = []
    for idx in top_indices:
        if idx < len(imagenet_classes):
            class_labels.append(imagenet_classes[idx])
        else:
            class_labels.append(f"class_{idx}")
    
    ax2.set_yticklabels(class_labels)
    ax2.set_xlabel('Confidence Score')
    ax2.set_title('Top 5 Predictions')
    ax2.invert_yaxis()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def test_different_models():
    """Test different classification models with their adapters."""
    models_to_test = [
        {
            "name": "ViT",
            "model_name": "google/vit-base-patch16-224",
            "image_size": 224
        },
        {
            "name": "ConvNeXT",
            "model_name": "facebook/convnext-base-224",
            "image_size": 224
        },
        {
            "name": "Swin",
            "model_name": "microsoft/swin-base-patch4-window7-224",
            "image_size": 224
        }
    ]
    
    # Download sample image
    print("📸 Downloading sample image...")
    image = download_sample_image()
    print(f"   Image size: {image.size}")
    
    # Create sample target
    target = create_sample_target()
    
    for model_info in models_to_test:
        print(f"\n🚀 Testing {model_info['name']} Model")
        print("=" * 50)
        
        # Initialize model configuration
        model_config = ClassificationModelConfig(
            model_name=model_info["model_name"],
            num_classes=1000,  # ImageNet classes
            image_size=model_info["image_size"]
        )
        
        # Initialize the classification model
        model = ClassificationModel(model_config)
        model.eval()
        
        # Initialize adapters
        print(f"🔧 Initializing {model_info['name']} adapters...")
        input_adapter = get_adapter(model_info["model_name"], image_size=model_info["image_size"])
        output_adapter = get_output_adapter(model_info["model_name"])
        
        # Process image and target using adapter
        print("⚙️ Processing image with adapter...")
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
                print(f"   Predicted class: {prediction['labels'].item()}")
                print(f"   Confidence: {prediction['scores'].max().item():.3f}")
                
                # Show top 3 predictions
                top_scores, top_indices = torch.topk(prediction['scores'], 3)
                for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
                    print(f"   Top {i+1}: Class {idx.item()}, Score {score.item():.3f}")
            else:
                print("   No predictions generated")
        
        # Visualize results
        print("🎨 Visualizing results...")
        if predictions:
            visualize_predictions(image, predictions[0], [], f"{model_info['name']} Predictions")
        
        print(f"✅ {model_info['name']} example completed successfully!")

def main():
    """Main function demonstrating classification usage."""
    print("🚀 Classification Adapter Examples")
    print("=" * 60)
    
    # Test different models
    test_different_models()
    
    print("\n🎉 All classification examples completed successfully!")

if __name__ == "__main__":
    main() 
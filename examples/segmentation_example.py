#!/usr/bin/env python3
"""
Example script demonstrating how to use the improved segmentation adapters.

This script shows:
1. How to load and configure different segmentation models (SegFormer, DeepLabV3, Mask2Former)
2. How to use the segmentation adapters for data preprocessing
3. How to perform inference with the models
4. How to visualize results
"""

import torch
import yaml
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# Import our custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tasks.semantic_segmentation.adapters import SegFormerAdapter, DeepLabV3Adapter, get_adapter, get_output_adapter
from tasks.semantic_segmentation.model import SemanticSegmentationModel, SemanticSegmentationModelConfig

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
    # Create a simple mask for demonstration
    mask = np.zeros((512, 512), dtype=np.uint8)
    # Add some regions to the mask
    mask[100:300, 100:400] = 1  # Background
    mask[200:250, 200:350] = 2  # Object
    
    return {
        "semantic_masks": torch.as_tensor(mask, dtype=torch.long),
        "image_id": torch.tensor([0])
    }

def visualize_segmentation(image: Image.Image, predictions: dict, title: str = "Segmentation Predictions"):
    """Visualize segmentation predictions on the image."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show predicted masks
    if "masks" in predictions:
        masks = predictions["masks"]
        if masks.dim() > 2:
            # Multi-class segmentation
            predicted_mask = masks.argmax(dim=0)
        else:
            # Single mask
            predicted_mask = masks
        
        axes[1].imshow(predicted_mask, cmap='tab10')
        axes[1].set_title("Predicted Segmentation")
        axes[1].axis('off')
        
        # Show overlay
        axes[2].imshow(image)
        axes[2].imshow(predicted_mask, alpha=0.5, cmap='tab10')
        axes[2].set_title("Overlay")
        axes[2].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No masks available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("No Masks")
        axes[1].axis('off')
        
        axes[2].text(0.5, 0.5, 'No overlay available', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("No Overlay")
        axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def test_different_models():
    """Test different segmentation models with their adapters."""
    models_to_test = [
        {
            "name": "SegFormer",
            "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
            "image_size": 512,
            "segmentation_type": "semantic"
        },
        {
            "name": "DeepLabV3",
            "model_name": "microsoft/deeplabv3-resnet-50",
            "image_size": 512,
            "segmentation_type": "semantic"
        },
        {
            "name": "Mask2Former",
            "model_name": "facebook/mask2former-swin-base-coco-instance",
            "image_size": 512,
            "segmentation_type": "instance"
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
        model_config = SemanticSegmentationModelConfig(
            model_name=model_info["model_name"],
            num_classes=150,  # ADE20K classes for semantic, COCO for instance
            segmentation_type=model_info["segmentation_type"],
            image_size=model_info["image_size"]
        )
        
        # Initialize the segmentation model
        model = SemanticSegmentationModel(model_config)
        model.eval()
        
        # Initialize adapters
        print(f"🔧 Initializing {model_info['name']} adapters...")
        input_adapter = get_adapter(model_info["model_name"], image_size=model_info["image_size"])
        output_adapter = get_output_adapter(model_info["model_name"], model_info["segmentation_type"])
        
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
                print(f"   Generated predictions with keys: {list(prediction.keys())}")
                
                if "masks" in prediction:
                    masks = prediction["masks"]
                    print(f"   Mask shape: {masks.shape}")
                    print(f"   Number of classes: {masks.shape[0] if masks.dim() > 2 else 1}")
                
                if "labels" in prediction:
                    labels = prediction["labels"]
                    print(f"   Predicted labels: {labels.unique().tolist()}")
            else:
                print("   No predictions generated")
        
        # Visualize results
        print("🎨 Visualizing results...")
        if predictions:
            visualize_segmentation(image, predictions[0], f"{model_info['name']} Predictions")
        
        print(f"✅ {model_info['name']} example completed successfully!")

def main():
    """Main function demonstrating segmentation usage."""
    print("🚀 Segmentation Adapter Examples")
    print("=" * 60)
    
    # Test different models
    test_different_models()
    
    print("\n🎉 All segmentation examples completed successfully!")

if __name__ == "__main__":
    main() 
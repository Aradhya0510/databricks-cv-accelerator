"""
Batch inference module for the Databricks Computer Vision Pipeline.

This module provides utilities for running batch inference on trained models
using MLflow model artifacts.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import albumentations as A
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.utils.config import load_config

class BatchInference:
    def __init__(self, model_uri: str, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model from MLflow
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = A.Compose([
            A.Resize(config["data"]["image_size"], config["data"]["image_size"]),
            A.Normalize(mean=config["data"]["mean"], std=config["data"]["std"]),
            ToTensorV2(),
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image."""
        transformed = self.transform(image=image)
        return transformed["image"].unsqueeze(0).to(self.device)
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Make prediction for a single image."""
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
            return {
                "class_id": prediction.item(),
                "probabilities": probabilities[0].cpu().numpy().tolist()
            }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Make predictions for a batch of images."""
        return [self.predict(image) for image in images]

def main():
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model-uri", type=str, required=True, help="MLflow model URI")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with images")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSON file for predictions")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize inference
    inference = BatchInference(args.model_uri, config)
    
    # Process images
    predictions = []
    for image_file in os.listdir(args.input_dir):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(args.input_dir, image_file)
            image = np.array(Image.open(image_path).convert("RGB"))
            
            prediction = inference.predict(image)
            prediction["image_file"] = image_file
            predictions.append(prediction)
    
    # Save predictions
    with open(args.output_file, "w") as f:
        json.dump(predictions, f, indent=2)

if __name__ == "__main__":
    main() 
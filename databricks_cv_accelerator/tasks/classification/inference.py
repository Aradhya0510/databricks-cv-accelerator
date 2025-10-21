import os
import yaml
import mlflow
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from .model import ClassificationModel
from .data import ClassificationDataModule

class ClassificationInference:
    """Inference class for classification models."""
    
    def __init__(self, config: Dict):
        """Initialize the inference class.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path: str) -> None:
        """Load the model from the given path.
        
        Args:
            model_path: Path to the model checkpoint
        """
        self.model = ClassificationModel.load_from_checkpoint(
            model_path,
            config=self.config
        )
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess a single image for inference.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Get preprocessing transforms from data module
        data_module = ClassificationDataModule(self.config)
        transform = data_module.get_transform("test")
        
        # Apply transforms
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
        
    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        return_probs: bool = False
    ) -> Union[int, Tuple[int, np.ndarray]]:
        """Make prediction for a single image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_probs: Whether to return class probabilities
            
        Returns:
            Predicted class index and optionally probabilities
        """
        if self.model is None:
            raise ValueError("Model must be loaded before inference")
            
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1)
            
        if return_probs:
            return pred.item(), probs[0].cpu().numpy()
        return pred.item()
    
    def predict_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        batch_size: int = 32,
        return_probs: bool = False
    ) -> Union[List[int], Tuple[List[int], List[np.ndarray]]]:
        """Make predictions for a batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            return_probs: Whether to return class probabilities
            
        Returns:
            List of predicted class indices and optionally probabilities
        """
        if self.model is None:
            raise ValueError("Model must be loaded before inference")
            
        # Get preprocessing transforms
        data_module = ClassificationDataModule(self.config)
        transform = data_module.get_transform("test")
        
        all_preds = []
        all_probs = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                batch_tensors.append(transform(img))
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            if return_probs:
                all_probs.extend(probs.cpu().numpy())
        
        if return_probs:
            return all_preds, all_probs
        return all_preds
    
    def visualize_prediction(
        self,
        image: Union[str, Image.Image, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize model prediction for a single image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            save_path: Optional path to save visualization
        """
        # Get prediction and probabilities
        pred_idx, probs = self.predict(image, return_probs=True)
        
        # Get class names
        data_module = ClassificationDataModule(self.config)
        class_names = data_module.class_names
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Plot image
        plt.subplot(1, 2, 1)
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        plt.imshow(img)
        plt.title(f"Predicted: {class_names[pred_idx]}")
        plt.axis("off")
        
        # Plot probabilities
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(class_names))
        plt.barh(y_pos, probs)
        plt.yticks(y_pos, class_names)
        plt.xlabel("Probability")
        plt.title("Class Probabilities")
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def main():
    """Main function to run inference."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    inference = ClassificationInference(config)
    
    # Load model
    inference.load_model(args.model_path)
    
    # Get image paths
    image_paths = []
    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    else:
        image_paths = [
            str(p) for p in Path(args.image_path).glob("**/*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    
    # Process images
    for image_path in image_paths:
        # Get output path
        output_path = os.path.join(
            args.output_dir,
            f"{Path(image_path).stem}_prediction.png"
        )
        
        # Visualize prediction
        inference.visualize_prediction(image_path, output_path)
        
        print(f"Processed {image_path} -> {output_path}")

if __name__ == "__main__":
    main() 
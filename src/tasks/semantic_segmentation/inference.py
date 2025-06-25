import os
import yaml
import mlflow
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .model import SemanticSegmentationModel

class SemanticSegmentationInference:
    def __init__(self, model_path: str, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = SemanticSegmentationModel.load_from_checkpoint(model_path, config=self.config)
        self.model.eval()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Load class names
        self.class_names = self._load_class_names()
        
        # Create colormap for visualization
        self.colormap = self._create_colormap()
    
    def _load_class_names(self) -> List[str]:
        """Load COCO class names."""
        # You can replace this with your own class names file
        return [
            'background',  # Add background class
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def _create_colormap(self) -> ListedColormap:
        """Create a colormap for visualization."""
        # Generate random colors for each class
        np.random.seed(42)
        colors = np.random.rand(self.config['model']['num_classes'], 3)
        colors[0] = [0, 0, 0]  # Background is black
        
        return ListedColormap(colors)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize
        image = cv2.resize(image, (self.config['data']['image_size'], self.config['data']['image_size']))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(self.config['data']['mean'])) / np.array(self.config['data']['std'])
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.to(self.device)
    
    def postprocess_predictions(
        self,
        predictions: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Convert model predictions to segmentation mask."""
        # Get class predictions
        predictions = torch.argmax(predictions, dim=1)
        
        # Convert to numpy
        mask = predictions.cpu().numpy()[0]
        
        # Resize to original size
        mask = cv2.resize(
            mask,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        return mask
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single image."""
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Postprocess predictions
        mask = self.postprocess_predictions(outputs["logits"], image.shape[:2])
        
        return mask
    
    def visualize(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_path: str = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Visualize segmentation mask on the image."""
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot original image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Plot segmentation mask
        plt.imshow(mask, cmap=self.colormap, alpha=alpha)
        
        # Add legend
        patches = []
        for i, class_name in enumerate(self.class_names):
            if i == 0:  # Skip background
                continue
            color = self.colormap.colors[i]
            patch = plt.Rectangle((0, 0), 1, 1, fc=color)
            patches.append((patch, class_name))
        
        plt.legend(
            [p[0] for p in patches],
            [p[1] for p in patches],
            loc='upper right',
            bbox_to_anchor=(1.3, 1)
        )
        
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            return None
        
        return plt.gcf()

def batch_inference(
    model_path: str,
    config_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: int = 8
):
    """Run batch inference on a directory of images."""
    # Initialize inference
    inference = SemanticSegmentationInference(model_path, config_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = list(Path(input_dir).glob('*.jpg')) + list(Path(input_dir).glob('*.png'))
    
    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        
        for image_file in batch_files:
            # Load image
            image = cv2.imread(str(image_file))
            
            # Run inference
            mask = inference.predict(image)
            
            # Visualize and save
            output_path = os.path.join(output_dir, f'{image_file.stem}_pred.jpg')
            inference.visualize(image, mask, output_path)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        batch_inference(args.model, args.config, args.input, args.output, args.batch_size)
    else:
        # Single image inference
        inference = SemanticSegmentationInference(args.model, args.config)
        image = cv2.imread(args.input)
        mask = inference.predict(image)
        inference.visualize(image, mask, args.output) 
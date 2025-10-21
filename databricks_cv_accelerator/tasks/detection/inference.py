import os
import yaml
import mlflow
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .model import DetectionModel

class DetectionInference:
    def __init__(self, model_path: str, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = DetectionModel.load_from_checkpoint(model_path, config=self.config)
        self.model.eval()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Load class names
        self.class_names = self._load_class_names()
    
    def _load_class_names(self) -> List[str]:
        """Load COCO class names."""
        # You can replace this with your own class names file
        return [
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
    ) -> List[Dict[str, Any]]:
        """Convert model predictions to bounding boxes and scores."""
        # Convert predictions to numpy
        predictions = predictions.cpu().numpy()
        
        # Scale boxes to original image size
        h, w = original_size
        scale_h = h / self.config['data']['image_size']
        scale_w = w / self.config['data']['image_size']
        
        results = []
        for pred in predictions:
            x1, y1, x2, y2, score, class_id = pred
            
            # Scale coordinates
            x1 *= scale_w
            y1 *= scale_h
            x2 *= scale_w
            y2 *= scale_h
            
            results.append({
                'box': [x1, y1, x2, y2],
                'score': score,
                'class_id': int(class_id),
                'class_name': self.class_names[int(class_id)]
            })
        
        return results
    
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on a single image."""
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Postprocess predictions
        results = self.postprocess_predictions(predictions, image.shape[:2])
        
        return results
    
    def visualize(
        self,
        image: np.ndarray,
        predictions: List[Dict[str, Any]],
        output_path: str = None
    ) -> np.ndarray:
        """Visualize predictions on the image."""
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Plot predictions
        for pred in predictions:
            box = pred['box']
            score = pred['score']
            class_name = pred['class_name']
            
            # Create rectangle
            rect = Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(
                box[0],
                box[1] - 5,
                f'{class_name}: {score:.2f}',
                color='red',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7)
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
    inference = DetectionInference(model_path, config_path)
    
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
            predictions = inference.predict(image)
            
            # Visualize and save
            output_path = os.path.join(output_dir, f'{image_file.stem}_pred.jpg')
            inference.visualize(image, predictions, output_path)

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
        inference = DetectionInference(args.model, args.config)
        image = cv2.imread(args.input)
        predictions = inference.predict(image)
        
        if args.output:
            inference.visualize(image, predictions, args.output)
        else:
            inference.visualize(image, predictions)
            plt.show() 
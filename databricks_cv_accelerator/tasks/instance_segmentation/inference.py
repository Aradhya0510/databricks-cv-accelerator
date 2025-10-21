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
import matplotlib.patches as patches

from .model import InstanceSegmentationModel

class InstanceSegmentationInference:
    def __init__(self, model_path: str, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = InstanceSegmentationModel.load_from_checkpoint(model_path, config=self.config)
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
        outputs: Dict[str, torch.Tensor],
        original_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Convert model predictions to instance segmentation results."""
        # Extract predictions
        pred_boxes = outputs.get("pred_boxes", torch.empty(0, 4))
        pred_masks = outputs.get("pred_masks", torch.empty(0, 1, 512, 512))
        pred_labels = outputs.get("pred_labels", torch.empty(0))
        pred_scores = outputs.get("pred_scores", torch.ones(pred_boxes.size(0)))
        
        # Convert to numpy
        pred_boxes = pred_boxes.cpu().numpy()
        pred_masks = pred_masks.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        
        # Resize masks to original size
        resized_masks = []
        for mask in pred_masks:
            mask = cv2.resize(
                mask[0],
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
            resized_masks.append(mask)
        
        # Scale bounding boxes to original size
        if len(pred_boxes) > 0:
            scale_x = original_size[1] / self.config['data']['image_size']
            scale_y = original_size[0] / self.config['data']['image_size']
            pred_boxes[:, [0, 2]] *= scale_x
            pred_boxes[:, [1, 3]] *= scale_y
        
        return {
            'boxes': pred_boxes,
            'masks': np.array(resized_masks),
            'labels': pred_labels,
            'scores': pred_scores
        }
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run inference on a single image."""
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Postprocess predictions
        results = self.postprocess_predictions(outputs, image.shape[:2])
        
        return results
    
    def visualize(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        output_path: str = None,
        alpha: float = 0.5,
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """Visualize instance segmentation results on the image."""
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot original image
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Filter predictions by confidence
        boxes = results['boxes']
        masks = results['masks']
        labels = results['labels']
        scores = results['scores']
        
        valid_indices = scores >= confidence_threshold
        boxes = boxes[valid_indices]
        masks = masks[valid_indices]
        labels = labels[valid_indices]
        scores = scores[valid_indices]
        
        # Plot masks
        for i, (box, mask, label, score) in enumerate(zip(boxes, masks, labels, scores)):
            if label == 0:  # Skip background
                continue
            
            # Create colored mask
            color = self.colormap.colors[label]
            colored_mask = np.zeros((*mask.shape, 3))
            colored_mask[mask > 0] = color
            
            # Overlay mask
            ax.imshow(colored_mask, alpha=alpha)
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            class_name = self.class_names[label]
            ax.text(
                x1, y1 - 5,
                f'{class_name}: {score:.2f}',
                color='white',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
            )
        
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            return None
        
        return fig
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        image_id: str = "image"
    ):
        """Save results in COCO format."""
        # Convert to COCO format
        coco_results = []
        
        for i, (box, mask, label, score) in enumerate(zip(
            results['boxes'], results['masks'], results['labels'], results['scores']
        )):
            if label == 0:  # Skip background
                continue
            
            # Convert mask to polygon
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Get the largest contour
                contour = max(contours, key=cv2.contourArea)
                polygon = contour.flatten().tolist()
                
                coco_result = {
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                    'segmentation': [polygon],
                    'score': float(score)
                }
                coco_results.append(coco_result)
        
        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(coco_results, f, indent=2)

def batch_inference(
    model_path: str,
    config_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: int = 8
):
    """Run batch inference on a directory of images."""
    # Initialize inference
    inference = InstanceSegmentationInference(model_path, config_path)
    
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
            results = inference.predict(image)
            
            # Visualize and save
            output_path = os.path.join(output_dir, f'{image_file.stem}_pred.jpg')
            inference.visualize(image, results, output_path)
            
            # Save results in COCO format
            results_path = os.path.join(output_dir, f'{image_file.stem}_results.json')
            inference.save_results(results, results_path, image_file.stem)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        batch_inference(args.model, args.config, args.input, args.output, args.batch_size)
    else:
        # Single image inference
        inference = InstanceSegmentationInference(args.model, args.config)
        
        # Load image
        image = cv2.imread(args.input)
        
        # Run inference
        results = inference.predict(image)
        
        # Visualize and save
        if args.output:
            inference.visualize(image, results, args.output, confidence_threshold=args.confidence_threshold)
        else:
            inference.visualize(image, results, confidence_threshold=args.confidence_threshold)
            plt.show() 
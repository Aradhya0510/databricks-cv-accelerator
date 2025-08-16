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

from .model import UniversalSegmentationModel

class UniversalSegmentationInference:
    def __init__(self, model_path: str, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = UniversalSegmentationModel.load_from_checkpoint(model_path, config=self.config)
        self.model.eval()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Load class names
        self.class_names = self._load_class_names()
        
        # Create colormap for visualization
        self.colormap = self._create_colormap()
    
    def _load_class_names(self) -> List[str]:
        """Load COCO panoptic class names (things + stuff)."""
        # COCO panoptic classes: 80 things + 53 stuff = 133 classes
        thing_classes = [
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
        
        stuff_classes = [
            'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
            'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
            'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting',
            'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
            'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing',
            'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers',
            'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator',
            'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
            'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
            'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm',
            'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
            'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
            'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track',
            'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
            'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt',
            'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket',
            'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
            'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle',
            'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
            'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
            'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
            'clock', 'flag'
        ]
        
        return thing_classes + stuff_classes
    
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
        """Convert model predictions to panoptic segmentation results."""
        # Extract predictions
        pred_panoptic = outputs.get("pred_panoptic_seg", torch.zeros(1, original_size[0], original_size[1]))
        pred_boxes = outputs.get("pred_boxes", torch.empty(0, 4))
        pred_labels = outputs.get("pred_labels", torch.empty(0))
        pred_scores = outputs.get("pred_scores", torch.ones(pred_boxes.size(0)))
        
        # Convert to numpy
        pred_panoptic = pred_panoptic.cpu().numpy()[0]  # Remove batch dimension
        pred_boxes = pred_boxes.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        
        # Resize panoptic mask to original size
        pred_panoptic = cv2.resize(
            pred_panoptic,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Scale bounding boxes to original size
        if len(pred_boxes) > 0:
            scale_x = original_size[1] / self.config['data']['image_size']
            scale_y = original_size[0] / self.config['data']['image_size']
            pred_boxes[:, [0, 2]] *= scale_x
            pred_boxes[:, [1, 3]] *= scale_y
        
        return {
            'panoptic_mask': pred_panoptic,
            'boxes': pred_boxes,
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
        """Visualize panoptic segmentation results on the image."""
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot original image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot panoptic segmentation
        ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Create colored panoptic mask
        panoptic_mask = results['panoptic_mask']
        colored_mask = np.zeros((*panoptic_mask.shape, 3))
        
        # Color each instance/class
        unique_ids = np.unique(panoptic_mask)
        for instance_id in unique_ids:
            if instance_id == 0:  # Skip background
                continue
            
            # Get class ID from instance ID (assuming instance_id encodes class info)
            class_id = instance_id % len(self.class_names)
            if class_id >= len(self.class_names):
                class_id = 1  # Default to person if out of range
            
            color = self.colormap.colors[class_id]
            colored_mask[panoptic_mask == instance_id] = color
        
        # Overlay panoptic mask
        ax2.imshow(colored_mask, alpha=alpha)
        
        # Draw bounding boxes if available
        boxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
        
        if len(boxes) > 0:
            valid_indices = scores >= confidence_threshold
            boxes = boxes[valid_indices]
            labels = labels[valid_indices]
            scores = scores[valid_indices]
            
            for box, label, score in zip(boxes, labels, scores):
                if label >= len(self.class_names):
                    continue
                
                x1, y1, x2, y2 = box
                color = self.colormap.colors[label]
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none'
                )
                ax2.add_patch(rect)
                
                # Add label
                class_name = self.class_names[label]
                ax2.text(
                    x1, y1 - 5,
                    f'{class_name}: {score:.2f}',
                    color='white',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
                )
        
        ax2.set_title('Panoptic Segmentation')
        ax2.axis('off')
        
        plt.tight_layout()
        
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
        """Save results in COCO panoptic format."""
        # Convert to COCO panoptic format
        panoptic_results = {
            'image_id': image_id,
            'file_name': f'{image_id}_panoptic.png',
            'segments_info': []
        }
        
        panoptic_mask = results['panoptic_mask']
        unique_ids = np.unique(panoptic_mask)
        
        for instance_id in unique_ids:
            if instance_id == 0:  # Skip background
                continue
            
            # Get class ID from instance ID
            class_id = instance_id % len(self.class_names)
            if class_id >= len(self.class_names):
                class_id = 1  # Default to person if out of range
            
            # Create binary mask for this instance
            instance_mask = (panoptic_mask == instance_id).astype(np.uint8)
            
            # Convert mask to polygon
            contours, _ = cv2.findContours(
                instance_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Get the largest contour
                contour = max(contours, key=cv2.contourArea)
                polygon = contour.flatten().tolist()
                
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Determine if it's a thing or stuff class
                is_thing = class_id < 80  # First 80 classes are things
                
                segment_info = {
                    'id': int(instance_id),
                    'category_id': int(class_id),
                    'area': float(area),
                    'iscrowd': 0,
                    'bbox': cv2.boundingRect(contour)
                }
                
                panoptic_results['segments_info'].append(segment_info)
        
        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(panoptic_results, f, indent=2)
        
        # Save panoptic mask as image
        mask_path = output_path.replace('.json', '_mask.png')
        cv2.imwrite(mask_path, panoptic_mask.astype(np.uint8))

def batch_inference(
    model_path: str,
    config_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: int = 8
):
    """Run batch inference on a directory of images."""
    # Initialize inference
    inference = UniversalSegmentationInference(model_path, config_path)
    
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
            
            # Save results in COCO panoptic format
            results_path = os.path.join(output_dir, f'{image_file.stem}_panoptic.json')
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
        inference = UniversalSegmentationInference(args.model, args.config)
        
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
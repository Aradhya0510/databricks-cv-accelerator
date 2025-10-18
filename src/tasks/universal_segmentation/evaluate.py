import os
import yaml
import mlflow
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MulticlassJaccardIndex
# Import Dice with version compatibility
try:
    from torchmetrics import Dice
except ImportError:
    try:
        from torchmetrics.classification import Dice
    except ImportError:
        # Dice not available, will use JaccardIndex if needed
        Dice = None
from torchmetrics.detection import MeanAveragePrecision

from .model import UniversalSegmentationModel
from .data import UniversalSegmentationDataModule

class UniversalSegmentationEvaluator:
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
        
        # Initialize metrics
        self.num_classes = len(self.class_names)
        self.iou_metric = MulticlassJaccardIndex(num_classes=self.num_classes, ignore_index=0)
        self.dice_metric = Dice(num_classes=self.num_classes, ignore_index=0)
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="segm",
            class_metrics=True
        )
    
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
    
    def evaluate(self, data_module: UniversalSegmentationDataModule) -> Dict[str, Any]:
        """Evaluate model on validation dataset."""
        # Initialize metrics
        self.iou_metric = self.iou_metric.to(self.device)
        self.dice_metric = self.dice_metric.to(self.device)
        self.map_metric = self.map_metric.to(self.device)
        
        # Initialize results
        per_class_iou = torch.zeros(self.num_classes, device=self.device)
        per_class_dice = torch.zeros(self.num_classes, device=self.device)
        class_counts = torch.zeros(self.num_classes, device=self.device)
        
        # Run evaluation
        with torch.no_grad():
            for batch in data_module.val_dataloader():
                images = batch["pixel_values"].to(self.device)
                labels = batch["labels"]
                
                # Get predictions
                outputs = self.model(images)
                
                # Extract panoptic predictions
                pred_panoptic = outputs.get("pred_panoptic_seg", torch.zeros_like(images[:, :1]))
                pred_boxes = outputs.get("pred_boxes", torch.empty(0, 4))
                pred_labels = outputs.get("pred_labels", torch.empty(0))
                pred_scores = outputs.get("pred_scores", torch.ones(pred_boxes.size(0)))
                
                target_panoptic = labels.get("panoptic_masks", torch.zeros_like(images[:, :1]))
                target_boxes = labels.get("bounding_boxes", torch.empty(0, 4))
                target_labels = labels.get("class_labels", torch.empty(0))
                
                # Update segmentation metrics
                if pred_panoptic.numel() > 0 and target_panoptic.numel() > 0:
                    # Flatten masks for metric computation
                    pred_panoptic_flat = pred_panoptic.view(pred_panoptic.size(0), -1)
                    target_panoptic_flat = target_panoptic.view(target_panoptic.size(0), -1)
                    
                    self.iou_metric.update(pred_panoptic_flat, target_panoptic_flat)
                    self.dice_metric.update(pred_panoptic_flat, target_panoptic_flat)
                
                # Update detection metrics (mAP)
                if pred_boxes.numel() > 0 and target_boxes.numel() > 0:
                    # Format predictions for mAP metric
                    preds = [{
                        "boxes": pred_boxes,
                        "masks": pred_panoptic,
                        "scores": pred_scores,
                        "labels": pred_labels
                    }]
                    
                    targets = [{
                        "boxes": target_boxes,
                        "masks": target_panoptic,
                        "labels": target_labels
                    }]
                    
                    self.map_metric.update(preds, targets)
                
                # Calculate per-class metrics
                for class_idx in range(self.num_classes):
                    if class_idx == 0:  # Skip background
                        continue
                    
                    # Get class-specific masks
                    pred_class = (pred_panoptic == class_idx)
                    true_class = (target_panoptic == class_idx)
                    
                    # Calculate IoU
                    intersection = (pred_class & true_class).sum().float()
                    union = (pred_class | true_class).sum().float()
                    if union > 0:
                        per_class_iou[class_idx] += intersection / union
                    
                    # Calculate Dice
                    if (pred_class.sum() + true_class.sum()) > 0:
                        per_class_dice[class_idx] += (2 * intersection) / (pred_class.sum() + true_class.sum())
                    
                    # Update class count
                    if true_class.sum() > 0:
                        class_counts[class_idx] += 1
        
        # Calculate final metrics
        overall_iou = self.iou_metric.compute()
        overall_dice = self.dice_metric.compute()
        overall_map = self.map_metric.compute()
        
        # Calculate per-class metrics
        per_class_metrics = []
        for i, class_name in enumerate(self.class_names):
            if i == 0:  # Skip background
                continue
            
            if class_counts[i] > 0:
                class_metrics = {
                    'class_name': class_name,
                    'IoU': (per_class_iou[i] / class_counts[i]).item(),
                    'Dice': (per_class_dice[i] / class_counts[i]).item(),
                    'count': class_counts[i].item()
                }
                per_class_metrics.append(class_metrics)
        
        return {
            'overall_metrics': {
                'IoU': overall_iou.item(),
                'Dice': overall_dice.item(),
                'mAP': overall_map['map'].item() if isinstance(overall_map, dict) else overall_map.item()
            },
            'per_class_metrics': per_class_metrics
        }
    
    def plot_metrics(self, metrics: Dict[str, Any], output_dir: str = None):
        """Plot evaluation metrics."""
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot overall metrics
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=list(metrics['overall_metrics'].keys()),
            y=list(metrics['overall_metrics'].values())
        )
        plt.title('Overall Panoptic Segmentation Metrics')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'overall_metrics.png'))
            plt.close()
        else:
            plt.show()
        
        # Plot per-class metrics
        per_class_df = pd.DataFrame(metrics['per_class_metrics'])
        
        plt.figure(figsize=(15, 8))
        sns.barplot(
            data=per_class_df.melt(id_vars=['class_name', 'count']),
            x='class_name',
            y='value',
            hue='variable'
        )
        plt.title('Per-Class Panoptic Segmentation Metrics')
        plt.xticks(rotation=90)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'))
            plt.close()
        else:
            plt.show()
        
        # Plot class distribution
        plt.figure(figsize=(15, 6))
        sns.barplot(
            data=per_class_df,
            x='class_name',
            y='count'
        )
        plt.title('Class Distribution in Validation Set')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
            plt.close()
        else:
            plt.show()

def evaluate_model(
    model_path: str,
    config_path: str,
    output_dir: str = None
):
    """Evaluate model and generate reports."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data module
            data_module = UniversalSegmentationDataModule(config['data'])
    data_module.setup()
    
    # Initialize evaluator
    evaluator = UniversalSegmentationEvaluator(model_path, config_path)
    
    # Run evaluation
    metrics = evaluator.evaluate(data_module)
    
    # Plot metrics
    if output_dir:
        evaluator.plot_metrics(metrics, output_dir)
    
    # Log metrics to MLflow
    mlflow.log_metrics(metrics['overall_metrics'])
    
    # Save metrics to file
    if output_dir:
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
    
    return metrics 
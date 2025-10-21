import os
import yaml
import mlflow
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import seaborn as sns

from .model import DetectionModel
from .data import DetectionDataModule

class DetectionEvaluator:
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
        
        # Initialize COCO evaluator
        self.coco_gt = COCO(self.config['data']['val_annotations'])
    
    def _load_class_names(self) -> List[str]:
        """Load COCO class names."""
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
    
    def evaluate(self, data_module: DetectionDataModule) -> Dict[str, Any]:
        """Evaluate model on validation dataset."""
        # Initialize results
        results = []
        
        # Run evaluation
        with torch.no_grad():
            for batch in data_module.val_dataloader():
                images, targets = batch
                images = images.to(self.device)
                
                # Get predictions
                predictions = self.model(images)
                
                # Convert predictions to COCO format
                for pred, target in zip(predictions, targets):
                    image_id = target['image_id'].item()
                    
                    for box, score, class_id in zip(
                        pred['boxes'].cpu().numpy(),
                        pred['scores'].cpu().numpy(),
                        pred['labels'].cpu().numpy()
                    ):
                        # Convert to COCO format [x, y, width, height]
                        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                        
                        results.append({
                            'image_id': image_id,
                            'category_id': int(class_id),
                            'bbox': [x, y, w, h],
                            'score': float(score)
                        })
        
        # Convert results to COCO format
        coco_dt = self.coco_gt.loadRes(results)
        
        # Run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Get metrics
        metrics = {
            'mAP': coco_eval.stats[0],
            'mAP_50': coco_eval.stats[1],
            'mAP_75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5]
        }
        
        # Get per-class metrics
        per_class_metrics = []
        for i, class_name in enumerate(self.class_names):
            if i == 0:  # Skip background
                continue
            
            # Get class-specific metrics
            class_metrics = {
                'class_name': class_name,
                'AP': coco_eval.eval['precision'][0, :, i, 0, 2].mean(),
                'AP_50': coco_eval.eval['precision'][0, :, i, 0, 0].mean(),
                'AP_75': coco_eval.eval['precision'][0, :, i, 0, 1].mean()
            }
            per_class_metrics.append(class_metrics)
        
        return {
            'overall_metrics': metrics,
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
        plt.title('Overall Detection Metrics')
        plt.xticks(rotation=45)
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
            data=per_class_df.melt(id_vars=['class_name']),
            x='class_name',
            y='value',
            hue='variable'
        )
        plt.title('Per-Class Detection Metrics')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'))
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
    data_module = DetectionDataModule(config)
    
    # Initialize evaluator
    evaluator = DetectionEvaluator(model_path, config_path)
    
    # Run evaluation
    metrics = evaluator.evaluate(data_module)
    
    # Plot metrics
    evaluator.plot_metrics(metrics, output_dir)
    
    # Print metrics
    print("\nOverall Metrics:")
    for metric, value in metrics['overall_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPer-Class Metrics:")
    per_class_df = pd.DataFrame(metrics['per_class_metrics'])
    print(per_class_df.to_string(index=False))
    
    # Save metrics to file
    if output_dir:
        # Save overall metrics
        pd.DataFrame([metrics['overall_metrics']]).to_csv(
            os.path.join(output_dir, 'overall_metrics.csv'),
            index=False
        )
        
        # Save per-class metrics
        per_class_df.to_csv(
            os.path.join(output_dir, 'per_class_metrics.csv'),
            index=False
        )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, help='Path to output directory')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.config, args.output) 
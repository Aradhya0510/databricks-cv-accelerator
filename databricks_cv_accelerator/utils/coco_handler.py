"""
COCO Handler for managing COCO format data.

This module provides utilities for handling COCO format annotations and
converting between different bounding box formats.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

class COCOHandler:
    """Handler for COCO format data."""
    
    def __init__(self, annotation_file: Union[str, Path]):
        """
        Initialize the COCO handler.
        
        Args:
            annotation_file: Path to COCO format annotation file
        """
        self.annotation_file = Path(annotation_file)
        
        # Load COCO annotations
        with open(self.annotation_file, "r") as f:
            self.coco_data = json.load(f)
            
        # Create image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        # Create category id to name mapping
        self.cat_id_to_name = {
            cat["id"]: cat["name"] 
            for cat in self.coco_data["categories"]
        }
        
    def get_image_info(self, image_id: int) -> Dict:
        """Get image information from COCO data."""
        for img in self.coco_data["images"]:
            if img["id"] == image_id:
                return img
        raise ValueError(f"Image ID {image_id} not found in COCO data")
        
    def get_annotations(self, image_id: int) -> List[Dict]:
        """Get annotations for an image."""
        return self.img_to_anns.get(image_id, [])
        
    def get_category_name(self, category_id: int) -> str:
        """Get category name from category ID."""
        return self.cat_id_to_name.get(category_id, f"unknown_{category_id}")
    
    def convert_bbox_to_xyxy(self, bbox: List[float]) -> List[float]:
        """Convert COCO bbox format [x,y,width,height] to [x0,y0,x1,y1]."""
        return [
            bbox[0],
            bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3]
        ]
    
    def convert_bbox_to_xywh(self, bbox: List[float]) -> List[float]:
        """Convert [x0,y0,x1,y1] to COCO bbox format [x,y,width,height]."""
        return [
            bbox[0],
            bbox[1],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1]
        ]
    
    def prepare_target(self, annotations: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert COCO annotations to target format."""
        boxes = []
        labels = []
        
        for ann in annotations:
            # Convert bbox from [x,y,width,height] to [x0,y0,x1,y1]
            bbox = self.convert_bbox_to_xyxy(ann["bbox"])
            boxes.append(bbox)
            labels.append(ann["category_id"])
            
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    def convert_to_coco_format(
        self,
        image_id: int,
        boxes: Union[torch.Tensor, np.ndarray],
        scores: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        confidence_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Convert predictions to COCO format.
        
        Args:
            image_id: ID of the image
            boxes: Predicted boxes in [x0,y0,x1,y1] format
            scores: Confidence scores
            labels: Predicted labels
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of COCO format predictions
        """
        # Convert to numpy if tensors
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        # Filter by confidence
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Convert to COCO format
        coco_predictions = []
        for box, score, label in zip(boxes, scores, labels):
            # Convert [x0,y0,x1,y1] to [x,y,width,height]
            bbox = self.convert_bbox_to_xywh(box)
            
            coco_predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [float(x) for x in bbox],
                "score": float(score)
            })
            
        return coco_predictions
    
    def save_predictions(
        self,
        predictions: List[Dict],
        output_file: Union[str, Path]
    ):
        """
        Save predictions in COCO format.
        
        Args:
            predictions: List of COCO format predictions
            output_file: Path to save predictions
        """
        output = {
            "images": self.coco_data["images"],
            "categories": self.coco_data["categories"],
            "annotations": predictions
        }
        
        with open(output_file, "w") as f:
            json.dump(output, f)
            
    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        return [cat["name"] for cat in self.coco_data["categories"]] 
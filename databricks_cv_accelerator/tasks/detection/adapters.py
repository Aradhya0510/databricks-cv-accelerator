from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
from transformers import AutoImageProcessor
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

class BaseAdapter(ABC):
    """Base class for model-specific data adapters."""
    
    @abstractmethod
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """Process a single sample.
        
        Args:
            image: A PIL image
            target: A dictionary containing 'boxes' and 'labels'
            
        Returns:
            A tuple containing the processed image tensor and target dictionary
        """
        pass

class NoOpInputAdapter(BaseAdapter):
    """A "No-Operation" input adapter.
    - Converts image to a tensor
    - Keeps targets in the standard [x1, y1, x2, y2] absolute pixel format
    - Suitable for models like torchvision's Faster R-CNN
    """
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        target["class_labels"] = target["labels"]  # Rename for consistency
        return F.to_tensor(image), target

class DETRInputAdapter(BaseAdapter):
    """Input adapter for DETR-like models (DETR, YOLOS, etc.).
    - Converts image to a tensor
    - Converts bounding boxes from [x1, y1, x2, y2] absolute pixels to
      [cx, cy, w, h] normalized format
    - Handles image resizing according to DETR guidelines
    """
    def __init__(self, model_name: str, image_size: int = 800):
        self.image_size = image_size
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            size={"height": image_size, "width": image_size},
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            do_pad=True
        )
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # Get original image size
        w, h = image.size
        
        # Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h]
        boxes_xyxy = target["boxes"]
        if boxes_xyxy.shape[0] == 0:
            # Handle cases with no annotations
            boxes_cxcwh = torch.zeros((0, 4), dtype=torch.float32)
        else:
            # Convert to center format
            boxes_cxcwh = self._xyxy_to_cxcwh(boxes_xyxy)
        
        # Normalize boxes by original image size
        boxes_cxcwh_normalized = boxes_cxcwh / torch.tensor([w, h, w, h], dtype=torch.float32)
        
        # Update target dictionary
        adapted_target = {
            "boxes": boxes_cxcwh_normalized,
            "class_labels": target["labels"],
            "image_id": target["image_id"],
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w])  # Original size before processing
        }
        
        # Use processor for complete image preprocessing (resize, pad, normalize)
        processed = self.processor(
            image,
            return_tensors="pt",
            do_resize=True,  # Let processor handle resizing
            do_rescale=True,
            do_normalize=True,
            do_pad=True  # Enable padding to ensure consistent size
        )
        
        return processed.pixel_values.squeeze(0), adapted_target
    
    @staticmethod
    def _xyxy_to_cxcwh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """Converts [x1, y1, x2, y2] to [center_x, center_y, width, height]"""
        x1, y1, x2, y2 = boxes_xyxy.unbind(1)
        b = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
        return torch.stack(b, dim=-1)

class YOLOSInputAdapter(BaseAdapter):
    """Input adapter for YOLOS models.
    - Similar to DETR but optimized for YOLOS architecture
    - Converts image to a tensor
    - Converts bounding boxes from [x1, y1, x2, y2] absolute pixels to
      [cx, cy, w, h] normalized format
    - Handles image resizing according to YOLOS guidelines
    """
    def __init__(self, model_name: str, image_size: int = 800):
        self.image_size = image_size
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            size={"height": image_size, "width": image_size},
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            do_pad=True
        )
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # Get original image size
        w, h = image.size
        
        # Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h]
        boxes_xyxy = target["boxes"]
        if boxes_xyxy.shape[0] == 0:
            # Handle cases with no annotations
            boxes_cxcwh = torch.zeros((0, 4), dtype=torch.float32)
        else:
            # Convert to center format
            boxes_cxcwh = self._xyxy_to_cxcwh(boxes_xyxy)
        
        # Normalize boxes by original image size
        boxes_cxcwh_normalized = boxes_cxcwh / torch.tensor([w, h, w, h], dtype=torch.float32)
        
        # Update target dictionary
        adapted_target = {
            "boxes": boxes_cxcwh_normalized,
            "class_labels": target["labels"],
            "image_id": target["image_id"],
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w])  # Original size before processing
        }
        
        # Use processor for complete image preprocessing (resize, pad, normalize)
        # YOLOS doesn't require pixel_mask unlike DETR
        processed = self.processor(
            image,
            return_tensors="pt",
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            do_pad=True
        )
        
        return processed.pixel_values.squeeze(0), adapted_target
    
    @staticmethod
    def _xyxy_to_cxcwh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """Converts [x1, y1, x2, y2] to [center_x, center_y, width, height]"""
        x1, y1, x2, y2 = boxes_xyxy.unbind(1)
        b = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
        return torch.stack(b, dim=-1)

class DETROutputAdapter:
    """Adapter for DETR model outputs."""
    
    def __init__(self, model_name: str, image_size: int = 800):
        self.image_size = image_size
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            size={"height": image_size, "width": image_size}
        )
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt DETR outputs to standard format."""
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")
        
        return {
            "loss": loss,
            "pred_boxes": outputs.pred_boxes,
            "pred_logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
    
    def format_predictions(self, outputs: Dict[str, Any], batch: Optional[Dict[str, Any]] = None) -> List[Dict[str, torch.Tensor]]:
        """Format DETR outputs for metric computation."""
        detection_output = DetrObjectDetectionOutput(
            logits=outputs["pred_logits"],
            pred_boxes=outputs["pred_boxes"]
        )
        
        # Get target sizes from the batch
        target_sizes = []
        if batch and "labels" in batch:
            for target in batch["labels"]:
                if "size" in target:
                    target_sizes.append(target["size"].tolist())
                else:
                    target_sizes.append([self.image_size, self.image_size])
        else:
            # Fallback: use default size for all images
            batch_size = outputs["pred_logits"].shape[0]
            target_sizes = [[self.image_size, self.image_size]] * batch_size
        
        # Use processor's postprocessing
        processed_outputs = self.processor.post_process_object_detection(
            detection_output,
            threshold=0.7,
            target_sizes=target_sizes
        )
        
        preds = []
        for i, processed_output in enumerate(processed_outputs):
            boxes = processed_output["boxes"]
            scores = processed_output["scores"]
            labels = processed_output["labels"]
            
            # Get image_id from batch if available
            image_id = torch.tensor([i])
            if batch and "labels" in batch and i < len(batch["labels"]):
                image_id = batch["labels"][i].get("image_id", torch.tensor([i]))
            
            preds.append({
                "boxes": boxes,  # Already in [x1, y1, x2, y2] format in absolute pixels
                "scores": scores,
                "labels": labels,
                "image_id": image_id
            })
        
        return preds

    def format_targets(self, targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Format batch targets for metric computation.
        
        Converts targets from DETR format ([cx, cy, w, h] normalized) to metric format ([x1, y1, x2, y2] absolute pixels).
        
        Args:
            targets: List of target dictionaries, each containing:
                - boxes: Tensor of shape [N, 4] in [cx, cy, w, h] normalized format
                - class_labels: Tensor of shape [N] containing class indices
                - size: Tensor of shape [2] containing image size [h, w]
                - image_id: Tensor containing image ID
                
        Returns:
            List of target dictionaries for each image in the batch
        """
        formatted_targets = []
        
        for i, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["class_labels"]
            size = target["size"]
            image_id = target.get("image_id", torch.tensor([i]))
            
            # Convert from [cx, cy, w, h] normalized to [x1, y1, x2, y2] absolute pixels
            h, w = size
            boxes_xyxy = torch.zeros_like(boxes)
            boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w  # x1
            boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h  # y1
            boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w  # x2
            boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h  # y2
            
            # Create target dictionary
            formatted_target = {
                "boxes": boxes_xyxy,  # Now in [x1, y1, x2, y2] format in absolute pixels
                "labels": labels,
                "image_id": image_id
            }
            
            formatted_targets.append(formatted_target)
        
        return formatted_targets

class YOLOSOutputAdapter:
    """Adapter for YOLOS model outputs."""
    
    def __init__(self, model_name: str, image_size: int = 800):
        self.image_size = image_size
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            size={"height": image_size, "width": image_size}
        )
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt YOLOS outputs to standard format."""
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")
        
        return {
            "loss": loss,
            "pred_boxes": outputs.pred_boxes,
            "pred_logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
    
    def format_predictions(self, outputs: Dict[str, Any], batch: Optional[Dict[str, Any]] = None) -> List[Dict[str, torch.Tensor]]:
        """Format YOLOS outputs for metric computation."""
        detection_output = YolosObjectDetectionOutput(
            logits=outputs["pred_logits"],
            pred_boxes=outputs["pred_boxes"]
        )
        
        # Get target sizes from the batch
        target_sizes = []
        if batch and "labels" in batch:
            for target in batch["labels"]:
                if "size" in target:
                    target_sizes.append(target["size"].tolist())
                else:
                    target_sizes.append([self.image_size, self.image_size])
        else:
            # Fallback: use default size for all images
            batch_size = outputs["pred_logits"].shape[0]
            target_sizes = [[self.image_size, self.image_size]] * batch_size
        
        # Use processor's postprocessing
        processed_outputs = self.processor.post_process_object_detection(
            detection_output,
            threshold=0.7,
            target_sizes=target_sizes
        )
        
        preds = []
        for i, processed_output in enumerate(processed_outputs):
            boxes = processed_output["boxes"]
            scores = processed_output["scores"]
            labels = processed_output["labels"]
            
            # Get image_id from batch if available
            image_id = torch.tensor([i])
            if batch and "labels" in batch and i < len(batch["labels"]):
                image_id = batch["labels"][i].get("image_id", torch.tensor([i]))
            
            preds.append({
                "boxes": boxes,  # Already in [x1, y1, x2, y2] format in absolute pixels
                "scores": scores,
                "labels": labels,
                "image_id": image_id
            })
        
        return preds

    def format_targets(self, targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Format batch targets for metric computation.
        
        Converts targets from YOLOS format ([cx, cy, w, h] normalized) to metric format ([x1, y1, x2, y2] absolute pixels).
        
        Args:
            targets: List of target dictionaries, each containing:
                - boxes: Tensor of shape [N, 4] in [cx, cy, w, h] normalized format
                - class_labels: Tensor of shape [N] containing class indices
                - size: Tensor of shape [2] containing image size [h, w]
                - image_id: Tensor containing image ID
                
        Returns:
            List of target dictionaries for each image in the batch
        """
        formatted_targets = []
        
        for i, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["class_labels"]
            size = target["size"]
            image_id = target.get("image_id", torch.tensor([i]))
            
            # Convert from [cx, cy, w, h] normalized to [x1, y1, x2, y2] absolute pixels
            h, w = size
            boxes_xyxy = torch.zeros_like(boxes)
            boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w  # x1
            boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h  # y1
            boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w  # x2
            boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h  # y2
            
            # Create target dictionary
            formatted_target = {
                "boxes": boxes_xyxy,  # Now in [x1, y1, x2, y2] format in absolute pixels
                "labels": labels,
                "image_id": image_id
            }
            
            formatted_targets.append(formatted_target)
        
        return formatted_targets

def get_input_adapter(model_name: str, image_size: int = 800) -> BaseAdapter:
    """Get the appropriate input adapter for a model."""
    model_name_lower = model_name.lower()
    
    if "yolos" in model_name_lower:
        return YOLOSInputAdapter(model_name=model_name, image_size=image_size)
    elif "detr" in model_name_lower:
        return DETRInputAdapter(model_name=model_name, image_size=image_size)
    else:
        return NoOpInputAdapter()

def get_output_adapter(model_name: str, image_size: int = 800):
    """Get the appropriate output adapter for a model."""
    model_name_lower = model_name.lower()
    
    if "yolos" in model_name_lower:
        return YOLOSOutputAdapter(model_name=model_name, image_size=image_size)
    elif "detr" in model_name_lower:
        return DETROutputAdapter(model_name=model_name, image_size=image_size)
    else:
        # For other models, use DETR adapter as fallback
        return DETROutputAdapter(model_name=model_name, image_size=image_size)

# Keep the old function for backward compatibility
def get_adapter(model_name: str, image_size: int = 800) -> BaseAdapter:
    """Get the appropriate adapter for a model."""
    return get_input_adapter(model_name, image_size) 
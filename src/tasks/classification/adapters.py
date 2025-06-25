from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
from transformers import AutoImageProcessor
from PIL import Image
import torchvision.transforms.functional as F

class BaseAdapter(ABC):
    """Base class for model-specific data adapters."""
    
    @abstractmethod
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """Process a single sample.
        
        Args:
            image: A PIL image
            target: A dictionary containing 'class_labels'
            
        Returns:
            A tuple containing the processed image tensor and target dictionary
        """
        pass

class NoOpAdapter(BaseAdapter):
    """A "No-Operation" adapter.
    - Converts image to a tensor
    - Keeps targets in standard format
    - Suitable for models like torchvision's ResNet
    """
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        return F.to_tensor(image), target

class ViTAdapter(BaseAdapter):
    """Adapter for ViT-like models (ViT, DeiT, etc.).
    - Converts image to a tensor
    - Handles image resizing and preprocessing
    - Uses Hugging Face's AutoImageProcessor
    """
    def __init__(self, model_name: str, image_size: int = 224):
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
        # Use processor for complete image preprocessing
        processed = self.processor(
            image,
            return_tensors="pt",
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            do_pad=True
        )
        
        return processed.pixel_values.squeeze(0), target

class ConvNeXTAdapter(BaseAdapter):
    """Adapter for ConvNeXT models.
    - Similar to ViT but optimized for ConvNeXT architecture
    - Handles image resizing and preprocessing
    """
    def __init__(self, model_name: str, image_size: int = 224):
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
        # Use processor for complete image preprocessing
        processed = self.processor(
            image,
            return_tensors="pt",
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            do_pad=True
        )
        
        return processed.pixel_values.squeeze(0), target

class SwinAdapter(BaseAdapter):
    """Adapter for Swin Transformer models.
    - Handles image resizing and preprocessing for Swin architecture
    - Uses specific preprocessing for hierarchical vision transformers
    """
    def __init__(self, model_name: str, image_size: int = 224):
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
        # Use processor for complete image preprocessing
        processed = self.processor(
            image,
            return_tensors="pt",
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            do_pad=True
        )
        
        return processed.pixel_values.squeeze(0), target

class OutputAdapter(ABC):
    """Base class for model output adapters."""
    
    @abstractmethod
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt model outputs to standard format.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Adapted outputs in standard format
        """
        pass
    
    @abstractmethod
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to model-specific format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in model-specific format
        """
        pass
    
    @abstractmethod
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format model outputs for metric computation.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        pass
    
    @abstractmethod
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        pass

class ViTOutputAdapter(OutputAdapter):
    """Adapter for ViT model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt ViT outputs to standard format.
        
        Args:
            outputs: Raw ViT outputs
            
        Returns:
            Adapted outputs in standard format
        """
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")
        
        return {
            "loss": loss,
            "logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to ViT format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in ViT format
        """
        return {
            "labels": targets["class_labels"]
        }
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format ViT outputs for metric computation.
        
        Args:
            outputs: ViT outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        logits = outputs["logits"]
        batch_size = logits.shape[0]
        
        preds = []
        for i in range(batch_size):
            preds.append({
                "scores": torch.softmax(logits[i], dim=-1),
                "labels": torch.argmax(logits[i], dim=-1)
            })
        return preds
    
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        labels = targets["class_labels"]
        batch_size = labels.shape[0]
        
        target_list = []
        for i in range(batch_size):
            target_list.append({
                "labels": labels[i]
            })
        return target_list

class ConvNeXTOutputAdapter(OutputAdapter):
    """Adapter for ConvNeXT model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt ConvNeXT outputs to standard format.
        
        Args:
            outputs: Raw ConvNeXT outputs
            
        Returns:
            Adapted outputs in standard format
        """
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")
        
        return {
            "loss": loss,
            "logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to ConvNeXT format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in ConvNeXT format
        """
        return {
            "labels": targets["class_labels"]
        }
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format ConvNeXT outputs for metric computation.
        
        Args:
            outputs: ConvNeXT outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        logits = outputs["logits"]
        batch_size = logits.shape[0]
        
        preds = []
        for i in range(batch_size):
            preds.append({
                "scores": torch.softmax(logits[i], dim=-1),
                "labels": torch.argmax(logits[i], dim=-1)
            })
        return preds
    
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        labels = targets["class_labels"]
        batch_size = labels.shape[0]
        
        target_list = []
        for i in range(batch_size):
            target_list.append({
                "labels": labels[i]
            })
        return target_list

class SwinOutputAdapter(OutputAdapter):
    """Adapter for Swin Transformer model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt Swin outputs to standard format.
        
        Args:
            outputs: Raw Swin outputs
            
        Returns:
            Adapted outputs in standard format
        """
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")
        
        return {
            "loss": loss,
            "logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to Swin format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in Swin format
        """
        return {
            "labels": targets["class_labels"]
        }
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format Swin outputs for metric computation.
        
        Args:
            outputs: Swin outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        logits = outputs["logits"]
        batch_size = logits.shape[0]
        
        preds = []
        for i in range(batch_size):
            preds.append({
                "scores": torch.softmax(logits[i], dim=-1),
                "labels": torch.argmax(logits[i], dim=-1)
            })
        return preds
    
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        labels = targets["class_labels"]
        batch_size = labels.shape[0]
        
        target_list = []
        for i in range(batch_size):
            target_list.append({
                "labels": labels[i]
            })
        return target_list

class ResNetOutputAdapter(OutputAdapter):
    """Adapter for ResNet model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt ResNet outputs to standard format.
        
        Args:
            outputs: Raw ResNet outputs
            
        Returns:
            Adapted outputs in standard format
        """
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")
        
        return {
            "loss": loss,
            "logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
    
    def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt targets to ResNet format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in ResNet format
        """
        return {
            "labels": targets["class_labels"]
        }
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format ResNet outputs for metric computation.
        
        Args:
            outputs: ResNet outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        logits = outputs["logits"]
        batch_size = logits.shape[0]
        
        preds = []
        for i in range(batch_size):
            preds.append({
                "scores": torch.softmax(logits[i], dim=-1),
                "labels": torch.argmax(logits[i], dim=-1)
            })
        return preds
    
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        labels = targets["class_labels"]
        batch_size = labels.shape[0]
        
        target_list = []
        for i in range(batch_size):
            target_list.append({
                "labels": labels[i]
            })
        return target_list

def get_adapter(model_name: str, image_size: int = 224) -> BaseAdapter:
    """Get the appropriate adapter for a model."""
    model_name_lower = model_name.lower()
    
    if "vit" in model_name_lower:
        return ViTAdapter(model_name=model_name, image_size=image_size)
    elif "convnext" in model_name_lower:
        return ConvNeXTAdapter(model_name=model_name, image_size=image_size)
    elif "swin" in model_name_lower:
        return SwinAdapter(model_name=model_name, image_size=image_size)
    else:
        return NoOpAdapter()

def get_output_adapter(model_name: str) -> OutputAdapter:
    """Get the appropriate output adapter for a model."""
    model_name_lower = model_name.lower()
    
    if "vit" in model_name_lower:
        return ViTOutputAdapter()
    elif "convnext" in model_name_lower:
        return ConvNeXTOutputAdapter()
    elif "swin" in model_name_lower:
        return SwinOutputAdapter()
    elif "resnet" in model_name_lower:
        return ResNetOutputAdapter()
    else:
        # Default to ViT adapter for unknown models
        return ViTOutputAdapter() 
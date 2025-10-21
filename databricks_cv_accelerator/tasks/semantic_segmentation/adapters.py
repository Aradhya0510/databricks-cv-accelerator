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
            target: A dictionary containing segmentation masks and metadata
            
        Returns:
            A tuple containing the processed image tensor and target dictionary
        """
        pass

class NoOpInputAdapter(BaseAdapter):
    """A "No-Operation" input adapter.
    - Converts image to a tensor
    - Keeps targets in standard format
    - Suitable for models like torchvision's segmentation models
    """
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        return F.to_tensor(image), target

class SegFormerInputAdapter(BaseAdapter):
    """Input adapter for SegFormer models.
    - Handles image resizing and preprocessing for SegFormer architecture
    - Uses Hugging Face's AutoImageProcessor
    """
    def __init__(self, model_name: str, image_size: int = 512):
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

class DeepLabV3InputAdapter(BaseAdapter):
    """Input adapter for DeepLabV3 models.
    - Handles image resizing and preprocessing for DeepLabV3 architecture
    """
    def __init__(self, model_name: str, image_size: int = 512):
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

class SemanticSegmentationOutputAdapter(OutputAdapter):
    """Adapter for semantic segmentation model outputs."""
    
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt semantic segmentation outputs to standard format.
        
        Args:
            outputs: Raw model outputs
            
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
        """Adapt targets to semantic segmentation format.
        
        Args:
            targets: Standard format targets
            
        Returns:
            Adapted targets in semantic segmentation format
        """
        # For semantic segmentation, we expect semantic_masks
        if "semantic_masks" in targets:
            return {"labels": targets["semantic_masks"]}
        elif "labels" in targets:
            return targets
        else:
            raise ValueError("Targets must contain 'semantic_masks' or 'labels'")
    
    def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format model outputs for metric computation.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=1)
        
        # Convert to list of dictionaries
        result = []
        for i in range(predictions.shape[0]):
            result.append({
                "semantic_mask": predictions[i]
            })
        
        return result
    
    def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format targets for metric computation.
        
        Args:
            targets: Standard format targets
            
        Returns:
            List of target dictionaries for each image
        """
        # Get semantic masks
        if "semantic_masks" in targets:
            masks = targets["semantic_masks"]
        elif "labels" in targets:
            masks = targets["labels"]
        else:
            raise ValueError("Targets must contain 'semantic_masks' or 'labels'")
        
        # Convert to list of dictionaries
        result = []
        for i in range(masks.shape[0]):
            result.append({
                "semantic_mask": masks[i]
            })
        
        return result

def get_input_adapter(model_name: str, image_size: int = 512) -> BaseAdapter:
    """Get the appropriate input adapter for a given model name.
    
    Args:
        model_name: Name of the model
        image_size: Target image size
        
    Returns:
        Appropriate input adapter instance
    """
    model_name_lower = model_name.lower()
    
    if "segformer" in model_name_lower:
        return SegFormerInputAdapter(model_name, image_size)
    elif "deeplab" in model_name_lower:
        return DeepLabV3InputAdapter(model_name, image_size)
    else:
        # Default to no-op adapter for other models
        return NoOpInputAdapter()

def get_output_adapter(model_name: str) -> OutputAdapter:
    """Get the appropriate output adapter for a given model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Appropriate output adapter instance
    """
    # For semantic segmentation, we use a single output adapter
    # since the output format is consistent across models
    return SemanticSegmentationOutputAdapter()

# Keep the old function for backward compatibility
def get_semantic_adapter(model_name: str, image_size: int = 512) -> BaseAdapter:
    """Get the appropriate adapter for a given model name.
    
    Args:
        model_name: Name of the model
        image_size: Target image size
        
    Returns:
        Appropriate adapter instance
    """
    return get_input_adapter(model_name, image_size) 
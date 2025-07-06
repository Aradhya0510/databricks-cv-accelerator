# Instance Segmentation Adapters

This document explains the instance segmentation adapters in our framework, how they work, and why they were designed in specific ways to handle different instance segmentation model architectures.

## Overview

Instance segmentation adapters are responsible for converting input images and instance-level annotations into the format expected by different instance segmentation models. Each adapter handles the specific preprocessing requirements needed for different instance segmentation architectures, ensuring proper handling of individual object instances and their masks.

## Adapter Types

### 1. NoOpAdapter

**Purpose**: A minimal adapter that performs basic preprocessing without model-specific transformations.

**Why This Design**: 
- Acts as a fallback for models that don't require specific preprocessing
- Provides basic tensor conversion and resizing
- Maintains the original instance annotation format
- Suitable for basic instance segmentation models

**Input Requirements**:
- Raw PIL Image
- Instance segmentation target: `{"instance_masks": torch.tensor, "boxes": torch.tensor, "labels": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Resized and converted to tensor
- Target: Unchanged instance segmentation format

**Supported Models**:
- Basic instance segmentation models
- Custom models with standard preprocessing requirements

**Code Example**:
```python
adapter = NoOpAdapter()
processed_image, target = adapter(image, {
    "instance_masks": torch.tensor([[[0, 1], [1, 0]]]),  # NxHxW masks
    "boxes": torch.tensor([[100, 100, 200, 200]]),
    "labels": torch.tensor([1]),
    "image_id": torch.tensor([0])
})
```

### 2. Mask2FormerAdapter

**Purpose**: Handles preprocessing for Mask2Former models.

**Why This Design**:
- Mask2Former is a transformer-based instance segmentation model
- Requires specific preprocessing including resizing, normalization, and padding
- Uses Hugging Face's AutoImageProcessor optimized for transformer-based instance segmentation
- Handles the unique requirements of query-based instance segmentation

**Input Requirements**:
- Raw PIL Image
- Instance segmentation target: `{"instance_masks": torch.tensor, "boxes": torch.tensor, "labels": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Preprocessed tensor with Mask2Former-specific normalization and padding
- Target: Unchanged instance segmentation format

**Key Preprocessing Steps**:
1. **Resize**: Scales image to specified size (default 512x512)
2. **Rescale**: Converts pixel values to [0, 1] range
3. **Normalize**: Applies ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Pad**: Adds padding to maintain aspect ratio

**Why Mask2Former-Specific Design**:
- Mask2Former uses a query-based architecture for instance segmentation
- Requires specific image sizes for optimal query processing
- Uses transformer-specific normalization
- Optimized for instance-level mask prediction
- Handles the unique requirements of query-based object detection and segmentation

**Supported Models**:
- `facebook/mask2former-swin-base-coco-instance`
- `facebook/mask2former-swin-large-coco-instance`
- `facebook/mask2former-swin-tiny-coco-instance`
- `facebook/mask2former-resnet-50-coco-instance`
- `facebook/mask2former-resnet-101-coco-instance`
- Any Mask2Former instance segmentation model from Hugging Face

**Code Example**:
```python
adapter = Mask2FormerAdapter("facebook/mask2former-swin-base-coco-instance", image_size=512)
processed_image, target = adapter(image, {
    "instance_masks": torch.tensor([[[0, 1], [1, 0]]]),
    "boxes": torch.tensor([[100, 100, 200, 200]]),
    "labels": torch.tensor([1]),
    "image_id": torch.tensor([0])
})
# processed_image: torch.Tensor (3, 512, 512) with normalization
# target: {"instance_masks": torch.tensor, "boxes": torch.tensor, "labels": torch.tensor, "image_id": torch.tensor([0])}
```

## Output Adapters

### InstanceSegmentationOutputAdapter

**Purpose**: Converts instance segmentation model outputs to a standardized format for metric computation.

**Why This Design**:
- Different instance segmentation models return outputs in different formats
- Standardizes the output format for consistent metric computation
- Handles both loss and prediction extraction
- Ensures proper instance mask format for evaluation

**Input Format**: Raw model outputs (varies by model)

**Output Format**:
```python
{
    "loss": torch.tensor,  # Training loss
    "pred_masks": torch.tensor,  # Predicted instance masks (B, N, H, W)
    "pred_boxes": torch.tensor,  # Predicted bounding boxes (B, N, 4)
    "pred_logits": torch.tensor,  # Raw prediction logits (B, N, C)
    "loss_dict": dict  # Additional loss components
}
```

**Key Functions**:
1. **adapt_output()**: Extracts loss, masks, boxes, and logits from model outputs
2. **adapt_targets()**: Converts targets to model-specific format
3. **format_predictions()**: Formats outputs for metric computation
4. **format_targets()**: Formats targets for metric computation

**Why Standardized Output is Important**:
- Enables consistent evaluation metrics (mAP, IoU, Dice)
- Ensures compatibility with existing evaluation tools
- Maintains consistency across different instance segmentation models
- Simplifies metric computation and comparison
- Supports instance-level evaluation metrics

**Code Example**:
```python
output_adapter = InstanceSegmentationOutputAdapter()
adapted_outputs = output_adapter.adapt_output(model_outputs)
# adapted_outputs: {"loss": tensor, "pred_masks": tensor, "pred_boxes": tensor, "pred_logits": tensor, "loss_dict": {}}
```

## Factory Function

### get_instance_adapter()

**Purpose**: Automatically selects the appropriate adapter based on model name.

**Why This Design**:
- Eliminates the need to manually specify adapters
- Uses model name patterns to determine the correct adapter
- Provides fallback to NoOpAdapter for unknown models
- Ensures optimal preprocessing for each model type

**Selection Logic**:
```python
def get_instance_adapter(model_name: str, image_size: int = 512) -> BaseAdapter:
    model_name_lower = model_name.lower()
    
    if "mask2former" in model_name_lower:
        return Mask2FormerAdapter(model_name, image_size)
    else:
        return NoOpAdapter()
```

**Usage Example**:
```python
# Automatically selects Mask2FormerAdapter
adapter = get_instance_adapter("facebook/mask2former-swin-base-coco-instance")

# Falls back to NoOpAdapter
adapter = get_instance_adapter("unknown_model")
```

## Instance Mask Handling

### Why Instance Mask Handling is Critical

**Different Model Requirements**:
- **Mask2Former**: Expects instance masks in specific format for query processing
- **Standard Format**: Uses COCO format instance annotations
- **Evaluation Format**: Requires instance-level mask evaluation

**Instance Mask Processing**:
1. **Original**: COCO format instance annotations with polygon/RLE masks
2. **Processing**: Convert to model-specific format
3. **Evaluation**: Convert back to standard format for metrics

**Example Instance Mask Processing**:
```python
# Original COCO instance: Multiple objects with individual masks
# Standard format: NxHxW tensor with binary masks per instance
# Model-specific: May require additional processing for optimal performance
```

## Design Principles

### 1. Instance-Level Processing
- Each adapter handles individual object instances
- Maintains instance identity throughout processing
- Ensures proper instance-level evaluation
- Supports instance counting and tracking

### 2. Query-Based Architecture Support
- **Mask2Former**: Uses query-based architecture for instance detection
- **Query Processing**: Handles the unique requirements of query-based models
- **Instance Queries**: Manages instance-specific query processing
- **Mask Prediction**: Optimizes for instance-level mask prediction

### 3. COCO Instance Compatibility
- All adapters maintain COCO instance format compatibility
- Enables standard instance evaluation metrics (mAP, IoU)
- Supports existing instance segmentation datasets and tools
- Ensures proper instance-level classification and segmentation

### 4. Flexibility
- Easy to add new instance segmentation model support
- Configurable image sizes
- Fallback mechanisms for unknown models
- Support for various instance annotation formats

## Adding New Instance Segmentation Adapters

To add support for a new instance segmentation model:

1. **Create Input Adapter**:
```python
class NewInstanceAdapter(BaseAdapter):
    def __init__(self, model_name: str, image_size: int = 512):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # Model-specific preprocessing
        processed = self.processor(image, return_tensors="pt")
        
        # Instance mask processing if needed
        adapted_target = self._process_instance_masks(target)
        
        return processed.pixel_values.squeeze(0), adapted_target
    
    def _process_instance_masks(self, target: Dict) -> Dict:
        # Model-specific instance mask processing
        # ...
        return adapted_target
```

2. **Create Output Adapter**:
```python
class NewInstanceOutputAdapter(OutputAdapter):
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract model-specific outputs
        return {
            "loss": outputs.loss,
            "pred_masks": outputs.pred_masks,
            "pred_boxes": outputs.pred_boxes,
            "pred_logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
```

3. **Update Factory Function**:
```python
def get_instance_adapter(model_name: str, image_size: int = 512) -> BaseAdapter:
    model_name_lower = model_name.lower()
    
    if "new_model" in model_name_lower:
        return NewInstanceAdapter(model_name, image_size)
    # ... existing logic
```

## Instance Segmentation vs Semantic Segmentation

### Key Differences

**Semantic Segmentation**:
- Pixel-level classification
- Single mask per image
- Class-based labeling
- Background included

**Instance Segmentation**:
- Instance-level detection and segmentation
- Multiple masks per image (one per instance)
- Instance-based labeling
- Background excluded
- Individual object identification

### Adapter Implications

**Semantic Segmentation Adapters**:
- Focus on pixel-level classification
- Handle single mask processing
- Optimize for class-based metrics

**Instance Segmentation Adapters**:
- Focus on instance-level detection
- Handle multiple mask processing
- Optimize for instance-based metrics
- Support query-based architectures

## Best Practices

1. **Understand Model Architecture**: Study the model's preprocessing requirements thoroughly
2. **Instance Mask Handling**: Ensure proper instance mask format conversion
3. **Evaluation Compatibility**: Maintain compatibility with standard instance evaluation metrics
4. **Error Handling**: Provide robust error handling for edge cases
5. **Testing**: Test with various image sizes and instance annotation formats
6. **Documentation**: Document model-specific requirements and assumptions
7. **Performance Optimization**: Use model-specific optimizations for better performance
8. **Instance Identity**: Maintain instance identity throughout processing
9. **Query Processing**: Handle query-based architecture requirements properly

## Input Adapter Example

```python
class Mask2FormerInputAdapter(BaseAdapter):
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # ...
        return processed_image, adapted_target
```

## Output Adapter Example

```python
class Mask2FormerOutputAdapter:
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # ...
        return standardized_outputs
```

## Factory Functions

```python
input_adapter = get_input_adapter("facebook/mask2former-swin-base-coco-instance")  # Returns Mask2FormerInputAdapter
output_adapter = get_output_adapter("facebook/mask2former-swin-base-coco-instance")  # Returns Mask2FormerOutputAdapter
``` 
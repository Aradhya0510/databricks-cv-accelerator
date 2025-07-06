# Panoptic Segmentation Adapters

This document explains the panoptic segmentation adapters in our framework, how they work, and why they were designed in specific ways to handle different panoptic segmentation model architectures.

## Overview

Panoptic segmentation adapters are responsible for converting input images and panoptic annotations into the format expected by different panoptic segmentation models. Each adapter handles the specific preprocessing requirements needed for different panoptic segmentation architectures, ensuring proper handling of both "things" (countable objects) and "stuff" (background regions) in a unified segmentation framework.

## Adapter Types

### 1. NoOpAdapter

**Purpose**: A minimal adapter that performs basic preprocessing without model-specific transformations.

**Why This Design**: 
- Acts as a fallback for models that don't require specific preprocessing
- Provides basic tensor conversion and resizing
- Maintains the original panoptic annotation format
- Suitable for basic panoptic segmentation models

**Input Requirements**:
- Raw PIL Image
- Panoptic segmentation target: `{"panoptic_masks": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Resized and converted to tensor
- Target: Unchanged panoptic segmentation format

**Supported Models**:
- Basic panoptic segmentation models
- Custom models with standard preprocessing requirements

**Code Example**:
```python
adapter = NoOpAdapter()
processed_image, target = adapter(image, {
    "panoptic_masks": torch.tensor([[0, 1, 2], [0, 1, 2]]),  # HxW mask with instance IDs
    "image_id": torch.tensor([0])
})
```

### 2. Mask2FormerAdapter

**Purpose**: Handles preprocessing for Mask2Former panoptic segmentation models.

**Why This Design**:
- Mask2Former is a transformer-based panoptic segmentation model
- Requires specific preprocessing including resizing, normalization, and padding
- Uses Hugging Face's AutoImageProcessor optimized for transformer-based panoptic segmentation
- Handles the unique requirements of unified "things" and "stuff" segmentation

**Input Requirements**:
- Raw PIL Image
- Panoptic segmentation target: `{"panoptic_masks": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Preprocessed tensor with Mask2Former-specific normalization and padding
- Target: Unchanged panoptic segmentation format

**Key Preprocessing Steps**:
1. **Resize**: Scales image to specified size (default 512x512)
2. **Rescale**: Converts pixel values to [0, 1] range
3. **Normalize**: Applies ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Pad**: Adds padding to maintain aspect ratio

**Why Mask2Former-Specific Design**:
- Mask2Former uses a unified architecture for both instance and semantic segmentation
- Requires specific image sizes for optimal panoptic processing
- Uses transformer-specific normalization
- Optimized for unified "things" and "stuff" prediction
- Handles the unique requirements of panoptic quality computation

**Supported Models**:
- `facebook/mask2former-swin-base-coco-panoptic`
- `facebook/mask2former-swin-large-coco-panoptic`
- `facebook/mask2former-swin-tiny-coco-panoptic`
- `facebook/mask2former-resnet-50-coco-panoptic`
- `facebook/mask2former-resnet-101-coco-panoptic`
- Any Mask2Former panoptic segmentation model from Hugging Face

**Code Example**:
```python
adapter = Mask2FormerAdapter("facebook/mask2former-swin-base-coco-panoptic", image_size=512)
processed_image, target = adapter(image, {
    "panoptic_masks": torch.tensor([[0, 1, 2], [0, 1, 2]]),
    "image_id": torch.tensor([0])
})
# processed_image: torch.Tensor (3, 512, 512) with normalization
# target: {"panoptic_masks": torch.tensor, "image_id": torch.tensor([0])}
```

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
input_adapter = get_input_adapter("facebook/mask2former-swin-base-coco-panoptic")  # Returns Mask2FormerInputAdapter
output_adapter = get_output_adapter("facebook/mask2former-swin-base-coco-panoptic")  # Returns Mask2FormerOutputAdapter
```

## Panoptic Mask Handling

### Why Panoptic Mask Handling is Critical

**Different Model Requirements**:
- **Mask2Former**: Expects panoptic masks in specific format for unified processing
- **Standard Format**: Uses COCO panoptic format with instance IDs
- **Evaluation Format**: Requires panoptic quality computation

**Panoptic Mask Processing**:
1. **Original**: COCO panoptic format with "things" and "stuff" annotations
2. **Processing**: Convert to model-specific format
3. **Evaluation**: Convert back to standard format for metrics

**Example Panoptic Mask Processing**:
```python
# Original COCO panoptic: Unified "things" and "stuff" annotations
# Standard format: HxW tensor with instance IDs (things) and class IDs (stuff)
# Model-specific: May require additional processing for optimal performance
```

## Design Principles

### 1. Unified Architecture Support
- Each adapter handles both "things" and "stuff" segmentation
- Maintains unified processing for panoptic quality
- Ensures proper handling of instance IDs and class IDs
- Supports panoptic quality computation

### 2. Query-Based Architecture Support
- **Mask2Former**: Uses query-based architecture for unified segmentation
- **Query Processing**: Handles the unique requirements of query-based models
- **Unified Queries**: Manages both instance and semantic queries
- **Panoptic Prediction**: Optimizes for unified "things" and "stuff" prediction

### 3. COCO Panoptic Compatibility
- All adapters maintain COCO panoptic format compatibility
- Enables standard panoptic evaluation metrics (PQ, SQ, RQ)
- Supports existing panoptic segmentation datasets and tools
- Ensures proper unified segmentation evaluation

### 4. Flexibility
- Easy to add new panoptic segmentation model support
- Configurable image sizes
- Fallback mechanisms for unknown models
- Support for various panoptic annotation formats

## Adding New Panoptic Segmentation Adapters

To add support for a new panoptic segmentation model:

1. **Create Input Adapter**:
```python
class NewPanopticAdapter(BaseAdapter):
    def __init__(self, model_name: str, image_size: int = 512):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # Model-specific preprocessing
        processed = self.processor(image, return_tensors="pt")
        
        # Panoptic mask processing if needed
        adapted_target = self._process_panoptic_masks(target)
        
        return processed.pixel_values.squeeze(0), adapted_target
    
    def _process_panoptic_masks(self, target: Dict) -> Dict:
        # Model-specific panoptic mask processing
        # ...
        return adapted_target
```

2. **Create Output Adapter**:
```python
class NewPanopticOutputAdapter(OutputAdapter):
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract model-specific outputs
        return {
            "loss": outputs.loss,
            "pred_masks": outputs.pred_masks,
            "pred_logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
```

3. **Update Factory Function**:
```python
def get_panoptic_adapter(model_name: str, image_size: int = 512) -> BaseAdapter:
    model_name_lower = model_name.lower()
    
    if "new_model" in model_name_lower:
        return NewPanopticAdapter(model_name, image_size)
    # ... existing logic
```

## Panoptic Segmentation vs Other Segmentation Types

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

**Panoptic Segmentation**:
- Unified "things" and "stuff" segmentation
- Single unified mask per image
- Combined instance and semantic labeling
- Both background and objects included
- Instance IDs for "things", class IDs for "stuff"

### Adapter Implications

**Semantic Segmentation Adapters**:
- Focus on pixel-level classification
- Handle single mask processing
- Optimize for class-based metrics

**Instance Segmentation Adapters**:
- Focus on instance-level detection
- Handle multiple mask processing
- Optimize for instance-based metrics

**Panoptic Segmentation Adapters**:
- Focus on unified segmentation
- Handle combined "things" and "stuff" processing
- Optimize for panoptic quality metrics
- Support unified query-based architectures

## Panoptic Quality Metrics

### Understanding Panoptic Quality

**Panoptic Quality (PQ)**:
- Combines segmentation quality (SQ) and recognition quality (RQ)
- Measures both "things" and "stuff" segmentation accuracy
- Provides unified evaluation metric for panoptic segmentation

**Segmentation Quality (SQ)**:
- Measures mask quality for matched segments
- Computes IoU for correctly classified segments
- Focuses on pixel-level accuracy

**Recognition Quality (RQ)**:
- Measures detection accuracy
- Computes F1 score for segment detection
- Focuses on instance-level accuracy

### Adapter Support for Panoptic Quality

**Input Processing**:
- Maintains proper instance ID and class ID separation
- Ensures correct "things" vs "stuff" classification
- Supports panoptic quality computation

**Output Processing**:
- Formats predictions for panoptic quality evaluation
- Maintains instance identity for "things"
- Ensures proper class labeling for "stuff"

## Best Practices

1. **Understand Model Architecture**: Study the model's preprocessing requirements thoroughly
2. **Panoptic Mask Handling**: Ensure proper panoptic mask format conversion
3. **Evaluation Compatibility**: Maintain compatibility with standard panoptic evaluation metrics
4. **Error Handling**: Provide robust error handling for edge cases
5. **Testing**: Test with various image sizes and panoptic annotation formats
6. **Documentation**: Document model-specific requirements and assumptions
7. **Performance Optimization**: Use model-specific optimizations for better performance
8. **Unified Processing**: Handle both "things" and "stuff" processing properly
9. **Query Processing**: Handle query-based architecture requirements properly
10. **Panoptic Quality**: Ensure proper support for panoptic quality computation 
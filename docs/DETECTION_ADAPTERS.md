# Detection Adapters

This document explains the detection adapters in our framework, how they work, and why they were designed in specific ways to handle different object detection model architectures.

## Overview

Detection adapters are responsible for converting input images and COCO-format annotations into the format expected by different object detection models. Each adapter handles the specific preprocessing requirements and coordinate transformations needed for different detection architectures.

## Adapter Types

### 1. NoOpAdapter

**Purpose**: A minimal adapter that performs basic preprocessing without model-specific transformations.

**Why This Design**: 
- Acts as a fallback for models that don't require specific preprocessing
- Provides basic tensor conversion and resizing
- Maintains the original annotation format

**Input Requirements**:
- Raw PIL Image
- COCO format target: `{"boxes": torch.tensor, "labels": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Resized and converted to tensor
- Target: Unchanged COCO format

**Supported Models**:
- Basic detection models that expect simple tensor inputs
- Custom models with standard preprocessing requirements

**Code Example**:
```python
adapter = NoOpAdapter()
processed_image, target = adapter(image, {
    "boxes": torch.tensor([[100, 100, 200, 200]]),
    "labels": torch.tensor([1]),
    "image_id": torch.tensor([0])
})
```

### 2. DETRAdapter

**Purpose**: Handles preprocessing for DETR (DEtection TRansformer) models.

**Why This Design**:
- DETR models require specific preprocessing including resizing, normalization, and padding
- Uses Hugging Face's AutoImageProcessor optimized for transformer-based detection
- Handles the unique requirements of end-to-end object detection transformers

**Input Requirements**:
- Raw PIL Image
- COCO format target: `{"boxes": torch.tensor, "labels": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Preprocessed tensor with DETR-specific normalization and padding
- Target: Adapted with coordinate transformations

**Key Preprocessing Steps**:
1. **Resize**: Scales image to specified size (default 800x800)
2. **Rescale**: Converts pixel values to [0, 1] range
3. **Normalize**: Applies ImageNet normalization
4. **Pad**: Adds padding to maintain aspect ratio
5. **Coordinate Transformation**: Converts box coordinates to normalized format

**Why Coordinate Transformation is Needed**:
- DETR expects normalized coordinates (0-1 range)
- Original COCO annotations are in pixel coordinates
- Transformation ensures proper loss computation and prediction accuracy

**Supported Models**:
- `facebook/detr-resnet-50`
- `facebook/detr-resnet-101`
- `facebook/detr-resnet-50-dc5`
- `facebook/detr-resnet-101-dc5`
- Any DETR model from Hugging Face

**Code Example**:
```python
adapter = DETRAdapter("facebook/detr-resnet-50", image_size=800)
processed_image, target = adapter(image, {
    "boxes": torch.tensor([[100, 100, 200, 200]]),
    "labels": torch.tensor([1]),
    "image_id": torch.tensor([0])
})
# processed_image: torch.Tensor (3, 800, 800) with normalization
# target: {"boxes": normalized_boxes, "labels": torch.tensor([1]), "image_id": torch.tensor([0])}
```

### 3. YOLOSAdapter

**Purpose**: Handles preprocessing for YOLOS (You Only Look at One Sequence) models.

**Why This Design**:
- YOLOS is a transformer-based object detector that requires specific preprocessing
- Handles the unique requirements of YOLOS architecture
- Manages coordinate transformations specific to YOLOS

**Input Requirements**:
- Raw PIL Image
- COCO format target: `{"boxes": torch.tensor, "labels": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Preprocessed tensor with YOLOS-specific normalization
- Target: Adapted with YOLOS-specific coordinate transformations

**Key Preprocessing Steps**:
1. **Resize**: Scales image to specified size (default 800x800)
2. **Rescale**: Converts pixel values to [0, 1] range
3. **Normalize**: Applies YOLOS-specific normalization
4. **Pad**: Adds padding optimized for YOLOS architecture
5. **Coordinate Transformation**: Converts to YOLOS-specific format

**Why YOLOS-Specific Design**:
- YOLOS uses a different coordinate system than DETR
- Requires specific aspect ratio handling
- Optimized for the YOLOS transformer architecture

**Supported Models**:
- `hustvl/yolos-tiny`
- `hustvl/yolos-small`
- `hustvl/yolos-base`
- `hustvl/yolos-small-300`
- Any YOLOS model from Hugging Face

**Code Example**:
```python
adapter = YOLOSAdapter("hustvl/yolos-base", image_size=800)
processed_image, target = adapter(image, {
    "boxes": torch.tensor([[100, 100, 200, 200]]),
    "labels": torch.tensor([1]),
    "image_id": torch.tensor([0])
})
# processed_image: torch.Tensor (3, 800, 800) with YOLOS normalization
# target: {"boxes": yolos_boxes, "labels": torch.tensor([1]), "image_id": torch.tensor([0])}
```

## Input Adapter Example

```python
class DETRInputAdapter(BaseAdapter):
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # ...
        return processed_image, adapted_target
```

## Output Adapter Example

```python
class DETROutputAdapter:
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # ...
        return standardized_outputs
```

## Factory Functions

```python
input_adapter = get_input_adapter("facebook/detr-resnet-50")  # Returns DETRInputAdapter
output_adapter = get_output_adapter("facebook/detr-resnet-50")  # Returns DETROutputAdapter
```

## Coordinate Transformations

### Why Coordinate Transformations Are Critical

**Different Model Requirements**:
- **DETR**: Expects normalized coordinates (0-1 range)
- **YOLOS**: Uses specific coordinate system optimized for transformer architecture
- **COCO Format**: Uses pixel coordinates

**Transformation Process**:
1. **Original**: COCO pixel coordinates `[x1, y1, x2, y2]`
2. **Normalized**: Convert to model-specific format
3. **Evaluation**: Convert back to COCO format for metrics

**Example Transformation**:
```python
# Original COCO box: [100, 100, 200, 200] in 800x800 image
# DETR normalized: [0.125, 0.125, 0.25, 0.25] (divided by image size)
# YOLOS format: [center_x, center_y, width, height] in normalized coordinates
```

## Design Principles

### 1. Model-Specific Optimization
- Each adapter is optimized for its target model architecture
- Handles model-specific preprocessing requirements
- Ensures optimal performance for each model type

### 2. COCO Compatibility
- All adapters maintain COCO format compatibility
- Enables standard evaluation metrics
- Supports existing COCO datasets and tools

### 3. Coordinate System Handling
- Proper coordinate transformations for each model
- Maintains accuracy during preprocessing
- Ensures correct loss computation

### 4. Flexibility
- Easy to add new detection model support
- Configurable image sizes
- Fallback mechanisms for unknown models

## Adding New Detection Adapters

To add support for a new detection model:

1. **Create Input Adapter**:
```python
class NewDetectionAdapter(BaseAdapter):
    def __init__(self, model_name: str, image_size: int = 800):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # Model-specific preprocessing
        processed = self.processor(image, return_tensors="pt")
        
        # Coordinate transformation
        adapted_target = self._transform_target(target, image.size)
        
        return processed.pixel_values.squeeze(0), adapted_target
    
    def _transform_target(self, target: Dict, image_size: Tuple[int, int]) -> Dict:
        # Model-specific coordinate transformation
        # ...
        return adapted_target
```

2. **Create Output Adapter**:
```python
class NewDetectionOutputAdapter(OutputAdapter):
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract model-specific outputs
        return {
            "loss": outputs.loss,
            "pred_boxes": outputs.pred_boxes,
            "pred_logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
```

3. **Update Factory Function**:
```python
def get_adapter(model_name: str, image_size: int = 800) -> BaseAdapter:
    model_name_lower = model_name.lower()
    
    if "new_model" in model_name_lower:
        return NewDetectionAdapter(model_name=model_name, image_size=image_size)
    # ... existing logic
```

## Best Practices

1. **Understand Model Requirements**: Study the model's preprocessing requirements thoroughly
2. **Coordinate Transformations**: Ensure accurate coordinate transformations
3. **COCO Compatibility**: Maintain COCO format for evaluation
4. **Error Handling**: Provide robust error handling for edge cases
5. **Testing**: Test with various image sizes and annotation formats
6. **Documentation**: Document model-specific requirements and assumptions 
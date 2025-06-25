# Adapter System Overview

This document provides a comprehensive overview of the adapter system in our computer vision framework. The adapter system is designed to handle the diverse preprocessing requirements of different model architectures while maintaining a consistent interface for users.

## What Are Adapters?

Adapters are specialized components that convert input data (images and annotations) into the format expected by specific model architectures. They act as translators between our standardized data format and the model-specific requirements.

### Why Do We Need Adapters?

Different computer vision models have different preprocessing requirements:

- **Input Format**: Some models expect normalized tensors, others expect raw images
- **Image Size**: Models require specific input dimensions (224x224, 512x512, 800x800, etc.)
- **Normalization**: Different models use different mean/std values for normalization
- **Coordinate Systems**: Detection models may expect different coordinate formats
- **Mask Formats**: Segmentation models handle masks differently
- **Output Format**: Models return outputs in various structures

Adapters solve these differences by providing model-specific preprocessing while maintaining a consistent interface.

## Adapter Types by Task

### 1. Classification Adapters
**Purpose**: Handle image classification models (ViT, ConvNeXT, Swin, ResNet)

**Key Features**:
- Image preprocessing for classification tasks
- Class label handling
- Model-specific normalization

**Detailed Documentation**: [Classification Adapters](CLASSIFICATION_ADAPTERS.md)

**Supported Models**:
- Vision Transformers (ViT, DeiT)
- ConvNeXT models
- Swin Transformers
- ResNet models

### 2. Detection Adapters
**Purpose**: Handle object detection models (DETR, YOLOS)

**Key Features**:
- Image preprocessing for detection tasks
- Bounding box coordinate transformations
- COCO format compatibility

**Detailed Documentation**: [Detection Adapters](DETECTION_ADAPTERS.md)

**Supported Models**:
- DETR (DEtection TRansformer)
- YOLOS (You Only Look at One Sequence)

### 3. Semantic Segmentation Adapters
**Purpose**: Handle semantic segmentation models (SegFormer, DeepLabV3)

**Key Features**:
- Image preprocessing for segmentation tasks
- Mask handling for pixel-level classification
- Semantic mask format compatibility

**Detailed Documentation**: [Semantic Segmentation Adapters](SEMANTIC_SEGMENTATION_ADAPTERS.md)

**Supported Models**:
- SegFormer models
- DeepLabV3 models

### 4. Instance Segmentation Adapters
**Purpose**: Handle instance segmentation models (Mask2Former)

**Key Features**:
- Image preprocessing for instance segmentation
- Instance mask handling
- Query-based architecture support

**Detailed Documentation**: [Instance Segmentation Adapters](INSTANCE_SEGMENTATION_ADAPTERS.md)

**Supported Models**:
- Mask2Former models

### 5. Panoptic Segmentation Adapters
**Purpose**: Handle panoptic segmentation models (Mask2Former)

**Key Features**:
- Image preprocessing for panoptic segmentation
- Unified "things" and "stuff" handling
- Panoptic quality metric support

**Detailed Documentation**: [Panoptic Segmentation Adapters](PANOPTIC_SEGMENTATION_ADAPTERS.md)

**Supported Models**:
- Mask2Former panoptic models

## How Adapters Work

### Input Adapters

Input adapters handle the preprocessing of images and annotations before they are fed to the model.

**Standard Interface**:
```python
def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
    """
    Process a single sample.
    
    Args:
        image: A PIL image
        target: A dictionary containing annotations
        
    Returns:
        A tuple containing the processed image tensor and target dictionary
    """
```

**Common Preprocessing Steps**:
1. **Resize**: Scale image to model-specific dimensions
2. **Normalize**: Apply model-specific normalization
3. **Pad**: Add padding if needed
4. **Convert**: Transform to tensor format
5. **Transform Annotations**: Convert annotations to model-specific format

### Output Adapters

Output adapters convert model outputs into a standardized format for metric computation and evaluation.

**Standard Interface**:
```python
def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert model outputs to standard format."""
    
def adapt_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert targets to model-specific format."""
    
def format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
    """Format outputs for metric computation."""
    
def format_targets(self, targets: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """Format targets for metric computation."""
```

## Factory Functions

Each task has a factory function that automatically selects the appropriate adapter based on the model name:

```python
# Classification
adapter = get_adapter("google/vit-base-patch16-224")  # Returns ViTAdapter

# Detection
adapter = get_adapter("facebook/detr-resnet-50")  # Returns DETRAdapter

# Semantic Segmentation
adapter = get_semantic_adapter("nvidia/segformer-b0-finetuned-ade-512-512")  # Returns SegFormerAdapter

# Instance Segmentation
adapter = get_instance_adapter("facebook/mask2former-swin-base-coco-instance")  # Returns Mask2FormerAdapter

# Panoptic Segmentation
adapter = get_panoptic_adapter("facebook/mask2former-swin-base-coco-panoptic")  # Returns Mask2FormerAdapter
```

## Design Principles

### 1. Consistency
- All adapters follow the same interface
- Standardized input/output formats
- Consistent error handling across all adapters

### 2. Model-Specific Optimization
- Each adapter is optimized for its target model architecture
- Handles model-specific preprocessing requirements
- Ensures optimal performance for each model type

### 3. Flexibility
- Easy to add new model support
- Configurable parameters (image size, normalization, etc.)
- Fallback mechanisms for unknown models

### 4. Maintainability
- Clear separation of concerns
- Well-documented design decisions
- Easy to test and debug

## Usage Examples

### Basic Usage

```python
from src.tasks.classification.adapters import get_adapter
from src.tasks.detection.adapters import get_adapter as get_detection_adapter

# Classification
classifier_adapter = get_adapter("google/vit-base-patch16-224")
processed_image, target = classifier_adapter(image, {"class_labels": torch.tensor([1])})

# Detection
detector_adapter = get_detection_adapter("facebook/detr-resnet-50")
processed_image, target = detector_adapter(image, {
    "boxes": torch.tensor([[100, 100, 200, 200]]),
    "labels": torch.tensor([1]),
    "image_id": torch.tensor([0])
})
```

### Advanced Usage

```python
# Custom image size
adapter = get_adapter("google/vit-base-patch16-224", image_size=384)

# Model-specific preprocessing
processed_image, target = adapter(image, target)

# The adapter automatically handles:
# - Resizing to 384x384
# - ViT-specific normalization
# - Proper tensor conversion
```

## Adding New Adapters

To add support for a new model:

1. **Create Input Adapter**:
```python
class NewModelAdapter(BaseAdapter):
    def __init__(self, model_name: str, image_size: int = 224):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        processed = self.processor(image, return_tensors="pt")
        return processed.pixel_values.squeeze(0), target
```

2. **Create Output Adapter** (if needed):
```python
class NewModelOutputAdapter(OutputAdapter):
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
```

3. **Update Factory Function**:
```python
def get_adapter(model_name: str, image_size: int = 224) -> BaseAdapter:
    model_name_lower = model_name.lower()
    
    if "new_model" in model_name_lower:
        return NewModelAdapter(model_name=model_name, image_size=image_size)
    # ... existing logic
```

## Best Practices

### 1. Use Factory Functions
- Always use factory functions instead of manually instantiating adapters
- Factory functions provide automatic model detection and fallbacks
- Ensures consistent adapter selection

### 2. Understand Model Requirements
- Study the model's preprocessing requirements thoroughly
- Test with various input sizes and formats
- Document model-specific assumptions

### 3. Handle Edge Cases
- Provide robust error handling
- Include fallback mechanisms for unknown models
- Test with edge cases (empty annotations, unusual image sizes, etc.)

### 4. Performance Optimization
- Use model-specific optimizations when available
- Leverage Hugging Face's AutoImageProcessor for transformer models
- Minimize unnecessary preprocessing steps

### 5. Testing
- Create comprehensive tests for each adapter
- Test with various input formats and sizes
- Verify output format consistency

## Troubleshooting

### Common Issues

1. **Model Not Found**: Check if the model name is correct and supported
2. **Shape Mismatches**: Ensure consistent image sizes across training and inference
3. **Coordinate Errors**: Verify coordinate transformations for detection models
4. **Mask Format Issues**: Check mask format compatibility for segmentation models

### Debugging Tips

1. **Check Adapter Selection**: Verify the correct adapter is being selected
2. **Inspect Preprocessing**: Print intermediate results to debug preprocessing steps
3. **Validate Inputs**: Ensure inputs match expected formats
4. **Test with Simple Cases**: Start with simple examples before complex scenarios

## Future Enhancements

### Planned Features

1. **Additional Model Support**: More transformer and CNN architectures
2. **Advanced Preprocessing**: More sophisticated data augmentation strategies
3. **Multi-task Learning**: Support for models that handle multiple tasks
4. **Model Compression**: Integration with quantization and pruning techniques

### Contributing

To contribute new adapters:

1. **Follow the Design Pattern**: Use existing adapters as templates
2. **Add Comprehensive Tests**: Ensure thorough test coverage
3. **Update Documentation**: Document the new adapter in the appropriate file
4. **Update Factory Functions**: Add model detection logic
5. **Follow Best Practices**: Adhere to the established design principles

## Conclusion

The adapter system provides a flexible, maintainable, and extensible solution for handling diverse model architectures. By understanding how adapters work and following the established patterns, you can easily add support for new models and ensure optimal performance across different computer vision tasks.

For detailed information about specific adapter types, refer to the individual documentation files:

- [Classification Adapters](CLASSIFICATION_ADAPTERS.md)
- [Detection Adapters](DETECTION_ADAPTERS.md)
- [Semantic Segmentation Adapters](SEMANTIC_SEGMENTATION_ADAPTERS.md)
- [Instance Segmentation Adapters](INSTANCE_SEGMENTATION_ADAPTERS.md)
- [Panoptic Segmentation Adapters](PANOPTIC_SEGMENTATION_ADAPTERS.md) 
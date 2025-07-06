# Classification Adapters

This document explains the classification adapters in our framework, how they work, and why they were designed in specific ways to handle different model architectures.

## Overview

Classification adapters are responsible for converting input images and annotations into the format expected by different classification models. Each adapter is designed to handle the specific preprocessing requirements of different model architectures while maintaining a consistent interface.

## Adapter Types

### 1. NoOpAdapter

**Purpose**: A minimal adapter that performs no special preprocessing.

**Why This Design**: 
- Some models (like torchvision's ResNet) expect simple tensor inputs without special preprocessing
- Acts as a fallback for models that don't require specific preprocessing
- Maintains the original image format and target structure

**Input Requirements**:
- Raw PIL Image
- Standard target format: `{"class_labels": torch.tensor}`

**Output Format**:
- Image: Converted to tensor using `torchvision.transforms.functional.to_tensor()`
- Target: Unchanged standard format

**Supported Models**:
- torchvision ResNet models
- Any model that expects simple tensor inputs

**Code Example**:
```python
adapter = NoOpAdapter()
processed_image, target = adapter(image, {"class_labels": torch.tensor([1])})
# processed_image: torch.Tensor (C, H, W)
# target: {"class_labels": torch.tensor([1])}
```

### 2. ViTAdapter

**Purpose**: Handles preprocessing for Vision Transformer (ViT) and related models.

**Why This Design**:
- ViT models require specific preprocessing including resizing, normalization, and padding
- Uses Hugging Face's AutoImageProcessor which is optimized for transformer models
- Ensures consistent preprocessing across different ViT variants

**Input Requirements**:
- Raw PIL Image
- Standard target format: `{"class_labels": torch.tensor}`

**Output Format**:
- Image: Preprocessed tensor with specific size, normalization, and padding
- Target: Unchanged standard format

**Key Preprocessing Steps**:
1. **Resize**: Scales image to specified size (default 224x224)
2. **Rescale**: Converts pixel values to [0, 1] range
3. **Normalize**: Applies ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Pad**: Adds padding if needed to maintain aspect ratio

**Supported Models**:
- `google/vit-base-patch16-224`
- `google/vit-large-patch16-224`
- `google/deit-base-distilled-patch16-224`
- Any ViT or DeiT model from Hugging Face

**Code Example**:
```python
adapter = ViTAdapter("google/vit-base-patch16-224", image_size=224)
processed_image, target = adapter(image, {"class_labels": torch.tensor([1])})
# processed_image: torch.Tensor (3, 224, 224) with normalization
# target: {"class_labels": torch.tensor([1])}
```

### 3. ConvNeXTAdapter

**Purpose**: Handles preprocessing for ConvNeXT models.

**Why This Design**:
- ConvNeXT models have similar preprocessing requirements to ViT but are optimized for CNN architectures
- Uses the same AutoImageProcessor approach for consistency
- Ensures proper normalization and resizing for ConvNeXT performance

**Input Requirements**:
- Raw PIL Image
- Standard target format: `{"class_labels": torch.tensor}`

**Output Format**:
- Image: Preprocessed tensor with ConvNeXT-specific normalization
- Target: Unchanged standard format

**Key Preprocessing Steps**:
1. **Resize**: Scales image to specified size (default 224x224)
2. **Rescale**: Converts pixel values to [0, 1] range
3. **Normalize**: Applies ConvNeXT-specific normalization
4. **Pad**: Adds padding if needed

**Supported Models**:
- `facebook/convnext-tiny-224`
- `facebook/convnext-small-224`
- `facebook/convnext-base-224`
- `facebook/convnext-large-224`
- Any ConvNeXT model from Hugging Face

**Code Example**:
```python
adapter = ConvNeXTAdapter("facebook/convnext-base-224", image_size=224)
processed_image, target = adapter(image, {"class_labels": torch.tensor([1])})
# processed_image: torch.Tensor (3, 224, 224) with ConvNeXT normalization
# target: {"class_labels": torch.tensor([1])}
```

### 4. SwinAdapter

**Purpose**: Handles preprocessing for Swin Transformer models.

**Why This Design**:
- Swin Transformers use hierarchical vision transformers with specific preprocessing requirements
- Maintains consistency with other transformer adapters
- Ensures proper handling of the hierarchical structure

**Input Requirements**:
- Raw PIL Image
- Standard target format: `{"class_labels": torch.tensor}`

**Output Format**:
- Image: Preprocessed tensor optimized for Swin architecture
- Target: Unchanged standard format

**Key Preprocessing Steps**:
1. **Resize**: Scales image to specified size (default 224x224)
2. **Rescale**: Converts pixel values to [0, 1] range
3. **Normalize**: Applies Swin-specific normalization
4. **Pad**: Adds padding optimized for hierarchical processing

**Supported Models**:
- `microsoft/swin-tiny-patch4-window7-224`
- `microsoft/swin-small-patch4-window7-224`
- `microsoft/swin-base-patch4-window7-224`
- `microsoft/swin-large-patch4-window7-224`
- Any Swin Transformer model from Hugging Face

**Code Example**:
```python
adapter = SwinAdapter("microsoft/swin-base-patch4-window7-224", image_size=224)
processed_image, target = adapter(image, {"class_labels": torch.tensor([1])})
# processed_image: torch.Tensor (3, 224, 224) with Swin normalization
# target: {"class_labels": torch.tensor([1])}
```

## Input Adapter Example

```python
class ViTInputAdapter(BaseAdapter):
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # ...
        return processed_image, adapted_target
```

## Output Adapter Example

```python
class ViTOutputAdapter:
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # ...
        return standardized_outputs
```

## Factory Functions

```python
input_adapter = get_input_adapter("google/vit-base-patch16-224")  # Returns ViTInputAdapter
output_adapter = get_output_adapter("google/vit-base-patch16-224")  # Returns ViTOutputAdapter
```

## Factory Function

### get_adapter()

**Purpose**: Automatically selects the appropriate adapter based on model name.

**Why This Design**:
- Eliminates the need to manually specify adapters
- Uses model name patterns to determine the correct adapter
- Provides fallback to NoOpAdapter for unknown models

**Selection Logic**:
```python
def get_adapter(model_name: str, image_size: int = 224) -> BaseAdapter:
    model_name_lower = model_name.lower()
    
    if "vit" in model_name_lower:
        return ViTAdapter(model_name=model_name, image_size=image_size)
    elif "convnext" in model_name_lower:
        return ConvNeXTAdapter(model_name=model_name, image_size=image_size)
    elif "swin" in model_name_lower:
        return SwinAdapter(model_name=model_name, image_size=image_size)
    else:
        return NoOpAdapter()
```

**Usage Example**:
```python
# Automatically selects ViTAdapter
adapter = get_adapter("google/vit-base-patch16-224")

# Automatically selects ConvNeXTAdapter
adapter = get_adapter("facebook/convnext-base-224")

# Falls back to NoOpAdapter
adapter = get_adapter("unknown_model")
```

## Design Principles

### 1. Consistency
- All adapters follow the same interface
- Standardized input/output formats
- Consistent error handling

### 2. Flexibility
- Easy to add new model support
- Configurable image sizes
- Fallback mechanisms

### 3. Performance
- Efficient preprocessing using Hugging Face processors
- Minimal overhead for simple models
- Optimized for batch processing

### 4. Maintainability
- Clear separation of concerns
- Well-documented design decisions
- Easy to test and debug

## Adding New Adapters

To add support for a new classification model:

1. **Create Input Adapter**:
```python
class NewModelAdapter(BaseAdapter):
    def __init__(self, model_name: str, image_size: int = 224):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        processed = self.processor(image, return_tensors="pt")
        return processed.pixel_values.squeeze(0), target
```

2. **Update Factory Function**:
```python
def get_adapter(model_name: str, image_size: int = 224) -> BaseAdapter:
    model_name_lower = model_name.lower()
    
    if "new_model" in model_name_lower:
        return NewModelAdapter(model_name=model_name, image_size=image_size)
    # ... existing logic
```

3. **Add Tests**: Create comprehensive tests for the new adapter

4. **Update Documentation**: Add the new adapter to this documentation

## Best Practices

1. **Use AutoImageProcessor**: Leverage Hugging Face's optimized processors
2. **Maintain Standard Format**: Keep target format consistent across adapters
3. **Handle Edge Cases**: Provide fallbacks for unknown models
4. **Document Assumptions**: Clearly document model-specific requirements
5. **Test Thoroughly**: Ensure adapters work with various input sizes and formats 
# Semantic Segmentation Adapters

This document explains the semantic segmentation adapters in our framework, how they work, and why they were designed in specific ways to handle different semantic segmentation model architectures.

## Overview

Semantic segmentation adapters are responsible for converting input images and segmentation masks into the format expected by different semantic segmentation models. Each adapter handles the specific preprocessing requirements needed for different segmentation architectures, ensuring proper mask handling and pixel-level classification.

## Adapter Types

### 1. NoOpAdapter

**Purpose**: A minimal adapter that performs basic preprocessing without model-specific transformations.

**Why This Design**: 
- Acts as a fallback for models that don't require specific preprocessing
- Provides basic tensor conversion and resizing
- Maintains the original mask format
- Suitable for torchvision-based segmentation models

**Input Requirements**:
- Raw PIL Image
- Segmentation target: `{"semantic_masks": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Resized and converted to tensor
- Target: Unchanged segmentation format

**Supported Models**:
- torchvision segmentation models (DeepLabV3, FCN, etc.)
- Custom models with standard preprocessing requirements

**Code Example**:
```python
adapter = NoOpAdapter()
processed_image, target = adapter(image, {
    "semantic_masks": torch.tensor([[0, 1, 2], [0, 1, 2]]),  # HxW mask
    "image_id": torch.tensor([0])
})
```

### 2. SegFormerAdapter

**Purpose**: Handles preprocessing for SegFormer models.

**Why This Design**:
- SegFormer is a transformer-based semantic segmentation model
- Requires specific preprocessing including resizing, normalization, and padding
- Uses Hugging Face's AutoImageProcessor optimized for transformer-based segmentation
- Handles the unique requirements of hierarchical vision transformers

**Input Requirements**:
- Raw PIL Image
- Segmentation target: `{"semantic_masks": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Preprocessed tensor with SegFormer-specific normalization and padding
- Target: Unchanged segmentation format

**Key Preprocessing Steps**:
1. **Resize**: Scales image to specified size (default 512x512)
2. **Rescale**: Converts pixel values to [0, 1] range
3. **Normalize**: Applies ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Pad**: Adds padding to maintain aspect ratio

**Why SegFormer-Specific Design**:
- SegFormer uses a hierarchical transformer architecture
- Requires specific image sizes for optimal performance
- Uses transformer-specific normalization
- Optimized for multi-scale feature extraction

**Supported Models**:
- `nvidia/segformer-b0-finetuned-ade-512-512`
- `nvidia/segformer-b1-finetuned-ade-512-512`
- `nvidia/segformer-b2-finetuned-ade-512-512`
- `nvidia/segformer-b3-finetuned-ade-512-512`
- `nvidia/segformer-b4-finetuned-ade-512-512`
- `nvidia/segformer-b5-finetuned-ade-512-512`
- Any SegFormer model from Hugging Face

**Code Example**:
```python
adapter = SegFormerAdapter("nvidia/segformer-b0-finetuned-ade-512-512", image_size=512)
processed_image, target = adapter(image, {
    "semantic_masks": torch.tensor([[0, 1, 2], [0, 1, 2]]),
    "image_id": torch.tensor([0])
})
# processed_image: torch.Tensor (3, 512, 512) with normalization
# target: {"semantic_masks": torch.tensor, "image_id": torch.tensor([0])}
```

### 3. DeepLabV3Adapter

**Purpose**: Handles preprocessing for DeepLabV3 models.

**Why This Design**:
- DeepLabV3 is a CNN-based semantic segmentation model
- Requires specific preprocessing optimized for CNN architectures
- Uses Hugging Face's AutoImageProcessor for consistency
- Handles the unique requirements of atrous convolution networks

**Input Requirements**:
- Raw PIL Image
- Segmentation target: `{"semantic_masks": torch.tensor, "image_id": torch.tensor}`

**Output Format**:
- Image: Preprocessed tensor with DeepLabV3-specific normalization
- Target: Unchanged segmentation format

**Key Preprocessing Steps**:
1. **Resize**: Scales image to specified size (default 512x512)
2. **Rescale**: Converts pixel values to [0, 1] range
3. **Normalize**: Applies DeepLabV3-specific normalization
4. **Pad**: Adds padding optimized for atrous convolution

**Why DeepLabV3-Specific Design**:
- DeepLabV3 uses atrous convolutions for multi-scale feature extraction
- Requires specific image sizes for optimal atrous convolution performance
- Uses CNN-specific normalization
- Optimized for dense prediction tasks

**Supported Models**:
- `microsoft/deeplabv3-resnet-50`
- `microsoft/deeplabv3-resnet-101`
- `microsoft/deeplabv3-large-resnet-101`
- Any DeepLabV3 model from Hugging Face

**Code Example**:
```python
adapter = DeepLabV3Adapter("microsoft/deeplabv3-resnet-50", image_size=512)
processed_image, target = adapter(image, {
    "semantic_masks": torch.tensor([[0, 1, 2], [0, 1, 2]]),
    "image_id": torch.tensor([0])
})
# processed_image: torch.Tensor (3, 512, 512) with DeepLabV3 normalization
# target: {"semantic_masks": torch.tensor, "image_id": torch.tensor([0])}
```

## Output Adapters

### SemanticSegmentationOutputAdapter

**Purpose**: Converts semantic segmentation model outputs to a standardized format for metric computation.

**Why This Design**:
- Different segmentation models return outputs in different formats
- Standardizes the output format for consistent metric computation
- Handles both loss and prediction extraction
- Ensures proper mask format for evaluation

**Input Format**: Raw model outputs (varies by model)

**Output Format**:
```python
{
    "loss": torch.tensor,  # Training loss
    "logits": torch.tensor,  # Raw prediction logits (B, C, H, W)
    "loss_dict": dict  # Additional loss components
}
```

**Key Functions**:
1. **adapt_output()**: Extracts loss and logits from model outputs
2. **adapt_targets()**: Converts targets to model-specific format
3. **format_predictions()**: Formats outputs for metric computation
4. **format_targets()**: Formats targets for metric computation

**Why Standardized Output is Important**:
- Enables consistent evaluation metrics (IoU, Dice, Accuracy)
- Ensures compatibility with existing evaluation tools
- Maintains consistency across different segmentation models
- Simplifies metric computation and comparison

**Code Example**:
```python
output_adapter = SemanticSegmentationOutputAdapter()
adapted_outputs = output_adapter.adapt_output(model_outputs)
# adapted_outputs: {"loss": tensor, "logits": tensor, "loss_dict": {}}
```

## Factory Function

### get_semantic_adapter()

**Purpose**: Automatically selects the appropriate adapter based on model name.

**Why This Design**:
- Eliminates the need to manually specify adapters
- Uses model name patterns to determine the correct adapter
- Provides fallback to NoOpAdapter for unknown models
- Ensures optimal preprocessing for each model type

**Selection Logic**:
```python
def get_semantic_adapter(model_name: str, image_size: int = 512) -> BaseAdapter:
    model_name_lower = model_name.lower()
    
    if "segformer" in model_name_lower:
        return SegFormerAdapter(model_name, image_size)
    elif "deeplab" in model_name_lower:
        return DeepLabV3Adapter(model_name, image_size)
    else:
        return NoOpAdapter()
```

**Usage Example**:
```python
# Automatically selects SegFormerAdapter
adapter = get_semantic_adapter("nvidia/segformer-b0-finetuned-ade-512-512")

# Automatically selects DeepLabV3Adapter
adapter = get_semantic_adapter("microsoft/deeplabv3-resnet-50")

# Falls back to NoOpAdapter
adapter = get_semantic_adapter("unknown_model")
```

## Mask Handling

### Why Mask Handling is Critical

**Different Model Requirements**:
- **SegFormer**: Expects masks in specific format for transformer processing
- **DeepLabV3**: Uses CNN-optimized mask format
- **Standard Format**: Uses pixel-level class labels

**Mask Processing**:
1. **Original**: COCO format segmentation masks
2. **Processing**: Convert to model-specific format
3. **Evaluation**: Convert back to standard format for metrics

**Example Mask Processing**:
```python
# Original COCO mask: Polygon or RLE format
# Standard format: HxW tensor with class labels
# Model-specific: May require additional processing for optimal performance
```

## Design Principles

### 1. Model-Specific Optimization
- Each adapter is optimized for its target model architecture
- Handles model-specific preprocessing requirements
- Ensures optimal performance for each model type
- Maintains model-specific normalization and resizing

### 2. Mask Compatibility
- All adapters maintain mask format compatibility
- Enables standard evaluation metrics (IoU, Dice, Accuracy)
- Supports existing segmentation datasets and tools
- Ensures proper pixel-level classification

### 3. Transformer vs CNN Handling
- **Transformer Models** (SegFormer): Require specific preprocessing for attention mechanisms
- **CNN Models** (DeepLabV3): Optimized for convolutional operations
- **Hybrid Models**: Handle both transformer and CNN requirements

### 4. Flexibility
- Easy to add new segmentation model support
- Configurable image sizes
- Fallback mechanisms for unknown models
- Support for various mask formats

## Adding New Semantic Segmentation Adapters

To add support for a new semantic segmentation model:

1. **Create Input Adapter**:
```python
class NewSegmentationAdapter(BaseAdapter):
    def __init__(self, model_name: str, image_size: int = 512):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # Model-specific preprocessing
        processed = self.processor(image, return_tensors="pt")
        
        # Mask processing if needed
        adapted_target = self._process_masks(target)
        
        return processed.pixel_values.squeeze(0), adapted_target
    
    def _process_masks(self, target: Dict) -> Dict:
        # Model-specific mask processing
        # ...
        return adapted_target
```

2. **Create Output Adapter**:
```python
class NewSegmentationOutputAdapter(OutputAdapter):
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract model-specific outputs
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "loss_dict": getattr(outputs, "loss_dict", {})
        }
```

3. **Update Factory Function**:
```python
def get_semantic_adapter(model_name: str, image_size: int = 512) -> BaseAdapter:
    model_name_lower = model_name.lower()
    
    if "new_model" in model_name_lower:
        return NewSegmentationAdapter(model_name, image_size)
    # ... existing logic
```

## Best Practices

1. **Understand Model Architecture**: Study the model's preprocessing requirements thoroughly
2. **Mask Format Handling**: Ensure proper mask format conversion
3. **Evaluation Compatibility**: Maintain compatibility with standard evaluation metrics
4. **Error Handling**: Provide robust error handling for edge cases
5. **Testing**: Test with various image sizes and mask formats
6. **Documentation**: Document model-specific requirements and assumptions
7. **Performance Optimization**: Use model-specific optimizations for better performance 
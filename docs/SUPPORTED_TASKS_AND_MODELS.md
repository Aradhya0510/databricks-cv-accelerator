# Supported Tasks and Models

The framework supports multiple computer vision tasks with **zero-knowledge model management**. Simply specify the model name in your configuration, and the framework automatically handles all the complexity.

## 📋 Available Tasks

| Task | Description | AutoModel Class | Documentation |
|------|-------------|-----------------|---------------|
| **Object Detection** | Detect objects with bounding boxes | `AutoModelForObjectDetection` | [🔗 AutoModelForObjectDetection](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/auto#transformers.AutoModelForObjectDetection) |
| **Image Classification** | Classify images into categories | `AutoModelForImageClassification` | [🔗 AutoModelForImageClassification](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/auto#transformers.AutoModelForImageClassification) |
| **Semantic Segmentation** | Pixel-level classification | `AutoModelForSemanticSegmentation` | [🔗 AutoModelForSemanticSegmentation](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/auto#transformers.AutoModelForSemanticSegmentation) |
| **Instance Segmentation** | Individual object segmentation | `AutoModelForInstanceSegmentation` | [🔗 AutoModelForInstanceSegmentation](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/auto#transformers.AutoModelForInstanceSegmentation) |
| **Universal Segmentation** | Unified segmentation (panoptic/instance) | `AutoModelForUniversalSegmentation` | [🔗 AutoModelForUniversalSegmentation](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/auto#transformers.AutoModelForUniversalSegmentation) |

## 🚀 Popular Model Examples

### Object Detection
- `facebook/detr-resnet-50` - DETR with ResNet-50 backbone
- `hustvl/yolos-tiny` - YOLOS tiny model
- `facebook/detr-resnet-101` - DETR with ResNet-101 backbone
- `microsoft/table-transformer-detection` - Table detection model

### Image Classification
- `google/vit-base-patch16-224` - Vision Transformer (ViT)
- `microsoft/resnet-50` - ResNet-50
- `facebook/convnext-base-224` - ConvNeXT Base
- `microsoft/swin-base-patch4-window7-224` - Swin Transformer

### Semantic Segmentation
- `nvidia/segformer-b0-finetuned-ade-512-512` - SegFormer B0
- `facebook/detr-resnet-50-panoptic` - DETR Panoptic
- `facebook/mask2former-swin-base-coco-instance` - Mask2Former Instance

## 🔧 How to Use Any Model

Simply change the `model_name` in your configuration:

```yaml
model:
  model_name: "facebook/detr-resnet-50"  # Change this to any model from the links above
  task_type: "detection"
```

The framework automatically:
- ✅ Downloads the model from Hugging Face
- ✅ Sets up the correct AutoModel class
- ✅ Configures model-specific preprocessing
- ✅ Handles input/output format conversion
- ✅ Manages all dependencies

## 📚 Model Selection Guide

### For Object Detection
- **DETR models**: Good for complex scenes, transformer-based
- **YOLOS models**: Fast inference, good for real-time applications
- **Table Transformer**: Specialized for document/table detection

### For Image Classification
- **ViT models**: Excellent performance, transformer-based
- **ResNet models**: Proven architecture, good baseline
- **ConvNeXT models**: Modern CNN architecture, excellent performance
- **Swin Transformer**: Hierarchical vision transformer

### For Segmentation
- **SegFormer**: Efficient transformer for semantic segmentation
- **Mask2Former**: Unified framework for instance/panoptic segmentation

## 🎯 Zero-Knowledge Model Management

The framework implements **complete model abstraction** through Hugging Face Auto-classes:

### What You DON'T Need to Know:
- ❌ Model architecture internals (DETR, YOLOS, ResNet, etc.)
- ❌ Model-specific preprocessing requirements
- ❌ Input/output format differences
- ❌ Dependencies and compatibility issues

### What the Framework Handles Automatically:
- ✅ **Model Download**: Automatic download from Hugging Face Hub
- ✅ **Architecture Detection**: Correct AutoModel class selection
- ✅ **Preprocessing**: Model-specific image processing and normalization
- ✅ **Format Conversion**: Input/output format standardization
- ✅ **Dependency Management**: All required libraries and versions

### Example: Switching Models
```yaml
# Just change this one line in your config:
model:
  model_name: "facebook/detr-resnet-50"  # Change to:
  model_name: "hustvl/yolos-tiny"       # That's it!

# No other changes needed - adapters handle everything automatically
```

## 🔗 Finding More Models

Each task's documentation link above provides:
- **Complete API reference** for the AutoModel class
- **Usage examples** and best practices
- **Model compatibility** information
- **Parameter documentation** and configuration options

For the latest models and updates, visit the [Hugging Face Models Hub](https://huggingface.co/models) and filter by your desired task.

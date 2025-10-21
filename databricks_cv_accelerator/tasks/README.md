# Segmentation Tasks

This directory contains three separate segmentation modules for different types of segmentation tasks:

## Module Structure

### 1. Semantic Segmentation (`semantic_segmentation/`)
- **Purpose**: Pixel-level classification where each pixel is assigned a class label
- **Use Case**: Background/foreground separation, scene understanding
- **Models**: SegFormer, DeepLabV3, etc.
- **Output**: Single mask per image with class labels for each pixel

### 2. Instance Segmentation (`instance_segmentation/`)
- **Purpose**: Object-level segmentation where each instance of an object is separately identified
- **Use Case**: Object counting, individual object analysis
- **Models**: Mask2Former, Mask R-CNN, etc.
- **Output**: Multiple masks per image, one for each object instance

### 3. Universal Segmentation (`universal_segmentation/`)
- **Purpose**: Combines semantic and instance segmentation for complete scene understanding
- **Use Case**: Complete scene parsing, autonomous driving, robotics
- **Models**: Mask2Former, Universal FPN, etc.
- **Output**: Unified segmentation map with both "stuff" (background) and "things" (objects)

## Module Components

Each segmentation module contains the following components:

### Core Classes
- **Model**: PyTorch Lightning module for training and inference
- **DataModule**: Data loading and preprocessing
- **Evaluator**: Model evaluation and metrics computation
- **Inference**: Single image and batch inference
- **Adapters**: Model-specific data and output adapters

### Configuration
- **ModelConfig**: Model hyperparameters and settings
- **DataConfig**: Data loading and augmentation settings

## Usage Examples

### Semantic Segmentation

```python
from src.tasks.semantic_segmentation import (
    SemanticSegmentationModel,
    SemanticSegmentationDataModule,
    SemanticSegmentationModelConfig,
    SemanticSegmentationDataConfig
)

# Configuration
model_config = SemanticSegmentationModelConfig(
    model_name="nvidia/segformer-b0-finetuned-ade-512-512",
    num_classes=150,
    learning_rate=1e-4
)

data_config = SemanticSegmentationDataConfig(
    data_path="/path/to/data",
    annotation_file="/path/to/annotations.json",
    batch_size=8
)

# Initialize model and data
model = SemanticSegmentationModel(model_config)
data_module = SemanticSegmentationDataModule(data_config)

# Training
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```

### Instance Segmentation

```python
from src.tasks.instance_segmentation import (
    InstanceSegmentationModel,
    InstanceSegmentationDataModule,
    InstanceSegmentationModelConfig,
    InstanceSegmentationDataConfig
)

# Configuration
model_config = InstanceSegmentationModelConfig(
    model_name="facebook/mask2former-swin-base-coco-instance",
    num_classes=91,
    learning_rate=1e-4
)

data_config = InstanceSegmentationDataConfig(
    data_path="/path/to/data",
    annotation_file="/path/to/annotations.json",
    batch_size=4  # Smaller batch size for instance segmentation
)

# Initialize model and data
model = InstanceSegmentationModel(model_config)
data_module = InstanceSegmentationDataModule(data_config)

# Training
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```

### Universal Segmentation

```python
from src.tasks.universal_segmentation import (
    UniversalSegmentationModel,
    UniversalSegmentationDataModule,
    UniversalSegmentationModelConfig,
    UniversalSegmentationDataConfig
)

# Configuration
model_config = UniversalSegmentationModelConfig(
    model_name="facebook/mask2former-swin-base-coco-panoptic",
    num_classes=133,  # COCO universal classes
    learning_rate=1e-4
)

data_config = UniversalSegmentationDataConfig(
    data_path="/path/to/data",
    annotation_file="/path/to/annotations.json",
    batch_size=4
)

# Initialize model and data
model = UniversalSegmentationModel(model_config)
data_module = UniversalSegmentationDataModule(data_config)

# Training
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```

## Evaluation

Each module provides evaluation capabilities:

```python
from src.tasks.semantic_segmentation import SemanticSegmentationEvaluator

# Evaluate model
evaluator = SemanticSegmentationEvaluator(
    model_path="path/to/checkpoint.ckpt",
    config_path="path/to/config.yaml"
)

metrics = evaluator.evaluate(data_module)
evaluator.plot_metrics(metrics, output_dir="evaluation_results")
```

## Inference

Each module provides inference capabilities:

```python
from src.tasks.semantic_segmentation import SemanticSegmentationInference
import cv2

# Initialize inference
inference = SemanticSegmentationInference(
    model_path="path/to/checkpoint.ckpt",
    config_path="path/to/config.yaml"
)

# Load image
image = cv2.imread("path/to/image.jpg")

# Run inference
mask = inference.predict(image)

# Visualize results
inference.visualize(image, mask, output_path="output.jpg")
```

## Key Differences

| Aspect | Semantic | Instance | Universal |
|--------|----------|----------|----------|
| **Output Type** | Single mask | Multiple masks | Unified mask |
| **Object Handling** | Class-based | Instance-based | Both |
| **Background** | Included | Excluded | Included |
| **Metrics** | IoU, Dice | IoU, Dice, mAP | IoU, Dice, mAP |
| **Use Cases** | Scene parsing | Object detection | Complete understanding |

## Migration from Original Module

The original unified segmentation module has been removed. Use the specific modules instead:

### New Code
```python
# For semantic segmentation
from src.tasks.semantic_segmentation import SemanticSegmentationModel
model = SemanticSegmentationModel(config)

# For instance segmentation  
from src.tasks.instance_segmentation import InstanceSegmentationModel
model = InstanceSegmentationModel(config)

# For panoptic segmentation
from src.tasks.panoptic_segmentation import PanopticSegmentationModel
model = PanopticSegmentationModel(config)
```

## Benefits of Separation

1. **Simplified Code**: Each module focuses on one task type
2. **Better Performance**: Optimized for specific segmentation types
3. **Easier Maintenance**: Clear separation of concerns
4. **Focused Metrics**: Task-specific evaluation metrics
5. **Reduced Dependencies**: Only load what you need
6. **Better Documentation**: Clear purpose for each module

## Configuration Files

Each module supports YAML configuration files. Example:

```yaml
# semantic_segmentation_config.yaml
model:
  model_name: "nvidia/segformer-b0-finetuned-ade-512-512"
  num_classes: 150
  learning_rate: 1e-4
  weight_decay: 0.01
  epochs: 10

data:
  data_path: "/path/to/data"
  annotation_file: "/path/to/annotations.json"
  image_size: 512
  batch_size: 8
  num_workers: 4
```

## Supported Models

### Semantic Segmentation
- SegFormer (nvidia/segformer-*)
- DeepLabV3 (google/deeplabv3-*)
- U-Net variants
- Custom semantic segmentation models

### Instance Segmentation
- Mask2Former (facebook/mask2former-*)
- Mask R-CNN variants
- YOLO with segmentation heads
- Custom instance segmentation models

### Panoptic Segmentation
- Mask2Former (facebook/mask2former-*)
- Panoptic FPN
- DETR with panoptic heads
- Custom panoptic segmentation models 
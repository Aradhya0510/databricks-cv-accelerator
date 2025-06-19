# Databricks Computer Vision Architecture

An advanced, modular, and extensible reference architecture designed to simplify the adoption and deployment of sophisticated computer vision pipelines. This architecture leverages standard frameworks and protocols, promoting consistent and efficient workflows by integrating Databricks with PyTorch Lightning for structured model training, Ray for distributed computation and hyperparameter optimization, Hugging Face Transformers for a standardized, robust model repository, MLflow for uniform experiment tracking and model logging, and PyCOCOTools for standardized data annotations and management.

---

## 🎯 Why This Project?

Implementing production-ready computer vision solutions can be complex. This architecture aims to:

* **Simplify Deployment:** Abstract the complexities of distributed training, hyperparameter tuning, model tracking, and monitoring.
* **Best Practices:** Integrate industry-leading tools and frameworks to ensure scalability, reproducibility, and maintainability.
* **Ease of Adoption:** Provide clear, structured, and easy-to-follow workflows for rapid development and deployment.
* **Model Agnostic:** Support multiple model architectures through a unified adapter system.

---

## 🚀 Technology Stack and Its Significance

The architecture integrates standardized frameworks and tools to achieve a robust and maintainable computer vision pipeline:

* **PyTorch Lightning**: Establishes a standardized protocol for model training, validation, and testing, reducing boilerplate code and ensuring reproducibility through structured code.

* **Ray**: Provides standardized and scalable distributed training and hyperparameter tuning capabilities, crucial for leveraging large-scale computing clusters efficiently.

* **Hugging Face Transformers**: Serves as a unified model repository offering well-established architectures and pretrained models, ensuring rapid integration and deployment of state-of-the-art models such as DETR and YOLO.

* **MLflow**: Implements a uniform system for experiment tracking, logging metrics, and managing model versions, enhancing reproducibility, traceability, and deployment readiness.

* **Albumentations**: Provides standardized data augmentation techniques to enhance model generalization, consistency, and performance across diverse datasets.

* **PyCOCOTools**: Standardizes data annotation and management using the widely recognized COCO format, enabling consistent data handling, evaluation metrics, and interoperability across datasets and tasks.

---

## 🧩 Modularity and Extensibility

The project emphasizes modularity and extensibility through clear abstractions:

* **UnifiedTrainer**: Abstracts training logic, seamlessly managing local and distributed environments.
* **DetectionModel & DetectionDataModule**: Separately handle data and model logic, promoting independent maintenance and ease of integration.
* **Adapter System**: Facilitates straightforward integration of new Hugging Face models through data and output adapters.

---

## 🔧 How to Introduce New Models via Adapters

To integrate a new Hugging Face detection model into this architecture, follow these steps:

### Step 1: Create Data and Output Adapters

The architecture uses two types of adapters:

#### Data Adapter (Input Processing)
Implement the `BaseAdapter` abstract class to handle input data processing:

```python
from src.tasks.detection.adapters import BaseAdapter

class YourModelDataAdapter(BaseAdapter):
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # Process image and target for your specific model
        # Return processed image tensor and adapted target dictionary
        return processed_image, adapted_target
```

#### Output Adapter (Output Processing)
Create an output adapter for your model:

```python
class YourModelOutputAdapter:
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # Convert model outputs to standardized format
        return {
            "loss": outputs.get("loss"),
            "pred_boxes": outputs.pred_boxes,
            "pred_logits": outputs.logits,
            "loss_dict": outputs.get("loss_dict", {})
        }
    
    def format_predictions(self, outputs: Dict[str, Any], batch: Optional[Dict[str, Any]] = None) -> List[Dict[str, torch.Tensor]]:
        # Format predictions for metric computation
        # Return list of prediction dictionaries
        pass
    
    def format_targets(self, targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        # Format targets for metric computation
        # Return list of target dictionaries
        pass
```

### Step 2: Register Your Adapters

Update the adapter factory in `src/tasks/detection/adapters.py`:

```python
def get_adapter(model_name: str, image_size: int = 800) -> BaseAdapter:
    """Get the appropriate adapter for a model."""
    if "your_model" in model_name.lower():
        return YourModelDataAdapter(model_name=model_name, image_size=image_size)
    elif "detr" in model_name.lower():
        return DETRAdapter(model_name=model_name, image_size=image_size)
    else:
        return NoOpAdapter()
```

### Step 3: Configure Your Model

Create a new configuration in your YAML config:

```yaml
model:
  task_type: detection
  model_name: your_model_identifier
  num_classes: 80
  confidence_threshold: 0.7
  iou_threshold: 0.5
  max_detections: 100
  learning_rate: 1e-4
  weight_decay: 1e-4
  epochs: 300
  image_size: 800
```

---

## 🚦 Getting Started

### Prerequisites

* Databricks workspace with Unity Catalog enabled
* Python 3.8+ environment
* Access to GPU resources (recommended for training)

### Installation

Clone the repository into your Databricks workspace:

```bash
git clone https://github.com/Aradhya0510/databricks-cv-architecture.git
cd databricks-cv-architecture
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Data Setup

1. **Prepare your dataset** in COCO format:
   ```
   /path/to/dataset/
   ├── images/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── annotations.json
   ```

2. **Upload to Unity Catalog volume**:
   ```
   /Volumes/your_catalog/your_schema/your_volume/
   ├── data/
   │   ├── train/
   │   │   ├── images/
   │   │   └── annotations.json
   │   ├── val/
   │   │   ├── images/
   │   │   └── annotations.json
   │   └── test/
   │       ├── images/
   │       └── annotations.json
   ├── configs/
   ├── checkpoints/
   ├── logs/
   └── results/
   ```

### Configuration

Update the configuration file `configs/detection_config.yaml`:

```yaml
# Model Configuration
model:
  model_name: "facebook/detr-resnet-50"
  task_type: "detection"
  num_classes: 80
  pretrained: true
  confidence_threshold: 0.7
  iou_threshold: 0.5
  max_detections: 100
  learning_rate: 1e-4
  weight_decay: 1e-4
  epochs: 300
  image_size: 800

# Data Configuration
data:
  data_path: "/Volumes/your_catalog/your_schema/your_volume/data/train/images"
  annotation_file: "/Volumes/your_catalog/your_schema/your_volume/data/train/annotations.json"
  batch_size: 2
  num_workers: 4
  model_name: "facebook/detr-resnet-50"

# Training Configuration
training:
  max_epochs: 300
  early_stopping_patience: 20
  monitor_metric: "val_map"
  monitor_mode: "max"
  checkpoint_dir: "/Volumes/your_catalog/your_schema/your_volume/checkpoints/detection"
  save_top_k: 3
  distributed: true
  use_gpu: true
```

### Training

The project includes a series of notebooks that demonstrate the complete workflow:

1. **`notebooks/01_data_preparation.py`**: Dataset preparation and preprocessing
2. **`notebooks/02_model_training.py`**: Model training and evaluation

#### Quick Start Training

```python
# Load configuration
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule
from src.training.trainer import UnifiedTrainer
import yaml

# Load config
with open('configs/detection_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model and data module
model = DetectionModel(config=config['model'])
data_module = DetectionDataModule(config=config['data'])

# Setup data
data_module.setup('fit')

# Initialize trainer
trainer = UnifiedTrainer(
    config=config,
    model=model,
    data_module=data_module
)

# Start training
trainer.train()
```

---

## 📊 Current Features

### ✅ Implemented
- **DETR Model Support**: Full support for DETR (DEtection TRansformer) models
- **COCO Dataset Integration**: Seamless integration with COCO format datasets
- **Distributed Training**: Support for multi-GPU and multi-node training via Ray
- **MLflow Integration**: Comprehensive experiment tracking and model logging
- **Modular Architecture**: Clean separation of data, model, and training logic
- **Adapter System**: Extensible adapter framework for new model architectures
- **Configuration Management**: YAML-based configuration system
- **Metrics Tracking**: Mean Average Precision (mAP) and related metrics
- **Checkpointing**: Automatic model checkpointing and early stopping

### 🚧 In Progress
- **Additional Model Support**: YOLO, Faster R-CNN, and other architectures
- **Segmentation Tasks**: Support for semantic and instance segmentation
- **Classification Tasks**: Support for image classification
- **Advanced Augmentation**: Enhanced data augmentation strategies

---

## 🌱 Future Directions

* **Multi-task Learning**: Support for models that can handle detection, segmentation, and classification simultaneously
* **Advanced Adapters**: Plugin-based adapter registry for easier extensibility
* **Model Compression**: Integration with model compression and quantization techniques
* **Real-time Inference**: Optimized inference pipelines for production deployment
* **Custom Loss Functions**: Support for custom loss functions and training strategies

---

## 📚 Documentation & Contributions

Contributions are welcomed! Please document thoroughly and maintain consistent coding standards when contributing.

### Development Guidelines

1. **Follow the adapter pattern** for new model integrations
2. **Maintain model agnosticism** in core components
3. **Add comprehensive tests** for new features
4. **Update documentation** for any API changes
5. **Use type hints** for better code maintainability

---

### 📬 Questions & Feedback

Reach out via GitHub issues or email for support and suggestions!

## Contributing

We welcome contributions to improve the reference architecture. Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
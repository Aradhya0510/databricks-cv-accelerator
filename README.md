# Databricks Computer Vision Architecture

An advanced, modular, and extensible computer vision framework designed to significantly reduce the complexity involved in building, training, and deploying computer vision models on Databricks. This project aims to serve as a reference implementation for best practices, abstracting away many of the difficult decisions related to model architecture, dataset format, training orchestration, distributed computing, and experiment tracking. By using this framework, practitioners can more easily transition from problem formulation to model deployment using well-defined and reusable components that promote clarity, scalability, and productivity.

---

## 🎯 Vision and Approach

Computer vision projects are often burdened by the overwhelming number of decisions that practitioners must make before getting to meaningful experimentation. From choosing the right model architecture and preprocessing pipeline to configuring training infrastructure and tracking results, these decisions can stall progress and increase costs. This architecture solves that problem through:

* **Reducing cognitive and setup overhead:** The framework expects datasets in the MS COCO format—an image folder and a single `annotations.json`. For users bringing in new model architectures, only a small adapter class is needed to plug the model into the framework.
* **Promoting best practices:** Built on PyTorch Lightning, Ray, MLflow, Hugging Face Transformers, and Albumentations, the framework incorporates community-standard tools to ensure reproducibility, traceability, and scalability.
* **Enabling rapid experimentation:** Configuration files define models, datasets, and training settings, allowing users to switch tasks or models without rewriting core logic.

Users can walk through the provided example notebooks to:

* Set up their Databricks environment and Unity Catalog schema
* Upload and register datasets under Unity Catalog Volumes
* Initialize model and dataloader objects with a few lines of code
* Launch a full training loop and monitor experiments via the MLflow UI

Once training is running, checkpoints are automatically saved to configured volume paths via Lightning's `ModelCheckpoint` callback. Metrics, hyperparameters, and model artifacts are tracked in MLflow for easy reproducibility and comparison.

---

## 🚀 Technology Stack and Its Significance

* **Lightning**: Simplifies model training by structuring code for better maintainability and automatic integration of callbacks like early stopping and checkpointing.
* **Ray**: Powers distributed training and large-scale hyperparameter tuning. Enables multi-GPU/multi-node workloads on Databricks.
* **Hugging Face Transformers**: Provides high-quality pre-trained vision models such as DETR, YOLO, ViT, SegFormer, and Mask2Former with minimal setup.
* **MLflow**: Tracks parameters, metrics, models, and artifacts. Crucial for model governance, collaboration, and auditing.
* **Albumentations**: Standardizes and simplifies image augmentations to improve model robustness across datasets.
* **PyCOCOTools**: Provides reliable annotation parsing, evaluation, and visualization tools using the COCO format.

---

## 🧩 Modularity and Extensibility

* **UnifiedTrainer**: Handles all training logic, seamlessly switching between local and distributed modes.
* **DetectionModel & DetectionDataModule**: Encapsulate task-specific logic, keeping model code separate from dataset management.
* **Adapter System**: Minimizes the work needed to plug in new models—define input and output adapters (e.g., MyModelInputAdapter, MyModelOutputAdapter), register them with get_input_adapter() and get_output_adapter(), and start training.

---

## 🔧 How to Introduce New Models via Adapters

### Step 1: Create Input and Output Adapters

```python
class YourModelInputAdapter(BaseAdapter):
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        return processed_image, adapted_target
```

```python
class YourModelOutputAdapter:
    def adapt_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "loss": outputs.get("loss"),
            "pred_boxes": outputs.pred_boxes,
            "pred_logits": outputs.logits,
            "loss_dict": outputs.get("loss_dict", {})
        }
```

### Step 2: Register Your Adapters

```python
def get_input_adapter(model_name: str, image_size: int = 800) -> BaseAdapter:
    if "your_model" in model_name.lower():
        return YourModelInputAdapter(model_name=model_name, image_size=image_size)
    ...

def get_output_adapter(model_name: str) -> BaseAdapter:
    if "your_model" in model_name.lower():
        return YourModelOutputAdapter(model_name=model_name)
    ...
```

### Step 3: YAML Configuration

```yaml
model:
  task_type: detection
  model_name: your_model_identifier
  ...
```

---

## 🚦 Getting Started

### Prerequisites

* Databricks workspace with Unity Catalog enabled
* Python 3.8+ or higher
* Access to GPU resources for training performance

### Installation

```bash
git clone https://github.com/Aradhya0510/databricks-cv-architecture.git
cd databricks-cv-architecture
pip install -r requirements.txt
```

### Data Setup

Prepare your COCO dataset:

```
/path/to/dataset/
├── images/
├── annotations.json
```

Upload to Unity Catalog volume:

```
/Volumes/your_catalog/your_schema/your_volume/
├── data/
├── configs/
├── checkpoints/
├── logs/
└── results/
```

**📖 For detailed information on configuring train/validation/test splits, see [Data Configuration Guide](docs/DATA_CONFIGURATION.md)**

### Configuration Example

```yaml
model:
  model_name: "facebook/detr-resnet-50"
  ...
data:
  # New format with separate train/val/test paths (recommended)
  train_data_path: ".../train/images"
  train_annotation_file: ".../train/annotations.json"
  val_data_path: ".../val/images"
  val_annotation_file: ".../val/annotations.json"
  test_data_path: ".../test/images"
  test_annotation_file: ".../test/annotations.json"
training:
  max_epochs: 300
  early_stopping_patience: 20
  monitor_metric: "val_map"
  monitor_mode: "max"
  checkpoint_dir: "..."
  save_top_k: 3
  distributed: true
  use_gpu: true
```

### Training Examples

**Object Detection:**
```python
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule
from src.training.trainer import UnifiedTrainer
...
trainer.train()
```

**Semantic Segmentation:**
```python
from src.tasks.semantic_segmentation.model import SemanticSegmentationModel
from src.tasks.semantic_segmentation.data import SemanticSegmentationDataModule
from src.training.trainer import UnifiedTrainer
...
trainer.train()
```

**Instance Segmentation:**
```python
from src.tasks.instance_segmentation.model import InstanceSegmentationModel
from src.tasks.instance_segmentation.data import InstanceSegmentationDataModule
from src.training.trainer import UnifiedTrainer
...
trainer.train()
```

**Panoptic Segmentation:**
```python
from src.tasks.panoptic_segmentation.model import PanopticSegmentationModel
from src.tasks.panoptic_segmentation.data import PanopticSegmentationDataModule
from src.training.trainer import UnifiedTrainer
...
trainer.train()
```

---

## 📊 Current Features

### ✅ Implemented

* **Object Detection**: End-to-end support for DETR, YOLO, and other detection models
* **Image Classification**: Support for ViT, ConvNeXT, Swin Transformer, and ResNet models
* **Semantic Segmentation**: Dedicated module with SegFormer, DeepLabV3, and other semantic segmentation models
* **Instance Segmentation**: Specialized module for Mask2Former and other instance segmentation models
* **Panoptic Segmentation**: Complete module for unified scene understanding with Mask2Former and similar models
* **Full COCO dataset compatibility** across all tasks
* **Distributed training** using Ray on Databricks
* **MLflow-based experiment tracking** and logging
* **Lightning-based checkpointing** and early stopping
* **Adapter interface** for integrating new models
* **Config-driven modularity** with YAML
* **Comprehensive evaluation metrics**: mAP for detection, IoU/Dice for segmentation, accuracy/F1 for classification

### 🚧 In Progress

* Additional model integrations (Faster R-CNN, U-Net variants)
* Advanced augmentation strategies and pipelines
* Multi-task learning support
* Model compression and quantization

---

## 🌱 Future Directions

* Multi-task learning support
* Plugin registry for dynamic model/adapter loading
* Model compression and post-training quantization
* Real-time inference pipeline integration
* Custom loss function and optimizer configuration support
* Advanced visualization and analysis tools

---

## 📚 Documentation & Contributions

We welcome contributions from the community! To contribute:

* Follow the adapter design pattern for adding new models
* Keep core modules model-agnostic and task-aligned
* Add relevant tests and examples
* Update documentation for all new features
* Use type annotations and maintain readable formatting

For detailed information about each task module, see:
* [Tasks Documentation](src/tasks/README.md) - Comprehensive guide to all task modules
* [Improved Adapters Documentation](docs/IMPROVED_ADAPTERS.md) - Detailed adapter system documentation

---

### 📬 Questions & Feedback

Open an issue or start a discussion thread in the GitHub repo for support, ideas, or feedback.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

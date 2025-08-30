# Databricks Computer Vision Accelerator

An advanced, modular, and extensible computer vision accelerator designed to significantly reduce the complexity involved in building, training, and deploying computer vision models on Databricks. This project aims to serve as a reference implementation for best practices, abstracting away many of the difficult decisions related to model architecture, dataset format, training orchestration, distributed computing, and experiment tracking. By using this accelerator, practitioners can more easily transition from problem formulation to model deployment using well-defined and reusable components that promote clarity, scalability, and productivity.

---

## 🎯 Vision and Approach

Computer vision projects are often burdened by the overwhelming number of decisions that practitioners must make before getting to meaningful experimentation. From choosing the right model architecture and preprocessing pipeline to configuring training infrastructure and tracking results, these decisions can stall progress and increase costs. This architecture solves that problem through:

* **Complete Model Abstraction**: The framework provides **zero-knowledge model management** through Hugging Face's Auto-classes (`AutoModel`, `AutoConfig`, `AutoImageProcessor`). Users can work with any Hugging Face model without understanding its internals - just specify the model name in configuration.
* **Complete Training Loop Abstraction**: Built on PyTorch Lightning, the framework **eliminates the need for training code**. No training loops, optimizers, or schedulers to write - Lightning handles everything automatically.
* **Seamless MLflow Integration**: **Zero-configuration experiment tracking** through Lightning's native MLflow integration. All metrics, parameters, and artifacts are logged automatically.
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

* **Lightning**: **Completely abstracts the training loop** - no training code needed. Provides automatic distributed training, checkpointing, early stopping, and MLflow integration.
* **Hugging Face Transformers**: **Completely abstracts model management** through Auto-classes. Provides high-quality pre-trained vision models such as DETR, YOLO, ViT, SegFormer, and Mask2Former with zero setup - just specify the model name.
* **Ray**: Powers distributed training and large-scale hyperparameter tuning. Enables multi-GPU/multi-node workloads on Databricks.
* **MLflow**: **Seamlessly integrated through Lightning** for automatic experiment tracking, model registry, and serving. Tracks parameters, metrics, models, and artifacts without any manual setup.
* **Albumentations**: Standardizes and simplifies image augmentations to improve model robustness across datasets.
* **PyCOCOTools**: Provides reliable annotation parsing, evaluation, and visualization tools using the COCO format.

---

## 🏗️ Architectural Abstraction Layers

Our framework provides **complete abstraction at every level** of the computer vision pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   YAML Config   │  │   Python API    │  │   Notebooks     │ │
│  │   (Zero Code)   │  │   (Minimal)     │  │   (Interactive) │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                 FRAMEWORK ABSTRACTION LAYERS                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Model Layer    │  │ Training Layer  │  │  Data Layer     │ │
│  │  (AutoModel)    │  │  (Lightning)    │  │  (Adapters)     │ │
│  │  Zero Knowledge │  │  Zero Code      │  │  Zero Format    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Hugging Face  │  │ PyTorch Lightning│  │     MLflow      │ │
│  │   Transformers  │  │                 │  │                 │ │
│  │   (Auto-Classes)│  │ (Training Loop) │  │ (Observability) │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Abstraction Benefits:

**🔧 Model Management (Zero Knowledge)**
```
Traditional: model = MyCustomModel(backbone='resnet50', heads='detection', ...)
Framework:  model = AutoModel.from_pretrained("facebook/detr-resnet-50")
```

**⚡ Training Loop (Zero Code)**
```
Traditional: for epoch in epochs: for batch in dataloader: loss = model(batch)...
Framework:  trainer.fit(model, datamodule)  # Lightning handles everything
```

**📊 Observability (Zero Configuration)**
```
Traditional: mlflow.log_metric("loss", loss); mlflow.log_param("lr", lr)...
Framework:  # Lightning automatically logs everything to MLflow
```

---

## 🧩 Modularity and Extensibility

* **UnifiedTrainer**: Handles all training logic, seamlessly switching between local and distributed modes.
* **DetectionModel & DetectionDataModule**: Encapsulate task-specific logic, keeping model code separate from dataset management.
* **Adapter System**: **Complete abstraction of model-specific data processing** - input and output adapters handle all format conversions automatically. Minimizes the work needed to plug in new models—define input and output adapters (e.g., MyModelInputAdapter, MyModelOutputAdapter), register them with get_input_adapter() and get_output_adapter(), and start training.

---

## 🔧 How to Introduce New Models via Adapters

### Step 1: Create Input and Output Adapters

```python
class YourModelInputAdapter(BaseAdapter):
    def __init__(self, model_name: str, image_size: int = 800):
        # Use AutoImageProcessor for automatic preprocessing
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            size={"height": image_size, "width": image_size},
            do_resize=True, do_rescale=True, do_normalize=True
        )
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # Automatic preprocessing through AutoImageProcessor
        processed = self.processor(image, return_tensors="pt")
        return processed.pixel_values.squeeze(0), adapted_target
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
  model_name: your_model_identifier  # Just change this line
  ...
```

**That's it!** The framework handles everything else automatically through Hugging Face Auto-classes and PyTorch Lightning.

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:
- Access to a Databricks workspace with Unity Catalog enabled
- A COCO format dataset uploaded to your Unity Catalog volume
- Appropriate compute resources (GPU recommended for training)

### Step 1: Repository Setup

Clone this repository in your Databricks workspace:

```bash
git clone https://github.com/Aradhya0510/databricks-cv-architecture.git
cd databricks-cv-architecture
```

### Step 2: Unity Catalog Volume Structure

Organize your Unity Catalog volume with the following structure:

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

### Step 3: Configuration Management

1. **Create a configs folder** in your Unity Catalog volume for your project/experiment
2. **Select the appropriate config file** from the `configs/` directory in the cloned repository based on your task (detection, classification, segmentation) and model choice
3. **Customize the configuration** by updating:
   - Data paths to point to your Unity Catalog volume
   - Model parameters (batch size, learning rate, epochs)
   - Training settings (checkpoint directory, monitoring metrics)
   - Compute-specific parameters (GPU settings, distributed training options)

Example configuration:
```yaml
model:
  model_name: "facebook/detr-resnet-50"  # Just change this to switch models
  task_type: "detection"
  num_classes: 80

data:
  train_data_path: "/Volumes/your_catalog/your_schema/your_volume/data/train/images"
  train_annotation_file: "/Volumes/your_catalog/your_schema/your_volume/data/train/annotations.json"
  val_data_path: "/Volumes/your_catalog/your_schema/your_volume/data/val/images"
  val_annotation_file: "/Volumes/your_catalog/your_schema/your_volume/data/val/annotations.json"
  batch_size: 16
  num_workers: 4

training:
  max_epochs: 300
  learning_rate: 1e-4
  checkpoint_dir: "/Volumes/your_catalog/your_schema/your_volume/checkpoints"
```

### Step 4: Notebook Workflow

Follow the provided reference notebooks in sequence. For object detection, start by copying and customizing `detection_detr_config.yaml` to your Unity Catalog configs folder.

**Recommended Notebook Sequence:**
1. **`00_setup_and_config.py`** - Environment validation and configuration setup
2. **`01_data_preparation.py`** - Dataset analysis and preprocessing validation
3. **`02_model_training.py`** - Model training with MLflow tracking
4. **`03_hparam_tuning.py`** - Hyperparameter optimization (optional)
5. **`04_model_evaluation.py`** - Comprehensive model evaluation
6. **`05_model_registration_deployment.py`** - Model deployment and serving setup

**Important:** Ensure your configuration parameters align with your available compute resources, particularly GPU memory, batch size, and training duration.

### Step 5: MLflow Integration

**Zero-Configuration MLflow Integration:** The framework uses Lightning's native MLflow integration for automatic experiment tracking, model registry, and serving.

**Key Features:**
- **Automatic checkpoint logging** with `log_model="all"`
- **Automatic parameter logging** from LightningModule's `save_hyperparameters()`
- **Automatic metric logging** through Lightning's `self.log()` calls
- **Automatic model registry** in Unity Catalog
- **Volume checkpointing** for persistent storage

**Usage Example:**
```python
from utils.logging import create_databricks_logger
from training.trainer import UnifiedTrainer

# Create logger with automatic integration
mlf_logger = create_databricks_logger(
    experiment_name=experiment_name,
    run_name=run_name,
    log_model="all"  # Automatically log all checkpoints
)

# Initialize trainer with logger
unified_trainer = UnifiedTrainer(
    config=config,
    model=model,
    data_module=data_module,
    logger=mlf_logger
)

# Start training - all logging handled automatically
result = unified_trainer.train()
```

**Everything is automatic:**
- ✅ Parameter logging
- ✅ Metric logging  
- ✅ Checkpoint logging
- ✅ Model registry
- ✅ Model serving

**For more details, see:** [`SIMPLIFIED_MLFLOW_INTEGRATION.md`](SIMPLIFIED_MLFLOW_INTEGRATION.md)

---

## 🎯 Supported Tasks and Models

The framework supports multiple computer vision tasks with **zero-knowledge model management**. Simply specify the model name in your configuration, and the framework automatically handles all the complexity.

**Supported Tasks:** Object Detection, Image Classification, Semantic Segmentation, Instance Segmentation, Universal Segmentation

**Popular Models:** DETR, YOLOS, ViT, ResNet, ConvNeXT, SegFormer, Mask2Former

**How to Use:** Just change the `model_name` in your configuration - the framework handles everything else automatically.

**📖 For complete details:** [Supported Tasks and Models](docs/SUPPORTED_TASKS_AND_MODELS.md)

---

## 🎯 Zero-Knowledge Development Philosophy

The framework is designed for **zero-knowledge development** - you don't need deep ML expertise to use it effectively:

### What You DON'T Need to Know:
- ❌ Model architecture internals (DETR, YOLOS, ResNet, etc.)
- ❌ Training loop implementation
- ❌ Optimizer and scheduler configuration
- ❌ Distributed training setup
- ❌ MLflow logging code
- ❌ Data preprocessing formats

### What You DO Need to Know:
- ✅ How to write a YAML configuration file
- ✅ Your dataset structure (COCO format)
- ✅ Basic ML concepts (learning rate, batch size, epochs)

### Example: Switching from DETR to YOLOS
```yaml
# Just change this one line in your config:
model:
  model_name: "facebook/detr-resnet-50"  # Change to:
  model_name: "hustvl/yolos-tiny"       # That's it!

# No other changes needed - adapters handle everything automatically
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

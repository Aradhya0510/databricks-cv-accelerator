# Computer Vision Training Pipeline - Databricks Lakehouse App

A comprehensive, no-code/low-code interface for training, evaluating, and deploying computer vision models on Databricks.

## 🎯 Overview

This Lakehouse App provides an intuitive web interface for the entire computer vision model lifecycle:

- **Configuration Management**: Create and manage training configurations without writing YAML
- **Data Exploration**: Visualize and validate your datasets
- **Model Training**: Launch and monitor training jobs with real-time metrics
- **Evaluation**: Comprehensive model performance analysis
- **Model Registry**: Register models to Unity Catalog with versioning
- **Deployment**: Deploy models to Databricks Model Serving endpoints
- **Inference**: Interactive model testing and batch inference
- **History Management**: Track all pipeline activities and analytics

## 🚀 Quick Start

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to GPU clusters (for training)
- Unity Catalog volumes set up for data storage

### Installation

1. **Clone or copy the `lakehouse_app` directory** to your Databricks workspace

2. **Install dependencies** (if running locally first):
   ```bash
   pip install -r requirements.txt
   ```

3. **Deploy to Databricks**:
   - Upload the `lakehouse_app` directory to your workspace
   - Create a Databricks Lakehouse App using the `app.yaml` configuration
   - Or run locally with: `streamlit run app.py`

### First Run

1. Navigate to the app URL provided by Databricks
2. Go to **⚙️ Config Setup** page
3. Create your first configuration:
   - Select a CV task (Detection, Classification, etc.)
   - Choose a model (DETR, ResNet, ViT, etc.)
   - Configure data paths in Unity Catalog Volumes
   - Set training parameters
4. Save and activate the configuration
5. Proceed to **📊 Data EDA** to explore your data
6. Launch training from **🚀 Training** page

## 📋 Features

### 1. ⚙️ Configuration Setup

**Create configurations without writing YAML:**
- Select from 5 CV tasks (Detection, Classification, Semantic/Instance/Universal Segmentation)
- Choose from 15+ pre-trained models (DETR, YOLOS, ResNet, ViT, SegFormer, Mask2Former)
- Intuitive forms for all parameters
- Real-time validation
- Load and edit existing configurations

**Key Features:**
- Smart defaults based on model selection
- Template configurations
- YAML preview and download
- Configuration versioning

### 2. 📊 Data EDA

**Explore your dataset before training:**
- Dataset statistics (image counts, size distribution)
- Class distribution visualization
- Sample image viewer
- Data quality validation
- Annotation validation (for detection/segmentation)

**Supported Data Formats:**
- COCO JSON (for detection/segmentation)
- ImageFolder (for classification)
- Delta Tables
- Unity Catalog Volumes

### 3. 🚀 Training

**Launch and monitor training jobs:**
- One-click job submission
- Cluster configuration (GPU selection)
- Real-time training monitoring
- Live metrics visualization (loss, accuracy, mAP)
- Training progress tracking
- Job control (cancel, pause, resume)

**Distributed Training:**
- Single-node multi-GPU (DDP)
- Multi-node with Ray (optional)
- Automatic GPU detection

### 4. 📈 Evaluation

**Comprehensive model analysis:**
- Task-specific metrics (Accuracy, mAP, mIoU)
- Metric history visualization
- Model comparison
- Error analysis
- Confusion matrices
- Per-class performance

**Export Capabilities:**
- PDF/HTML/Markdown reports
- Metric exports (JSON, CSV)
- Visualization downloads

### 5. 📦 Model Registration

**Register models to Unity Catalog:**
- Model versioning
- Model cards with metadata
- Stage management (None, Staging, Production)
- Tags and descriptions
- Lineage tracking

**Features:**
- Browse registered models
- Version comparison
- Model promotion workflows
- Automatic PyFunc wrapping

### 6. 🌐 Deployment

**Deploy to production:**
- Databricks Model Serving endpoints
- Workload size selection (Small/Medium/Large)
- Scale-to-zero configuration
- Endpoint monitoring
- Batch inference jobs

**Deployment Options:**
- Real-time serving endpoints
- Batch inference on Delta tables
- Multiple endpoint management

### 7. 🎮 Inference Playground

**Interactive model testing:**
- Single image upload and inference
- Batch image processing
- URL-based inference
- Real-time visualization
- Confidence threshold adjustment
- Result downloads (images + JSON)

**Visualization:**
- Detection: Bounding boxes with labels
- Classification: Top-K predictions with confidence
- Segmentation: Colored mask overlays

### 8. 📜 History & Management

**Track all activities:**
- Complete activity timeline
- Pipeline analytics
- Resource usage tracking
- User preferences
- Data export/import

**Analytics:**
- Training run statistics
- Model performance trends
- Cost tracking
- Activity distribution

## 🏗️ Architecture

### Directory Structure

```
lakehouse_app/
├── app.yaml                    # App configuration
├── requirements.txt            # Dependencies
├── app.py                      # Main entry point
├── .streamlit/
│   └── config.toml            # Streamlit config
├── pages/                     # Multi-page app
│   ├── 1_⚙️_Config_Setup.py
│   ├── 2_📊_Data_EDA.py
│   ├── 3_🚀_Training.py
│   ├── 4_📈_Evaluation.py
│   ├── 5_📦_Model_Registration.py
│   ├── 6_🌐_Deployment.py
│   ├── 7_🎮_Inference.py
│   └── 8_📜_History.py
├── utils/                     # Core utilities
│   ├── config_generator.py   # YAML generation
│   ├── databricks_client.py  # Jobs/MLflow API
│   └── state_manager.py      # Session state
└── components/                # UI components
    ├── config_forms.py        # Form builders
    ├── visualizations.py      # Charts
    ├── image_viewer.py        # Image display
    └── metrics_display.py     # Metrics UI
```

### Technology Stack

- **Frontend**: Streamlit 1.31.0
- **Visualization**: Plotly, Altair, Matplotlib
- **Backend Integration**: Databricks SDK, MLflow
- **Data Processing**: PySpark, PyArrow, Pandas
- **ML Frameworks**: PyTorch, Transformers, Lightning
- **Image Processing**: Pillow, OpenCV

## 🔧 Configuration

### Unity Catalog Setup

1. **Create a catalog and schema:**
   ```sql
   CREATE CATALOG IF NOT EXISTS main;
   CREATE SCHEMA IF NOT EXISTS main.cv_models;
   ```

2. **Create a volume for data storage:**
   ```sql
   CREATE VOLUME IF NOT EXISTS main.cv_models.cv_data;
   ```

3. **Organize your data:**
   ```
   /Volumes/main/cv_models/cv_data/
   ├── data/
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── configs/
   ├── checkpoints/
   └── results/
   ```

### Environment Variables

Set in `app.yaml` or `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 8080

[theme]
primaryColor = "#FF3621"
backgroundColor = "#FFFFFF"
```

## 📊 Supported Tasks & Models

### Object Detection
- **Models**: DETR (ResNet-50/101), YOLOS (Tiny/Small)
- **Metrics**: mAP, mAP@50, mAP@75, Per-class AP
- **Data Format**: COCO JSON

### Image Classification
- **Models**: ResNet-50, ViT-Base, ConvNeXT, Swin Transformer
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Data Format**: ImageFolder structure

### Semantic Segmentation
- **Models**: SegFormer (B0/B1), MiT-B0, Mask2Former
- **Metrics**: mIoU, Pixel Accuracy, Per-class IoU
- **Data Format**: Masks + Images

### Instance Segmentation
- **Models**: Mask2Former (Swin-Base/Small)
- **Metrics**: AP, AP@50, AP@75, Per-class AP
- **Data Format**: COCO JSON

### Universal Segmentation
- **Models**: Mask2Former Panoptic
- **Metrics**: PQ, mIoU, Combined metrics
- **Data Format**: COCO Panoptic JSON

## 🔌 Integration with Existing Codebase

The app seamlessly integrates with your existing CV framework:

1. **Uses existing jobs**: Calls `jobs/model_training.py` and `jobs/model_registration_deployment.py`
2. **Leverages configs**: Generates YAML compatible with your `configs/` directory
3. **MLflow integration**: Connects to your MLflow experiments
4. **Unity Catalog**: Reads/writes to your volumes and tables

## 💡 Best Practices

### Configuration
- Use descriptive names for configs
- Store configs in Unity Catalog Volumes
- Version your configurations
- Test with small datasets first

### Training
- Start with small models for testing
- Monitor resource usage
- Use early stopping
- Save checkpoints to volumes

### Data
- Validate data before training
- Check class distribution
- Use augmentation for small datasets
- Store data in Unity Catalog Volumes

### Deployment
- Test endpoints before production
- Enable scale-to-zero for cost savings
- Monitor endpoint metrics
- Use batch inference for large datasets

## 🐛 Troubleshooting

### Common Issues

**1. "No active configuration found"**
- Solution: Create or load a configuration in the Config Setup page

**2. "Job submission failed"**
- Check source path is correct
- Verify cluster configuration
- Ensure GPU quota is available

**3. "Cannot access data path"**
- Verify Unity Catalog permissions
- Check volume paths are correct
- Ensure data exists at specified paths

**4. "MLflow experiment not found"**
- Use absolute workspace paths (e.g., `/Users/email@company.com/experiments`)
- Create experiment manually if needed

### Debug Mode

Enable debug logging in Streamlit:
```bash
streamlit run app.py --logger.level=debug
```

## 🤝 Contributing

To extend the app:

1. **Add new pages**: Create new files in `pages/` directory
2. **Add utilities**: Extend classes in `utils/` directory
3. **Add components**: Create reusable UI in `components/` directory
4. **Update config generator**: Add new models/tasks in `config_generator.py`

## 📝 License

This Lakehouse App is part of the Databricks Computer Vision Reference Implementation.

## 🙏 Acknowledgments

Built on top of:
- Databricks Lakehouse Platform
- Streamlit framework
- PyTorch Lightning
- HuggingFace Transformers
- MLflow

## 📚 Additional Resources

- [Databricks Lakehouse Apps Documentation](https://docs.databricks.com/en/apps/index.html)
- [Unity Catalog Guide](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
- [Model Serving Documentation](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
- [MLflow on Databricks](https://docs.databricks.com/en/mlflow/index.html)

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review existing configurations
3. Consult the main CV framework documentation
4. Contact your Databricks representative

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Maintained by**: Databricks CV Team


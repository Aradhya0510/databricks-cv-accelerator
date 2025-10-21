# Standard GPU Notebooks

Use these notebooks if you're running on **standard Databricks GPU compute** (single or multi-GPU clusters).

## 🚀 Quick Start

Run notebooks in order:

1. **`00_setup_and_config.py`** - Environment setup and configuration
2. **`01_data_preparation.py`** - Prepare and validate COCO dataset
3. **`02_model_training.py`** - Train computer vision models (full version)
   - OR `02_model_training_simple.py` - Train models (simplified version)
4. **`03_hparam_tuning.py`** - Hyperparameter tuning with Ray Tune
5. **`04_model_evaluation.py`** - Evaluate trained models
6. **`05_model_registration_deployment.py`** - Register and deploy to Model Registry
7. **`06_model_monitoring.py`** - Monitor model performance

## 🖥️ Compute Requirements

- **Cluster Type**: Standard Databricks GPU cluster
- **GPU**: NVIDIA T4, A10, A100, or V100
- **Single GPU**: Works with 1 GPU
- **Multi-GPU**: Automatically uses DDP for distributed training
- **Runtime**: Databricks Runtime ML 14.0+ with GPU

## 🔄 Distributed Training

Standard GPU notebooks use:
- **Single GPU**: Direct training on one GPU
- **Multi-GPU**: PyTorch Lightning DDP (DistributedDataParallel)
- **Configuration**: Set `distributed: true` in config YAML

## 🌟 Using Serverless GPU?

If you have access to **Databricks Serverless GPU compute**, use:
→ **`../notebooks_serverless/`** instead

Serverless GPU provides:
- Auto-scaling GPU resources
- No cluster management
- Pay-per-use pricing
- Faster startup times

## 📝 Configuration

All notebooks use the unified config system:
- **Configs**: `../configs/*.yaml`
- **Loader**: `from databricks_cv_accelerator.config import load_config`

Example:
```python
config = load_config("/Workspace/.../configs/detection_detr_config.yaml")
```

## 🎯 Supported Tasks

- **Object Detection**: DETR, YOLOS
- **Image Classification**: ResNet, ViT
- **Semantic Segmentation**: SegFormer, MiT-B0
- **Instance Segmentation**: Mask2Former
- **Universal Segmentation**: Mask2Former Panoptic

## 📚 Documentation

- **Config Guide**: `../configs/README.md`
- **Package Structure**: `../PACKAGE_STRUCTURE.md`
- **Main README**: `../README.md`

## 💡 Tips

1. **Start Simple**: Use `02_model_training_simple.py` for quick experiments
2. **Monitor Resources**: Check GPU utilization in Databricks UI
3. **Checkpointing**: Models auto-save to configured checkpoint directory
4. **MLflow**: All experiments auto-logged to MLflow

## ⚠️ Troubleshooting

- **Out of Memory**: Reduce `batch_size` in config
- **Multi-GPU Issues**: Ensure cluster has multiple GPUs attached
- **Import Errors**: Run `00_setup_and_config.py` first


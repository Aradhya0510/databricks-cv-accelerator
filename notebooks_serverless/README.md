# Serverless GPU Notebooks

Use these notebooks if you're running on **Databricks Serverless GPU compute**.

## 🚀 Quick Start

Run notebooks in order:

1. **`00_setup_and_config_serverless.py`** - Serverless environment setup and configuration
2. **`01_data_preparation.py`** - Prepare and validate COCO dataset
3. **`02_model_training_serverless.py`** - Train models on Serverless GPU (full version)
   - OR `02_model_training_simple_serverless.py` - Train models (simplified version)
4. **`03_hparam_tuning.py`** - Hyperparameter tuning with Ray Tune
5. **`04_model_evaluation_serverless.py`** - Evaluate trained models
6. **`05_model_registration_deployment.py`** - Register and deploy to Model Registry
7. **`06_model_monitoring.py`** - Monitor model performance

## ☁️ Serverless GPU Requirements

- **Compute Type**: Databricks Serverless GPU (no cluster needed!)
- **GPU Options**: 
  - **A10**: Up to 8 GPUs per job (recommended for most workloads)
  - **H100**: Single GPU only (high performance)
- **Runtime**: Serverless GPU environment (auto-configured)
- **Dependencies**: Automatically installed in serverless environment

## 🎯 Key Differences from Standard GPU

### Execution Model
Serverless GPU uses the `@distributed` decorator:

```python
from serverless_gpu import distributed

@distributed(gpus=4, gpu_type='A10', remote=True)
def distributed_train(config_dict, DetectionModel, ...):
    # Self-contained training function
    # All dependencies passed as arguments
    # Runs on remote serverless compute
    trainer = UnifiedTrainer(...)
    return trainer.train()

# Execute on serverless GPU
result = distributed_train.distributed(config, DetectionModel, ...)
```

### Why the Function Wrapper?
- **Serialization**: Code is packaged and sent to remote serverless compute
- **Self-Contained**: Function must have all dependencies as arguments
- **Scalable**: Automatically distributed across specified GPUs
- **Isolated**: Clean environment per execution

## 🔧 Configuration

Serverless GPU settings are in the **`serverless:`** section of config YAML:

```yaml
serverless:
  enabled: true
  gpu_type: A10      # Options: A10, H100
  gpu_count: 4       # Number of GPUs (H100 only supports 1)
```

All notebooks use the unified config system:
- **Configs**: `../configs/*.yaml`
- **Loader**: `from databricks_cv_accelerator.config import load_config`

## 🌟 Using Standard GPU Clusters?

If you're using **standard Databricks GPU clusters**, use:
→ **`../notebooks/`** instead

Standard GPU is better for:
- Interactive development and debugging
- Fine-grained resource control
- Custom cluster configurations
- Long-running notebooks

## 📊 When to Use Serverless GPU

✅ **Best For:**
- Production training jobs
- Automated workflows
- Large-scale experiments
- Cost optimization (pay-per-use)
- No cluster management overhead

❌ **Not Ideal For:**
- Interactive debugging (harder to inspect mid-execution)
- Very small/quick experiments (startup overhead)
- Custom environment requirements

## 🎯 Supported Tasks

- **Object Detection**: DETR, YOLOS
- **Image Classification**: ResNet, ViT
- **Semantic Segmentation**: SegFormer, MiT-B0
- **Instance Segmentation**: Mask2Former
- **Universal Segmentation**: Mask2Former Panoptic

## 💡 Tips

1. **GPU Selection**:
   - A10: Great for most models, supports up to 8 GPUs
   - H100: High performance, single GPU only

2. **Debugging**: Use standard GPU notebooks first, then switch to serverless for production

3. **Checkpointing**: Save to Volumes for persistence across runs

4. **Monitoring**: Check MLflow for training metrics

5. **Cost Optimization**: Use appropriate GPU count (more isn't always better)

## ⚠️ Troubleshooting

### Serialization Errors
```
TypeError: cannot pickle 'module' object
```
**Solution**: Ensure all dependencies are passed as function arguments, not captured from outer scope

### Import Errors in Distributed Function
```
ModuleNotFoundError: No module named 'databricks_cv_accelerator'
```
**Solution**: 
1. Add workspace path to serverless environment dependencies
2. Or pass classes/functions as arguments to `@distributed` function

### GPU Type Errors
```
H100 only supports single-node workflows
```
**Solution**: Set `gpu_count: 1` for H100 GPUs

### Out of Memory
**Solution**: Reduce `batch_size` in config or use fewer GPUs

## 📚 Documentation

- **Serverless GPU Guide**: `DEPLOYMENT_GUIDE.md`
- **Config Reference**: `../configs/README.md`
- **Package Structure**: `../PACKAGE_STRUCTURE.md`
- **Main README**: `../README.md`

## 🔗 Resources

- [Databricks Serverless GPU Documentation](https://docs.databricks.com/serverless-compute/index.html)
- [Distributed Training Best Practices](https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html)


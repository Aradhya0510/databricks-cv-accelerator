# MLflow Troubleshooting Guide

This guide helps resolve common MLflow integration issues, particularly when working with Databricks managed MLflow.

## Common Issues and Solutions

### 1. "Resource Not Found" Error in MLflow UI

**Problem:** MLflow UI shows "Resource not found" or experiments don't appear.

**Root Cause:** Missing `mlflow.pytorch.autolog()` call.

**Solution:**
```python
import mlflow
mlflow.pytorch.autolog()  # Add this before training
```

**Why this is needed:**
- MLflow autolog automatically creates runs and logs metrics
- Without autolog, PyTorch Lightning's MLFlowLogger may not properly integrate with Databricks managed MLflow
- Autolog ensures proper experiment and run creation

### 2. Experiments Not Appearing in MLflow UI

**Problem:** Training runs complete but don't show up in MLflow UI.

**Solutions:**

1. **Check experiment name format:**
```python
# Use proper Databricks experiment naming
experiment_name = f"/Users/{username}/your_experiment_name"
mlflow.set_experiment(experiment_name)
```

2. **Ensure MLflow run is active:**
```python
with mlflow.start_run(run_name="your_run_name"):
    # Your training code here
    trainer.fit(model, datamodule=data_module)
```

3. **Verify tracking URI:**
```python
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow registry URI: {mlflow.get_registry_uri()}")
```

### 3. Metrics Not Logging

**Problem:** Training metrics don't appear in MLflow.

**Solutions:**

1. **Use PyTorch Lightning's logging:**
```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log('train_loss', loss, on_step=True, on_epoch=True)
    return loss
```

2. **Enable autolog before training:**
```python
mlflow.pytorch.autolog()
```

3. **Check logger configuration:**
```python
from lightning.pytorch.loggers import MLFlowLogger

mlflow_logger = MLFlowLogger(
    experiment_name=experiment_name,
    run_name=run_name,
    log_model=True
)

trainer = pl.Trainer(
    logger=mlflow_logger,
    # ... other parameters
)
```

### 4. Model Artifacts Not Saving

**Problem:** Model files don't appear in MLflow artifacts.

**Solutions:**

1. **Enable model logging in MLFlowLogger:**
```python
mlflow_logger = MLFlowLogger(
    experiment_name=experiment_name,
    log_model=True  # This is crucial
)
```

2. **Use ModelCheckpoint callback:**
```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="model_best",
    monitor="val_loss",
    mode="min",
    save_top_k=1
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback],
    # ... other parameters
)
```

### 5. Databricks-Specific Issues

**Problem:** MLflow doesn't work properly in Databricks environment.

**Solutions:**

1. **Use proper experiment naming:**
```python
# Get current user
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{username}/your_experiment"
```

2. **Check Databricks MLflow settings:**
```python
# Verify MLflow is properly configured
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Registry URI: {mlflow.get_registry_uri()}")
```

3. **Use Databricks-specific logger:**
```python
from src.utils.logging import create_databricks_logger

logger = create_databricks_logger(
    experiment_name=experiment_name,
    run_name=run_name
)
```

### 6. Distributed Training Issues

**Problem:** MLflow logging doesn't work with distributed training.

**Solutions:**

1. **Use Ray MLflow integration:**
```python
from ray.air.integrations.mlflow import MLflowLoggerCallback

mlflow_logger = MLflowLoggerCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    registry_uri=mlflow.get_registry_uri(),
    experiment_name=experiment_name
)
```

2. **Ensure proper run management:**
```python
# In distributed training, ensure only one process logs
if trainer.is_global_zero:
    mlflow.log_metrics(metrics)
```

## Complete Working Example

```python
import mlflow
import lightning as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# 1. Enable autolog
mlflow.pytorch.autolog()

# 2. Set up experiment
experiment_name = "/Users/your-username/your-experiment"
mlflow.set_experiment(experiment_name)

# 3. Create logger
mlflow_logger = MLFlowLogger(
    experiment_name=experiment_name,
    run_name="your_run_name",
    log_model=True
)

# 4. Set up callbacks
callbacks = [
    ModelCheckpoint(
        dirpath="./checkpoints",
        filename="model_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
]

# 5. Create trainer
trainer = pl.Trainer(
    max_epochs=10,
    logger=mlflow_logger,
    callbacks=callbacks
)

# 6. Start training with MLflow run
with mlflow.start_run(run_name="your_run_name"):
    mlflow.log_params({
        "max_epochs": 10,
        "learning_rate": 0.001
    })
    
    trainer.fit(model, datamodule=data_module)
```

## Debugging Checklist

- [ ] `mlflow.pytorch.autolog()` is called before training
- [ ] Experiment name follows Databricks format: `/Users/username/experiment_name`
- [ ] `mlflow.set_experiment()` is called
- [ ] Training is wrapped in `mlflow.start_run()`
- [ ] MLFlowLogger has `log_model=True`
- [ ] ModelCheckpoint callback is configured
- [ ] PyTorch Lightning model uses `self.log()` for metrics
- [ ] Tracking URI points to Databricks managed MLflow

## Getting Help

If you're still experiencing issues:

1. Check the MLflow logs in your Databricks notebook
2. Verify your Databricks workspace has MLflow enabled
3. Ensure you have proper permissions for MLflow experiments
4. Check the [MLflow documentation](https://mlflow.org/docs/latest/ml/tracking/autolog/#autolog-pytorch) for PyTorch autolog
5. Review the [PyTorch Lightning MLflow integration](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html) 
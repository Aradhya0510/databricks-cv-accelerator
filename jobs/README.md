# Databricks Jobs - Production Training Scripts

This directory contains production-ready Python scripts designed to run on **Databricks Jobs Compute** with full multi-GPU DDP support.

## Why Use Jobs Instead of Notebooks?

PyTorch Lightning restricts multi-GPU usage in interactive/notebook environments:

```
INFO:pytorch_lightning.utilities.rank_zero:Trainer will use only 1 of 4 GPUs because 
it is running inside an interactive / notebook environment. Multi-GPU inside interactive 
environments is considered experimental and unstable.
```

**Solution**: Run these Python scripts on **Databricks Jobs compute** with proper cluster configuration to unlock full multi-GPU DDP training!

### ⚠️ CRITICAL: Cluster Configuration Requirements

**Do NOT use Single Node clusters with local Spark mode!**

Single Node clusters (`num_workers: 0` with `spark.master: local[*]`) behave like interactive compute and have critical limitations:
- ❌ GPU scheduling disabled
- ❌ DDP may be blocked
- ❌ Treated as interactive environment

**✅ Use one of these instead:**
- **Serverless compute** (recommended) - fully managed, no configuration
- **Multi-node jobs cluster** - `num_workers: 1` or more (not 0)

See [Cluster Configuration](#-cluster-configuration-for-production-jobs) section below for details.

---

## 📁 Available Scripts

### 1. `model_training.py` - Production Training

Train CV models with full multi-GPU DDP support on Jobs compute.

**Features:**
- ✅ Automatic multi-GPU DDP when multiple GPUs detected
- ✅ Command-line argument parsing with argparse
- ✅ MLflow experiment tracking
- ✅ Automatic checkpointing

### 2. `model_registration_deployment.py` - Model Deployment

Register trained models to Unity Catalog and deploy to Model Serving.

**Features:**
- ✅ Automated Unity Catalog registration
- ✅ Optional endpoint deployment
- ✅ Model versioning and tagging

---

## 🚀 How to Use with Databricks Lakeflow Jobs UI

### Step 1: Create a Training Job

1. **Navigate to Workflows** → **Create Job**

2. **Configure the task:**
   - **Task name**: `train-detr-model`
   - **Type**: **Python script**
   - **Source**: **Workspace**
   - **Path**: `/Workspace/Users/your.email@databricks.com/path-to-repo/jobs/model_training.py`

3. **Add Parameters as JSON Array:**
   
   In the **Parameters** field, paste this **JSON array of strings**:
   
   **Recommended (with explicit strategy for Lakeflow Jobs):**
   ```json
   ["--config_path", "/Volumes/users/aradhya_chouhan/cv_arch_test/configs/detection_detr_config.yaml", "--src_path", "/Workspace/Users/aradhya.chouhan@databricks.com/Computer Vision/databricks-cv-accelerator/src", "--force_strategy", "ddp", "--devices", "auto", "--force_jobs"]
   ```
   
   **Minimal (relies on auto-detection):**
   ```json
   ["--config_path", "/Volumes/users/aradhya_chouhan/cv_arch_test/configs/detection_detr_config.yaml", "--src_path", "/Workspace/Users/aradhya.chouhan@databricks.com/Computer Vision/databricks-cv-accelerator/src"]
   ```
   
   **Important Notes:**
   - ✅ Must be a valid JSON array with straight quotes `"`
   - ✅ Each flag and value is a separate string in the array
   - ✅ Replace paths with your actual paths
   - ✅ `--force_strategy ddp` and `--force_jobs` ensure multi-GPU in Lakeflow Jobs
   - ❌ Do NOT use key-value pairs like `key: value`
   - ❌ Do NOT use shell-style `--key "value"` format

4. **Select Compute:**
   
   ⚠️ **IMPORTANT**: Choose one of these configurations:
   
   **Option A: Serverless (Recommended)**
   - Select **Serverless** or specify an `environment_key`
   - No manual cluster configuration needed
   - Fully managed GPU allocation
   
   **Option B: Jobs Cluster (Multi-Node)**
   - **Node type**: `g5.12xlarge` (4x A10G GPUs)
   - **Workers**: `1` or more (**NOT 0**)
   - **Runtime**: Databricks Runtime ML 14.3 LTS or higher
   - ⚠️ Do NOT use Single Node with `local[*]` Spark mode
   
   See [Cluster Configuration](#-cluster-configuration-for-production-jobs) for detailed setup.

5. **Click "Create"** and **"Run now"**

### Step 2: Create a Deployment Job

1. **Create another job**

2. **Configure the task:**
   - **Type**: **Python script**
   - **Path**: `/Workspace/.../jobs/model_registration_deployment.py`

3. **Add Parameters as JSON Array:**
   
   ```json
   ["--config_path", "/Volumes/users/aradhya_chouhan/cv_arch_test/configs/detection_detr_config.yaml", "--src_path", "/Workspace/Users/aradhya.chouhan@databricks.com/Computer Vision/databricks-cv-accelerator/src", "--checkpoint_path", "/Volumes/users/aradhya_chouhan/cv_arch_test/checkpoints/best-model.ckpt", "--model_name", "detr_coco_detector", "--deploy", "true"]
   ```

4. **Select Compute:** CPU cluster (e.g., `i3.xlarge`)

5. **Run**

---

## 📋 Parameter Reference

### Training Script Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--config_path` | Yes | Path to YAML configuration | `/Volumes/users/name/volume/configs/detection_detr_config.yaml` |
| `--src_path` | Yes | Path to src directory | `/Workspace/Users/your.email@databricks.com/repo-name/src` |
| `--force_strategy` | No | Lightning strategy (`auto`, `ddp`, `ddp_notebook`) | `ddp` |
| `--devices` | No | Number of devices or `auto` | `auto`, `4` |
| `--force_jobs` | No | Force non-interactive Jobs mode (flag) | - |

**Example Parameters JSON (Recommended for Lakeflow Jobs):**
```json
[
  "--config_path", "/Volumes/users/aradhya_chouhan/cv_arch_test/configs/detection_detr_config.yaml",
  "--src_path", "/Workspace/Users/aradhya.chouhan@databricks.com/Computer Vision/databricks-cv-accelerator/src",
  "--force_strategy", "ddp",
  "--devices", "auto",
  "--force_jobs"
]
```

**Parameter Details:**

- **`--force_strategy`**: Explicitly set PyTorch Lightning strategy
  - `ddp` (recommended): Multi-GPU DDP for Jobs with ≥2 GPUs
  - `auto`: Environment-aware auto-detection
  - `ddp_notebook`: For notebooks (experimental, not recommended)

- **`--devices`**: Control GPU allocation
  - `auto`: Auto-detect available GPUs (recommended)
  - `4`: Use exactly 4 GPUs

- **`--force_jobs`**: Override environment detection to treat as non-interactive
  - Use when Lakeflow Jobs incorrectly detects as notebook/interactive
  - Enables CUDA probing and multi-GPU DDP
  - Recommended for Lakeflow Jobs

### Deployment Script Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--config_path` | Yes | - | Path to YAML configuration |
| `--src_path` | Yes | - | Path to src directory |
| `--checkpoint_path` | Yes | - | Path to model checkpoint (.ckpt) |
| `--model_name` | Yes | - | Model name in Unity Catalog |
| `--deploy` | No | false | Deploy to endpoint (true/false) |
| `--endpoint_name` | No | {model_name}-endpoint | Serving endpoint name |
| `--workload_size` | No | Small | Endpoint size (Small/Medium/Large) |
| `--catalog` | No | main | Unity Catalog name |
| `--schema` | No | cv_models | Unity Catalog schema |

**Example Parameters JSON:**
```json
[
  "--config_path", "/Volumes/users/aradhya_chouhan/cv_arch_test/configs/detection_detr_config.yaml",
  "--src_path", "/Workspace/Users/aradhya.chouhan@databricks.com/Computer Vision/databricks-cv-accelerator/src",
  "--checkpoint_path", "/Volumes/users/aradhya_chouhan/cv_arch_test/checkpoints/best-model.ckpt",
  "--model_name", "detr_coco_detector",
  "--deploy", "true",
  "--endpoint_name", "detr-prod"
]
```

---

## 💡 How Parameters Work

Databricks Jobs for Python script tasks expect parameters as a **JSON array of strings**:

```json
["--flag1", "value1", "--flag2", "value2"]
```

These are passed to your script as command-line arguments and parsed with `argparse`:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True)
parser.add_argument("--src_path", required=True)
args = parser.parse_args()
```

### Common Mistakes to Avoid

❌ **Wrong:** Key-value pairs in UI
```
Key: config_path
Value: /Volumes/.../config.yaml
```

❌ **Wrong:** Shell-style string
```
--config_path "/Volumes/.../config.yaml" --src_path "/Workspace/.../src"
```

✅ **Correct:** JSON array of strings
```json
["--config_path", "/Volumes/.../config.yaml", "--src_path", "/Workspace/.../src"]
```

---

## 🎯 Complete Example: Training Job

### Job Configuration

```
Task Name: train-detr-production
Type: Python script
Path: /Workspace/Users/aradhya.chouhan@databricks.com/Computer Vision/databricks-cv-accelerator/jobs/model_training.py

Parameters:
["--config_path", "/Volumes/users/aradhya_chouhan/cv_arch_test/configs/detection_detr_config.yaml", "--src_path", "/Workspace/Users/aradhya.chouhan@databricks.com/Computer Vision/databricks-cv-accelerator/src"]

Cluster:
- Node type: g5.12xlarge (4x A10G GPUs)
- Workers: 0
- Runtime: Databricks ML 14.3 LTS
```

### Expected Output

```
🚀 Databricks CV Model Training - Jobs Compute
================================================================================
📦 Using src path: /Workspace/Users/aradhya.chouhan@databricks.com/Computer Vision/databricks-cv-accelerator/src
📋 Config path: /Volumes/users/aradhya_chouhan/cv_arch_test/configs/detection_detr_config.yaml
📋 Loading configuration from: /Volumes/users/aradhya_chouhan/cv_arch_test/configs/detection_detr_config.yaml

🎯 Multi-GPU training enabled: 4 GPUs detected

🔧 Initializing detection model and data module...
✅ Model and data initialized!

🎯 Starting Training
================================================================================
✅ Using DDP training strategy.
```

---

## 🔧 Troubleshooting

### Issue: "error: unrecognized arguments"

**Cause:** Parameters not formatted as JSON array

**Solution:** Ensure Parameters field contains a valid JSON array:
```json
["--config_path", "/path/to/config.yaml", "--src_path", "/path/to/src"]
```

### Issue: "ModuleNotFoundError: No module named 'config'"

**Cause:** Incorrect `src_path`

**Solution:** Verify `src_path` points to your actual src directory. Look at the job file path to determine correct path:
- If job is at: `/Workspace/Users/your.email@databricks.com/Computer Vision/repo-name/jobs/model_training.py`
- Then src_path is: `/Workspace/Users/your.email@databricks.com/Computer Vision/repo-name/src`

### Issue: "error: argument --config_path is required"

**Cause:** Missing required parameter in JSON array

**Solution:** Ensure both `--config_path` and `--src_path` are in the Parameters JSON array

---

## 📊 Cluster Configuration for Production Jobs

### ⚠️ CRITICAL: Avoid Single Node Local Spark Mode

**Single Node clusters with local Spark mode behave like interactive environments and have limitations:**

- ❌ `spark.master: local[*, 4]` with `num_workers: 0` triggers interactive-style behavior
- ❌ GPU scheduling is disabled in local mode
- ❌ DDP strategy may be blocked by PyTorch Lightning's interactive detection
- ❌ Treated as `ResourceClass: SingleNode` (interactive patterns)

### ✅ Recommended: Serverless Compute (Simplest)

**Best for most use cases** - fully managed, no cluster configuration needed:

```yaml
resources:
  jobs:
    cv_training_job:
      name: CV Model Training
      tasks:
        - task_key: train_model
          spark_python_task:
            python_file: /Workspace/.../jobs/model_training.py
            parameters:
              - --config_path
              - /Volumes/.../config.yaml
              - --src_path
              - /Workspace/.../src
          environment_key: default
      environments:
        - environment_key: default
          spec:
            client: "1"
            dependencies:
              - pytorch
              - transformers
```

**Benefits:**
- ✅ True non-interactive compute
- ✅ No manual cluster configuration
- ✅ Auto-scaling and GPU support
- ✅ Fastest startup times

### ✅ Alternative: Multi-Node Jobs Cluster

**For explicit GPU control** - use at least 1 worker (not Single Node):

```yaml
job_clusters:
  - job_cluster_key: gpu_training_cluster
    new_cluster:
      spark_version: 17.1.x-gpu-ml-scala2.13
      node_type_id: g5.12xlarge        # 4x A10G GPUs
      driver_node_type_id: g5.12xlarge
      num_workers: 1                    # IMPORTANT: >= 1, not 0
      spark_conf: {}                    # NO local[*] or singleNode
      data_security_mode: STANDARD
      runtime_engine: STANDARD
```

**Key Requirements:**
- `num_workers: 1` or more (eliminates Single Node local mode)
- Remove `spark.master: local[*]` from `spark_conf`
- Remove `spark.databricks.cluster.profile: singleNode`
- Remove `ResourceClass: SingleNode` from custom tags

**Instance Recommendations:**
- **Multi-GPU Training:** `g5.12xlarge` (4x A10G, 24GB each)
- **Single-GPU Training:** `g5.xlarge` (1x A10G, 24GB)
- **Cost-Effective:** `g4dn.xlarge` (1x T4, 16GB)

### Deployment (CPU Only)

For model registration/deployment jobs (no GPU needed):

```
Node Type: i3.xlarge (4 cores, 30GB RAM)
Workers: 0 (acceptable for CPU-only tasks)
Runtime: Databricks ML 14.3 LTS or higher
```

---

## 🆚 Notebooks vs Jobs Comparison

| Feature | Notebooks (Interactive) | Jobs (Non-Interactive) |
|---------|------------------------|------------------------|
| **Multi-GPU DDP** | ⚠️ Limited (1 GPU) | ✅ Full Support (All GPUs) |
| **Stability** | Experimental | Production-Grade |
| **Parameters** | Manual in cells | JSON array in UI |
| **Scheduling** | ❌ Not supported | ✅ Cron, triggers |
| **Best For** | Development | Production |

---

## 📊 MLflow Experiment Configuration

### ⚠️ IMPORTANT: Experiment Names Must Be Absolute Workspace Paths

On Databricks, MLflow experiment names **must be absolute workspace paths**, not plain strings.

Add this to your config YAML under `training`:

```yaml
training:
  max_epochs: 100
  
  # REQUIRED: MLflow experiment name (absolute workspace path)
  experiment_name: "/Users/your.email@databricks.com/cv_detection_experiment"
  
  checkpoint_dir: /Volumes/.../checkpoints
  # ... other training params
```

### Valid Experiment Name Formats

**Option 1: User's Home Folder**
```yaml
experiment_name: "/Users/aradhya.chouhan@databricks.com/cv_detection_detr"
```

**Option 2: Shared Workspace**
```yaml
experiment_name: "/Shared/cv_experiments/detection"
```

**Option 3: Explicit Workspace Path**
```yaml
experiment_name: "/Workspace/Users/aradhya.chouhan@databricks.com/experiments/detr"
```

### Common Mistakes

❌ **Wrong: Plain string**
```yaml
experiment_name: "cv/detection/detr"  # Will fail!
```

❌ **Wrong: Relative path**
```yaml
experiment_name: "experiments/detection"  # Will fail!
```

✅ **Correct: Absolute workspace path**
```yaml
experiment_name: "/Users/your.email@databricks.com/cv_detection"
```

---

## 🐛 Troubleshooting

### DDP Strategy Error: "Not compatible with interactive environment"

**Error Message:**
```
MisconfigurationException: `Trainer(strategy='ddp')` is not compatible with an 
interactive environment. Run your code as a script, or choose a notebook-compatible 
strategy: `Trainer(strategy='ddp_notebook')`.
```

**Root Cause:**  
You're using a **Single Node cluster with local Spark mode** (`spark.master: local[*, 4]`, `num_workers: 0`). This configuration:
- Runs Spark locally on the driver (no separate executors)
- Is treated as `ResourceClass: SingleNode` (interactive-style compute)
- Triggers PyTorch Lightning's interactive environment detection
- **Disables GPU scheduling for distributed workloads** - you won't get proper multi-GPU utilization

**Solution:**  

You **must** reconfigure your cluster. Choose one of these production-grade configurations:

**Option 1: Serverless Compute** (easiest & recommended)
```yaml
# Remove the entire new_cluster block
# Add environment_key to your task
tasks:
  - task_key: train_model
    spark_python_task:
      python_file: /Workspace/.../jobs/model_training.py
      parameters: [...]
    environment_key: default
```

**Benefits:**
- ✅ True non-interactive compute
- ✅ Proper GPU scheduling and utilization
- ✅ No manual cluster management

**Option 2: Multi-Node Jobs Cluster**
```yaml
job_clusters:
  - job_cluster_key: gpu_cluster
    new_cluster:
      num_workers: 1              # Change from 0 to >= 1
      spark_conf: {}              # Remove local[*] and singleNode profile
      # Remove ResourceClass: SingleNode from custom_tags
```

**Benefits:**
- ✅ Full control over GPU instance types
- ✅ Proper distributed compute environment
- ✅ Production-grade GPU scheduling

See the [Cluster Configuration](#-cluster-configuration-for-production-jobs) section above for complete setup instructions.

---

### ModuleNotFoundError: No module named 'config'

**Cause:**  
The `src` directory is not in Python's search path, or the `--src_path` parameter was not passed correctly.

**Solution:**
1. Ensure you're passing parameters as a JSON array in the Jobs UI (not a plain string)
2. Verify `--src_path` points to your actual `src` folder location
3. Check the job output for `📦 Using src path:` to confirm the correct path is being used

---

### UserWarnings about "copying from non-meta parameter"

**Error Message:**
```
UserWarning: for conv1.weight: copying from a non-meta parameter in the checkpoint 
to a meta parameter in the current model, which is a no-op.
```

**Cause:**  
Databricks Jobs environment initializes models with meta tensors for memory efficiency, leading to harmless warnings during weight loading.

**Solution:**  
Already suppressed in `model_training.py` via:
```python
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
```

These warnings are cosmetic and don't affect training. No action needed.

---

## 🎓 Next Steps

1. **Prepare your config YAML** with valid `experiment_name` in `training` section
2. **Upload config** to Unity Catalog Volumes
3. **Note your repository path** in Databricks Workspace
4. **Create a training job** with proper JSON parameters
5. **Run and monitor** in MLflow (experiments will appear at your specified path)
6. **Register best model** using deployment script

For more information, see the main [README.md](../README.md).

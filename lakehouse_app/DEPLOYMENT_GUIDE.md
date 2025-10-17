# Deployment Guide - CV Training Pipeline Lakehouse App

This guide walks you through deploying the Computer Vision Training Pipeline as a Databricks Lakehouse App.

## 📋 Prerequisites

### 1. Databricks Workspace Requirements
- Databricks workspace with Lakehouse Apps enabled
- Unity Catalog enabled
- Access to GPU clusters
- Permissions to create apps, clusters, and jobs

### 2. Unity Catalog Setup
- Catalog, schema, and volume for data storage
- Model registry permissions
- Volumes for checkpoints and results

### 3. Source Code Access
- CV framework code in workspace (e.g., `/Workspace/Repos/<username>/Databricks_CV_ref/`)
- App code uploaded or synced

## 🚀 Deployment Steps

### Step 1: Prepare Unity Catalog

```sql
-- Create catalog and schema
CREATE CATALOG IF NOT EXISTS main;
CREATE SCHEMA IF NOT EXISTS main.cv_models;

-- Create volumes
CREATE VOLUME IF NOT EXISTS main.cv_models.cv_data;

-- Verify volumes
SHOW VOLUMES IN main.cv_models;
```

### Step 2: Upload App Code

**Option A: Using Databricks Repos (Recommended)**
1. Connect your Git repository
2. Sync the repository to workspace
3. App will be at `/Workspace/Repos/<username>/Databricks_CV_ref/lakehouse_app/`

**Option B: Manual Upload**
1. Zip the `lakehouse_app` directory
2. Upload via Databricks UI to `/Workspace/Users/<username>/lakehouse_app/`
3. Extract files

### Step 3: Create Lakehouse App

**Using Databricks UI:**

1. Go to **Apps** in your workspace
2. Click **Create App**
3. Configure the app:
   - **Name**: `cv-training-pipeline`
   - **Path**: Path to `lakehouse_app` directory
   - **Description**: Computer Vision Training Pipeline
   - **Compute**: Choose or create compute profile

**Using Databricks CLI:**

```bash
databricks apps create \
  --name cv-training-pipeline \
  --path /Workspace/Users/<username>/lakehouse_app \
  --description "Computer Vision Training Pipeline"
```

### Step 4: Configure App

The app will use the `app.yaml` configuration:

```yaml
command: ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]

env:
  - name: STREAMLIT_SERVER_HEADLESS
    value: "true"
  - name: STREAMLIT_BROWSER_GATHER_USAGE_STATS
    value: "false"
```

### Step 5: Start the App

1. Click **Start** in the Apps UI
2. Wait for app to provision (2-3 minutes)
3. Click the app URL to open

## ⚙️ Configuration

### Environment Variables

Set these in the app configuration if needed:

```yaml
env:
  - name: DATABRICKS_HOST
    value: "https://<workspace-url>"
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
```

### Streamlit Configuration

Edit `.streamlit/config.toml` for customization:

```toml
[server]
headless = true
port = 8080
enableCORS = false

[theme]
primaryColor = "#FF3621"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

## 🔐 Security & Permissions

### Required Permissions

**Workspace Permissions:**
- CREATE apps
- READ/WRITE to Unity Catalog
- CREATE/RUN jobs
- CREATE/MANAGE clusters

**Unity Catalog Permissions:**
```sql
-- Grant permissions to users
GRANT USE CATALOG ON CATALOG main TO `user@company.com`;
GRANT USE SCHEMA ON SCHEMA main.cv_models TO `user@company.com`;
GRANT ALL PRIVILEGES ON VOLUME main.cv_models.cv_data TO `user@company.com`;
```

### Authentication

The app uses Databricks authentication automatically:
- User credentials are inherited from workspace session
- No additional authentication required
- Token-based API calls handled by Databricks SDK

## 📊 Resource Management

### Compute Requirements

**For the App:**
- Small: 2 cores, 8 GB RAM (recommended)
- No GPU needed for the app interface

**For Training Jobs:**
- GPU clusters (g5.4xlarge or larger)
- Configured per job in the app

### Cost Optimization

1. **Enable scale-to-zero** for serving endpoints
2. **Use job clusters** instead of all-purpose clusters for training
3. **Set timeouts** for long-running jobs
4. **Monitor resource usage** in the History page

## 🔄 Updates & Maintenance

### Updating the App

1. **Via Repos:**
   ```bash
   git pull
   # App will auto-reload on next access
   ```

2. **Manual Update:**
   - Upload new version
   - Restart the app

### Monitoring

**App Health:**
- Check app status in Apps UI
- View logs: Click "Logs" in app details
- Monitor resource usage

**Training Jobs:**
- Use the Training page for active monitoring
- Check MLflow for experiment tracking
- View job history in Jobs UI

## 🐛 Troubleshooting

### App Won't Start

**Issue**: App fails to start

**Solutions:**
1. Check logs for error messages
2. Verify `requirements.txt` dependencies
3. Ensure `app.yaml` is valid
4. Check compute profile has enough resources

```bash
# View app logs
databricks apps logs cv-training-pipeline
```

### Permission Errors

**Issue**: "Access denied" errors

**Solutions:**
1. Verify Unity Catalog permissions
2. Check workspace permissions
3. Ensure user has access to volumes

### Import Errors

**Issue**: "Module not found" errors

**Solutions:**
1. Verify all dependencies in `requirements.txt`
2. Check Python version compatibility
3. Rebuild app environment

### Data Access Issues

**Issue**: Cannot read/write data

**Solutions:**
1. Verify volume paths
2. Check Unity Catalog permissions
3. Ensure volumes exist
4. Test with `dbfs` commands

## 📈 Scaling

### For More Users

1. **Increase app compute:**
   - Medium: 4 cores, 16 GB RAM
   - Large: 8 cores, 32 GB RAM

2. **Use shared clusters** for training jobs

3. **Implement queuing** for job submissions

### For More Data

1. **Use Delta tables** for large datasets
2. **Enable caching** in data loading
3. **Use distributed training** (Ray/DDP)
4. **Optimize data formats** (Parquet, Delta)

## 🔗 Integration

### With Existing Systems

**MLflow Integration:**
- Uses workspace MLflow automatically
- Experiments tracked to workspace path
- Models registered to Unity Catalog

**Job Integration:**
- Calls existing `jobs/*.py` scripts
- Uses workspace compute
- Inherits job configurations

**Data Integration:**
- Reads from Unity Catalog Volumes
- Supports Delta tables
- Compatible with DBFS

## 🧪 Testing

### Before Production

1. **Test with sample data:**
   - Use small dataset
   - Run quick training job
   - Verify end-to-end workflow

2. **Test all features:**
   - Config creation
   - Job submission
   - Model registration
   - Deployment

3. **Load testing:**
   - Multiple concurrent users
   - Large batch inference
   - Long training runs

### Health Checks

Create a monitoring dashboard:
- App uptime
- Job success rate
- Endpoint availability
- Resource usage

## 📚 Additional Setup

### Custom Models

To add custom models:

1. Edit `utils/config_generator.py`:
   ```python
   TASK_MODELS = {
       "detection": [
           {"name": "custom/model", "display": "Custom Model", "size": "Medium"},
           # ... existing models
       ]
   }
   ```

2. Restart the app

### Custom Themes

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#YOUR_COLOR"
backgroundColor = "#YOUR_COLOR"
secondaryBackgroundColor = "#YOUR_COLOR"
textColor = "#YOUR_COLOR"
font = "sans serif"
```

## 🆘 Support

### Getting Help

1. Check app logs
2. Review Databricks documentation
3. Consult workspace admin
4. Contact Databricks support

### Useful Commands

```bash
# List apps
databricks apps list

# Get app details
databricks apps get cv-training-pipeline

# View logs
databricks apps logs cv-training-pipeline

# Restart app
databricks apps restart cv-training-pipeline

# Delete app
databricks apps delete cv-training-pipeline
```

## 📝 Checklist

Before going live:

- [ ] Unity Catalog setup complete
- [ ] Permissions configured
- [ ] App deployed and tested
- [ ] Sample training job successful
- [ ] Model registration working
- [ ] Endpoints can be created
- [ ] Documentation shared with users
- [ ] Monitoring configured
- [ ] Backup/recovery plan in place

---

**Questions?** Contact your Databricks representative or workspace admin.


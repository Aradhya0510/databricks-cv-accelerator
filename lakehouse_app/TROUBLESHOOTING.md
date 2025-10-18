# Databricks App Deployment Troubleshooting

## Issue: "App Not Available" Error

If you see "App Not Available" after deployment, despite the app showing as "active", this typically indicates the app is crashing on startup.

### Common Causes:

1. **Import errors** - The app cannot find required modules
2. **Dependency issues** - Missing or incompatible packages
3. **Path problems** - Source code path is incorrect
4. **Permission issues** - App cannot access required resources

### Solution Steps:

#### 1. Check App Logs (Most Important!)

In your Databricks workspace terminal, run:
```bash
databricks apps logs cv-training-pipeline
```

This will show you the actual error causing the app to crash.

#### 2. Verify Deployment Path

Make sure you're deploying from the correct location. Based on your workspace structure, use:

```bash
# If repo is at: /Workspace/Users/<user>/Databricks_CV_ref/
cd /Workspace/Users/aradhya.chouhan@databricks.com/Databricks_CV_ref

# Deploy the app
databricks apps deploy cv-training-pipeline --source-code-path lakehouse_app
```

**Note:** Use relative path `lakehouse_app` from the repo root, not the full absolute path.

#### 3. Ensure All Dependencies Are Available

The app needs access to the parent `src` module. You have two options:

**Option A: Keep the app self-contained (Recommended)**

Modify the app to not depend on parent directory imports. See "Making the App Self-Contained" below.

**Option B: Deploy the entire repository**

```bash
# Deploy from workspace root, but this is NOT recommended for Databricks Apps
# Apps should be self-contained within a single directory
```

### Making the App Self-Contained (Recommended Fix)

The current issue is that `app.py` and page files try to import from the parent `src` directory:

```python
# This causes problems in deployment:
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))
```

#### Fix Steps:

1. **Copy required utilities into lakehouse_app**
   
   If the pages need anything from `src/`, copy those specific modules into `lakehouse_app/utils/` or create wrappers.

2. **Update app.yaml to install parent repo as a package** (Alternative)
   
   Add this to your `app.yaml`:
   ```yaml
   command: ["sh", "-c", "pip install -e /path/to/parent && streamlit run app.py --server.port 8080 --server.address 0.0.0.0"]
   ```

3. **Remove parent directory path manipulation**
   
   If lakehouse_app is truly self-contained, remove these lines from `app.py` and all page files:
   ```python
   # Remove or comment out:
   workspace_root = Path(__file__).parent.parent
   if str(workspace_root) not in sys.path:
       sys.path.insert(0, str(workspace_root))
   ```

### Redeploy After Fixes

```bash
# Stop the existing app
databricks apps stop cv-training-pipeline

# Delete the app (if needed for a clean restart)
databricks apps delete cv-training-pipeline

# Deploy again
cd /Workspace/Users/aradhya.chouhan@databricks.com/Databricks_CV_ref
databricks apps deploy cv-training-pipeline --source-code-path lakehouse_app

# Check logs immediately
databricks apps logs cv-training-pipeline
```

### Verify Deployment Structure

Your `lakehouse_app` directory should contain:
```
lakehouse_app/
├── app.py              # Main entry point
├── app.yaml            # Deployment config
├── requirements.txt    # All dependencies
├── .streamlit/
│   └── config.toml
├── pages/              # Streamlit pages
│   ├── 1_⚙️_Config_Setup.py
│   └── ...
├── components/         # UI components
│   └── ...
├── utils/              # Utility modules (should be self-contained)
│   └── ...
```

### Debug Checklist:

- [ ] Check app logs: `databricks apps logs cv-training-pipeline`
- [ ] Verify deployment path is correct
- [ ] Ensure `app.yaml` command is correct
- [ ] Verify all imports in `app.py` and pages can be resolved
- [ ] Check that `requirements.txt` includes all necessary packages
- [ ] Test locally first: `streamlit run lakehouse_app/app.py`
- [ ] Verify app has necessary Databricks workspace permissions

### Testing Locally First:

Before deploying, test the app locally to catch import errors:

```bash
cd /Workspace/Users/aradhya.chouhan@databricks.com/Databricks_CV_ref/lakehouse_app
streamlit run app.py
```

If this fails locally, fix the errors before deploying.

### Common Error Patterns in Logs:

- **"ModuleNotFoundError"** → Import/dependency issue
- **"FileNotFoundError"** → Path issue
- **"PermissionError"** → Access control issue  
- **"Port already in use"** → App didn't stop properly

### Getting Help:

If issues persist, check:
1. App logs for specific error messages
2. Databricks workspace permissions for the app
3. Whether the cluster/serverless compute has necessary libraries

### Contact:

For further assistance, share:
- Full output of `databricks apps logs cv-training-pipeline`
- Your deployment command
- Any modifications you made to the code


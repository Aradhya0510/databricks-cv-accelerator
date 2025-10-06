# Quick Start Guide - Version 0.1.0

## 🚀 Running the App

### From the streamlit_app directory:

```bash
./run.sh
```

That's it! The app will open at `http://localhost:8501`

---

## 📝 Using the App

### 1. Configuration Builder Tab

**Step-by-step workflow:**

1. **Select Model**
   - Choose from DETR or YOLOS architectures
   - Adjust confidence threshold, IoU threshold, max detections

2. **Configure Data Paths**
   - Enter your Unity Catalog: catalog/schema/volume
   - Paths auto-populate with the correct structure
   - Add optional test dataset if needed
   - Configure batch size, workers, image size
   - Enable/disable augmentation

3. **Training Settings**
   - Set epochs, learning rate, weight decay
   - Configure early stopping and monitoring
   - Choose checkpoint directory
   - (Optional) Enable distributed training

4. **MLflow Settings**
   - Name your experiment and run
   - Add custom tags

5. **Output Configuration**
   - Set results directory
   - Configure visualization options

### 2. Generate YAML

Click the buttons at the bottom:
- **Validate Configuration** - Check for errors
- **Generate YAML** - Create the config file
- **Deploy Job** - (Coming in Phase 2)

### 3. Preview Tab

- View generated YAML with syntax highlighting
- Download the file
- Copy/paste into your configs/ directory
- Use with existing CV Accelerator notebooks

---

## 🎯 Example Workflow

**Quick test configuration:**

1. Model: DETR ResNet-50
2. Catalog: `my_catalog`, Schema: `cv_data`, Volume: `training`
3. Batch Size: 16
4. Max Epochs: 50
5. Click "Validate" → "Generate YAML"
6. Download and use!

---

## 🐛 Troubleshooting

**App won't start:**
```bash
# Ensure you're in streamlit_app directory
cd streamlit_app

# Check venv exists
ls -la .venv

# If not, create it:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Module import errors:**
```bash
# The app needs access to parent src/ directory
# This is handled automatically, but if issues:
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
```

**Port already in use:**
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

---

## 📦 What's Included

- ✅ Full object detection configuration
- ✅ 4 model variants (DETR 50/101, YOLOS tiny/small)
- ✅ All config sections (model, data, training, mlflow, output)
- ✅ Form validation
- ✅ YAML preview and download
- ✅ Help tooltips throughout

---

## 🚧 Current Limitations (v0.1.0)

- ⚠️ No persistence (refresh loses changes)
- ⚠️ Detection task only (classification/segmentation coming)
- ⚠️ No job deployment yet
- ⚠️ Can't save/load configs (coming in v0.2.0 with Lakebase)

---

## 📖 Full Documentation

See `README.md` for complete documentation including:
- Architecture details
- Adding new models
- Custom validation
- Deployment to Databricks Apps

---

## 🎉 Version 0.2.0 Preview

Coming soon:
- 💾 Save/load configurations (Lakebase)
- 🚀 One-click job deployment (Databricks SDK)
- 📊 Job history and monitoring
- 🎯 Classification and segmentation tasks
- 👥 Team collaboration features

---

**Questions?** Check `../CLAUDE.md` for development notes and session history.

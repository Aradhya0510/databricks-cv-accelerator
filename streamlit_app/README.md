# Databricks CV Accelerator - Streamlit Configuration App

## Version 0.1.0: Configuration Builder

A Streamlit-based web application for building training configurations for the Databricks CV Accelerator framework. This replaces manual YAML editing with an intuitive form-based interface.

---

## Features

### Current Release (v0.1.0)
- ✅ **Object Detection Configuration**: Support for DETR and YOLOS models
- ✅ **Interactive Form Builder**: All 5 configuration sections (model, data, training, mlflow, output)
- ✅ **Real-time YAML Preview**: Live preview with syntax highlighting
- ✅ **Validation**: Built-in configuration validation
- ✅ **Export**: Download generated YAML files
- ✅ **In-memory State**: Session-based configuration management

### Future Releases
- 🔄 **Lakebase Integration**: Save/load configurations
- 🔄 **Job Deployment**: One-click Databricks job creation
- 🔄 **History Tracking**: View past deployments and configs
- 🔄 **Multiple Task Types**: Classification and segmentation support

### Planned Features
- 🔮 **Collaborative Features**: Share configs with team
- 🔮 **Model Leaderboards**: Compare experiment results
- 🔮 **Advanced Branching**: Config versioning and A/B testing

---

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   streamlit run app.py
   ```

3. **Access the UI**
   - Opens automatically in browser at `http://localhost:8501`
   - If not, navigate to the URL shown in terminal

### Project Structure

```
streamlit_app/
├── app.py                  # Main Streamlit application
├── config_builder.py       # Form generation logic
├── yaml_generator.py       # YAML generation and validation
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── templates/             # YAML templates (future)
```

---

## Usage Guide

### Building a Configuration

1. **Select Model**
   - Choose from DETR or YOLOS architectures
   - Adjust detection-specific parameters (confidence, IoU, max detections)

2. **Configure Data**
   - Enter Unity Catalog paths (catalog/schema/volume)
   - Set training and validation data paths
   - Configure batch size, image size, and augmentations

3. **Set Training Parameters**
   - Max epochs, learning rate, weight decay
   - Early stopping and monitoring metrics
   - Distributed training options (optional)

4. **MLflow Configuration**
   - Experiment and run names
   - Custom tags for organization

5. **Output Settings**
   - Results directory
   - Visualization options

### Generating YAML

1. Fill out the configuration form
2. Click **"Validate Configuration"** to check for errors
3. Click **"Generate YAML"** to create the configuration file
4. Switch to **"YAML Preview"** tab to view/download

### Exporting Configuration

- Use the **"Download YAML"** button in the Preview tab
- Save to your local `configs/` directory
- Ready to use with existing CV Accelerator notebooks

---

## Configuration Sections

### 1. Model Configuration
- Model architecture selection
- Task-specific settings (detection thresholds)
- Training hyperparameters (learning rate, scheduler)

### 2. Data Configuration
- Unity Catalog paths
- Data loading parameters (batch size, workers)
- Image preprocessing (size, normalization)
- Augmentation options

### 3. Training Configuration
- Epochs and optimization settings
- Early stopping and monitoring
- Checkpointing strategy
- Distributed training setup

### 4. MLflow Configuration
- Experiment tracking settings
- Run naming and tags
- Model logging options

### 5. Output Configuration
- Results directories
- Prediction saving
- Visualization settings

---

## Supported Models

### Object Detection

| Model | Display Name | Image Size | Default Batch Size | Description |
|-------|-------------|------------|-------------------|-------------|
| `facebook/detr-resnet-50` | DETR ResNet-50 | 800 | 16 | Balanced speed/accuracy |
| `facebook/detr-resnet-101` | DETR ResNet-101 | 800 | 8 | Higher accuracy, slower |
| `hustvl/yolos-tiny` | YOLOS Tiny | 512 | 32 | Fast and lightweight |
| `hustvl/yolos-small` | YOLOS Small | 512 | 24 | Good tradeoff |

---

## Validation

The app performs multi-level validation:

1. **Form-level**: Prevents invalid inputs (e.g., negative numbers)
2. **Structure**: Ensures all required sections present
3. **Semantic**: Validates paths, ranges, dependencies
4. **Framework**: Integrates with existing `config_validator.py` (if available)

---

## Deployment to Databricks Apps

### Prerequisites
- Databricks workspace with Apps enabled
- Unity Catalog configured
- (Phase 2) Lakebase database provisioned

### Deployment Steps

1. **Package the App**
   ```bash
   # From project root
   cd streamlit_app
   ```

2. **Create Databricks App**
   - Use Databricks CLI or UI
   - Point to `app.py` as entry point
   - Configure environment with `requirements.txt`

3. **Set Up Authentication**
   - App will use Databricks OAuth automatically
   - No manual credential configuration needed

4. **Access the App**
   - Navigate to Apps section in Databricks UI
   - Launch the CV Accelerator Config Builder

---

## Development

### Adding New Models

1. Edit `config_builder.py`:
   ```python
   DETECTION_MODELS = {
       "your-model-id": {
           "display_name": "Your Model Name",
           "description": "Model description",
           "default_image_size": 512,
           "default_batch_size": 16,
       },
       # ... existing models
   }
   ```

2. No other changes needed - form auto-generates!

### Adding New Task Types

1. Create new builder function in `config_builder.py`:
   ```python
   def build_classification_config() -> Dict[str, Any]:
       # Similar to build_detection_config()
       pass
   ```

2. Add task selection in `app.py`:
   ```python
   task_type = st.selectbox("Task Type", ["detection", "classification"])
   if task_type == "detection":
       config = build_detection_config()
   elif task_type == "classification":
       config = build_classification_config()
   ```

### Custom Validation Rules

Edit `yaml_generator.py` to add custom validation:

```python
def validate_config_dict(config_dict: Dict[str, Any]) -> Tuple[bool, str]:
    # Add your custom validation logic
    if custom_condition:
        return False, "Custom error message"
    # ... rest of validation
```

---

## Troubleshooting

### "Module not found" errors
```bash
# Ensure parent src directory is accessible
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
streamlit run app.py
```

### Validation fails but config looks correct
- Check that paths use Unity Catalog format: `/Volumes/catalog/schema/volume/...`
- Ensure numeric values are positive
- Verify required fields are not empty

### YAML preview not updating
- Click "Generate YAML" button after making changes
- Check browser console for JavaScript errors
- Try refreshing the page

---

## Testing

### Manual Testing Checklist

- [ ] App launches without errors
- [ ] All form fields render correctly
- [ ] Model selection updates dependent fields
- [ ] Unity Catalog path builder works
- [ ] Validation catches invalid inputs
- [ ] YAML preview displays correctly
- [ ] Download button works
- [ ] Session state persists across tabs

### Example Test Config

Use these values for a quick test:

```yaml
Model: DETR ResNet-50
Num Classes: 80
Catalog: test_catalog
Schema: test_schema
Volume: test_volume
Batch Size: 16
Max Epochs: 50
```

---

## Known Limitations (v0.1.0)

- ❌ No persistence (configurations lost on page refresh)
- ❌ No job deployment (manual notebook execution still required)
- ❌ Detection task only (classification/segmentation coming in Phase 2)
- ❌ No multi-user support
- ❌ No configuration history

These will be addressed in future releases with Lakebase integration.

---

## Contributing

Improvements welcome! Focus areas:

1. **UI/UX**: Better form layouts, help text, tooltips
2. **Validation**: More comprehensive checks
3. **Templates**: Pre-built configs for common scenarios
4. **Documentation**: Usage guides, video tutorials

---

## Related Files

- **Main Project**: `../README.md` - Full CV Accelerator documentation
- **Config Schema**: `../src/config.py` - Configuration dataclasses
- **Validator**: `../src/utils/config_validator.py` - Validation logic
- **Example Configs**: `../configs/` - YAML templates

---

## Support

For issues or questions:
1. Check this README
2. Review `../CLAUDE.md` for development notes
3. Check existing configurations in `../configs/`
4. Review main project documentation

---

## Roadmap

### Version 0.1.0 ✅ (Current)
- Basic UI with detection configs
- YAML generation
- Form validation

### Version 0.2.0 🔄 (Planned)
- Lakebase integration for persistence
- Job deployment via Databricks SDK
- Configuration history
- Multiple task types

### Version 0.3.0+ 🔮 (Future)
- Team collaboration features
- Model leaderboards
- Advanced branching
- Template marketplace

---

**Version**: 0.1.0
**Last Updated**: October 1, 2025
**Status**: Development - Ready for testing

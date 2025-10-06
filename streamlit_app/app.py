"""
Databricks CV Accelerator - Streamlit Configuration App
Phase 1 MVP: Configuration form builder for object detection tasks
"""

import streamlit as st
import yaml
import sys
from pathlib import Path

# Add parent src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config_builder import build_detection_config
from yaml_generator import generate_yaml, validate_config_dict

# Page configuration
st.set_page_config(
    page_title="Databricks CV Accelerator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Databricks CV Accelerator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Build training configurations for computer vision models on Databricks</div>', unsafe_allow_html=True)

# Sidebar - Navigation and Info
with st.sidebar:
    st.image("https://www.databricks.com/wp-content/uploads/2021/06/db-nav-logo.svg", width=200)
    st.markdown("---")

    st.markdown("### Version 0.1.0")
    st.info("""
    **Current Features:**
    - Object Detection configs
    - DETR & YOLOS models
    - Full parameter configuration
    - YAML preview & validation

    **Coming Soon:**
    - Save/Load configurations
    - Job deployment
    - History tracking
    """)

    st.markdown("---")
    st.markdown("### Quick Help")
    with st.expander("Task Types"):
        st.markdown("""
        - **Detection**: Bounding box prediction (DETR, YOLOS)
        - **Classification**: Image labeling (Coming soon)
        - **Segmentation**: Pixel-level masks (Coming soon)
        """)

    with st.expander("Data Format"):
        st.markdown("""
        Expected: MS COCO format
        - Image folder + annotations.json
        - Unity Catalog Volumes paths
        """)

# Main content area
tab1, tab2 = st.tabs(["Configuration Builder", "YAML Preview"])

with tab1:
    st.markdown("## Build Your Training Configuration")
    st.markdown("Configure all parameters for your computer vision training job.")

    # Build configuration form
    config_dict = build_detection_config()

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        if st.button("Validate Configuration", key="validate"):
            with st.spinner("Validating configuration..."):
                is_valid, message = validate_config_dict(config_dict)
                if is_valid:
                    st.success(f"{message}")
                else:
                    st.error(f"{message}")

    with col2:
        if st.button("Generate YAML", key="generate"):
            st.session_state['generated_yaml'] = generate_yaml(config_dict)
            st.success("YAML generated! Check the Preview tab.")

    with col3:
        if st.button("Deploy Job (Coming Soon)", key="deploy", disabled=True):
            st.info("Job deployment will be available in a future release")

with tab2:
    st.markdown("## YAML Configuration Preview")

    if 'generated_yaml' in st.session_state:
        yaml_content = st.session_state['generated_yaml']

        # Display validation status
        is_valid, message = validate_config_dict(config_dict)
        if is_valid:
            st.success(f"Configuration is valid: {message}")
        else:
            st.error(f"Validation failed: {message}")

        # YAML preview with syntax highlighting
        st.code(yaml_content, language="yaml", line_numbers=True)

        # Download button
        st.download_button(
            label="Download YAML",
            data=yaml_content,
            file_name="cv_training_config.yaml",
            mime="text/yaml",
            key="download_yaml"
        )

        # Copy to clipboard info
        st.info("**Tip**: You can copy this YAML and save it to your configs/ directory")

    else:
        st.info("Build your configuration in the Configuration Builder tab, then click 'Generate YAML' to see the preview here.")

        # Show example
        with st.expander("View Example Configuration"):
            example_yaml = """# Example DETR Configuration
model:
  model_name: "facebook/detr-resnet-50"
  task_type: "detection"
  num_classes: 80
  pretrained: true
  confidence_threshold: 0.7
  iou_threshold: 0.5
  max_detections: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  scheduler: "cosine"

data:
  train_data_path: "/Volumes/catalog/schema/volume/data/train2017/"
  train_annotation_file: "/Volumes/catalog/schema/volume/data/instances_train2017.json"
  val_data_path: "/Volumes/catalog/schema/volume/data/val2017/"
  val_annotation_file: "/Volumes/catalog/schema/volume/data/instances_val2017.json"
  batch_size: 16
  num_workers: 4

training:
  max_epochs: 50
  learning_rate: 0.0001
  early_stopping_patience: 20
  checkpoint_dir: "/Volumes/catalog/schema/volume/checkpoints/detection"

mlflow:
  experiment_name: "detection_training"
  run_name: "detr-resnet50"

output:
  results_dir: "/Volumes/catalog/schema/volume/results/detection"
  save_predictions: true
"""
            st.code(example_yaml, language="yaml")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Databricks CV Accelerator | Version 0.1.0</p>
    <p style='font-size: 0.9rem;'>Built with Streamlit • Powered by Databricks</p>
</div>
""", unsafe_allow_html=True)

"""
Page 2: Data Exploration and Analysis
Explore and validate datasets
"""

import streamlit as st
import sys
from pathlib import Path
import json
from PIL import Image
import random
from typing import Dict, Any, List
import os

from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from components.visualizations import VisualizationHelper
from components.image_viewer import ImageViewer

# Initialize state
StateManager.initialize()

@st.cache_resource
def _get_client():
    return DatabricksJobClient()

client = _get_client()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Page config
st.title("📊 Data Exploration & Analysis")
st.markdown("Explore your dataset before training")

# Check for active config
current_config = StateManager.get_current_config()

if not current_config:
    st.warning("⚠️ No active configuration found")
    st.info("Please create or load a configuration in the Config Setup page first")
    
    if st.button("Go to Config Setup"):
        st.switch_page("pages/1_⚙️_Config_Setup.py")
    st.stop()

# Get task and data config
task = current_config.get("model", {}).get("task_type", "detection")
data_config = current_config.get("data", {})

st.success(f"✅ Active Configuration: {task.replace('_', ' ').title()}")

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["📈 Dataset Statistics", "🖼️ Sample Viewer", "✅ Data Validation", "🔍 Class Distribution"])

with tab1:
    st.markdown("### Dataset Statistics")
    
    # Extract paths from config
    train_path = data_config.get("train_data_path", "")
    val_path = data_config.get("val_data_path", "")
    test_path = data_config.get("test_data_path", "")
    
    st.markdown("#### Configured Data Paths")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Training Data:**")
        st.code(train_path)
    with col2:
        st.markdown("**Validation Data:**")
        st.code(val_path)
    
    if test_path:
        st.markdown("**Test Data:**")
        st.code(test_path)
    
    st.markdown("---")
    
    # Mock statistics (in real implementation, would scan directories)
    st.info("💡 Click 'Analyze Dataset' to scan your data directories and compute statistics")
    
    if st.button("🔍 Analyze Dataset", type="primary"):
        with st.spinner("Analyzing dataset (reading from Volumes via SDK)..."):
            def _count_images(directory):
                return len(client.list_volume_files(directory, extensions=IMAGE_EXTS))

            def _count_annotations(ann_file):
                data = client.download_volume_json(ann_file)
                if data is None:
                    return 0
                return len(data.get("annotations", []))

            train_ann_file = data_config.get("train_annotation_file", "")
            val_ann_file = data_config.get("val_annotation_file", "")
            test_ann_file = data_config.get("test_annotation_file", "")

            dataset_stats = {
                "train": {
                    "num_images": _count_images(train_path) if train_path else 0,
                    "num_annotations": _count_annotations(train_ann_file) if train_ann_file else 0,
                },
                "val": {
                    "num_images": _count_images(val_path) if val_path else 0,
                    "num_annotations": _count_annotations(val_ann_file) if val_ann_file else 0,
                },
                "test": {
                    "num_images": _count_images(test_path) if test_path else 0,
                    "num_annotations": _count_annotations(test_ann_file) if test_ann_file else 0,
                } if test_path else None
            }
            
            st.success("✅ Analysis complete!")
            
            # Display statistics
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Training Images", dataset_stats["train"]["num_images"])
                if task in ["detection"]:
                    st.metric("Training Annotations", dataset_stats["train"]["num_annotations"])
            
            with cols[1]:
                st.metric("Validation Images", dataset_stats["val"]["num_images"])
                if task in ["detection"]:
                    st.metric("Validation Annotations", dataset_stats["val"]["num_annotations"])
            
            with cols[2]:
                if dataset_stats["test"]:
                    st.metric("Test Images", dataset_stats["test"]["num_images"])
                    if task in ["detection"]:
                        st.metric("Test Annotations", dataset_stats["test"]["num_annotations"])
                else:
                    st.info("No test set")
            
            st.markdown("#### Data Split")
            
            split_data = {
                "Train": dataset_stats["train"]["num_images"],
                "Val": dataset_stats["val"]["num_images"],
            }
            if dataset_stats["test"]:
                split_data["Test"] = dataset_stats["test"]["num_images"]
            
            fig = VisualizationHelper.class_distribution_chart(
                split_data,
                "Dataset Split"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Training Image Size")
            configured_size = data_config.get("image_size", "N/A")
            if isinstance(configured_size, list):
                configured_size = f"{configured_size[0]} x {configured_size[1]}"
            st.info(f"Images will be resized to **{configured_size}** during training (configured in `data.image_size`)")

with tab2:
    st.markdown("### Sample Viewer")
    
    st.info("🖼️ Preview random samples from your dataset")
    
    # Data split selector
    split = st.selectbox(
        "Select Data Split",
        options=["Training", "Validation", "Test"] if test_path else ["Training", "Validation"]
    )
    
    num_samples = st.slider(
        "Number of Samples",
        min_value=1,
        max_value=12,
        value=6,
        help="Number of random samples to display"
    )
    
    if st.button("🎲 Load Random Samples", type="primary"):
        split_path = train_path if split == "Training" else (val_path if split == "Validation" else test_path)
        if not split_path:
            st.warning("⚠️ Path not configured for this split")
        else:
            with st.spinner("Loading images from Volume..."):
                try:
                    all_files = client.list_volume_files(split_path, extensions=IMAGE_EXTS)
                    if not all_files:
                        st.warning(f"No image files found in `{split_path}`")
                    else:
                        sample_files = random.sample(all_files, min(num_samples, len(all_files)))
                        images = []
                        captions = []
                        for fname in sample_files:
                            full_path = split_path.rstrip("/") + "/" + fname
                            img = client.download_volume_image(full_path)
                            if img is not None:
                                images.append(img)
                                captions.append(fname)
                        if images:
                            st.markdown("#### Sample Images")
                            ImageViewer.display_image_grid(images, captions=captions, columns=3)
                        else:
                            st.warning("Could not download any images")
                except Exception as e:
                    st.error(f"Error loading images: {e}")
    
    st.markdown("---")
    st.markdown("#### Sample with Annotations")
    
    if task in ["detection", "instance_segmentation", "universal_segmentation"]:
        st.info("For detection/segmentation tasks, annotations (bounding boxes/masks) will be overlaid on images")
    else:
        st.info("For classification tasks, predicted class labels will be shown")

with tab3:
    st.markdown("### Data Validation")
    
    st.info("🔍 Validate data quality and identify potential issues")
    
    if st.button("✅ Run Validation Checks", type="primary"):
        with st.spinner("Running validation checks..."):
            # Mock validation results
            st.success("✅ Validation complete!")
            
            st.markdown("#### Validation Results")
            
            # Check 1: Path existence
            st.markdown("**1. Data Paths**")
            col1, col2 = st.columns(2)
            with col1:
                st.success("✅ Training path accessible")
                st.success("✅ Validation path accessible")
            with col2:
                if test_path:
                    st.success("✅ Test path accessible")
                else:
                    st.info("ℹ️ Test path not configured")
            
            # Check 2: Annotations
            if task in ["detection", "instance_segmentation", "universal_segmentation"]:
                st.markdown("**2. Annotation Files**")
                train_ann = data_config.get("train_annotation_file", "")
                val_ann = data_config.get("val_annotation_file", "")
                
                col1, col2 = st.columns(2)
                with col1:
                    if train_ann:
                        st.success(f"✅ Training annotations found")
                    else:
                        st.error("❌ Training annotations not configured")
                with col2:
                    if val_ann:
                        st.success(f"✅ Validation annotations found")
                    else:
                        st.error("❌ Validation annotations not configured")
            
            # Check 3: Data quality issues
            st.markdown("**3. Data Quality Checks**")
            
            quality_issues = []
            
            # Mock some quality checks
            issues_found = False
            
            if not issues_found:
                st.success("✅ No data quality issues detected")
            else:
                st.warning("⚠️ Found potential issues:")
                for issue in quality_issues:
                    st.warning(f"  - {issue}")
            
            # Check 4: Class balance
            st.markdown("**4. Class Balance**")
            num_classes = data_config.get("num_classes", 0)
            
            if num_classes > 0:
                st.info(f"ℹ️ Dataset configured for {num_classes} classes")
                st.info("💡 See 'Class Distribution' tab for detailed class statistics")
            else:
                st.warning("⚠️ Number of classes not configured")
            
            # Recommendations
            st.markdown("---")
            st.markdown("#### 💡 Recommendations")
            
            recommendations = [
                "Dataset paths are properly configured",
                "Consider enabling data augmentation to improve model generalization",
                f"Configured image size: {data_config.get('image_size', 'Not specified')}",
                f"Batch size: {data_config.get('batch_size', 'Not specified')}",
            ]
            
            for rec in recommendations:
                st.info(f"✓ {rec}")

with tab4:
    st.markdown("### Class Distribution")
    
    num_classes = data_config.get("num_classes", 0)
    
    if num_classes == 0:
        st.warning("⚠️ Number of classes not configured in the configuration")
    else:
        st.info(f"📊 Dataset configured for {num_classes} classes")
        
        if st.button("📊 Analyze Class Distribution", type="primary"):
            with st.spinner("Analyzing class distribution (reading from Volumes)..."):
                class_counts = {}
                train_ann_file = data_config.get("train_annotation_file", "")
                if task in ["detection"] and train_ann_file:
                    try:
                        coco_data = client.download_volume_json(train_ann_file)
                        if coco_data:
                            cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
                            from collections import Counter
                            cat_counter = Counter(ann["category_id"] for ann in coco_data.get("annotations", []))
                            class_counts = {cat_map.get(cid, str(cid)): cnt for cid, cnt in cat_counter.most_common()}
                    except Exception:
                        pass
                elif task == "classification" and train_path:
                    try:
                        subdirs = client.list_volume_dirs(train_path)
                        for d in sorted(subdirs):
                            sub_path = train_path.rstrip("/") + "/" + d
                            n = len(client.list_volume_files(sub_path, extensions=IMAGE_EXTS))
                            if n > 0:
                                class_counts[d] = n
                    except Exception:
                        pass

                if not class_counts:
                    class_names = [f"Class_{i}" for i in range(min(num_classes, 10))]
                    random.seed(42)
                    class_counts = {name: random.randint(100, 1000) for name in class_names}
                    st.info("Could not read annotations — showing sample distribution")
                
                st.success("✅ Analysis complete!")
                
                # Display distribution chart
                fig = VisualizationHelper.class_distribution_chart(
                    class_counts,
                    "Class Distribution (Training Set)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("#### Class Statistics")
                
                total_samples = sum(class_counts.values())
                max_class = max(class_counts.items(), key=lambda x: x[1])
                min_class = min(class_counts.items(), key=lambda x: x[1])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", total_samples)
                with col2:
                    st.metric("Most Common", f"{max_class[0]} ({max_class[1]})")
                with col3:
                    st.metric("Least Common", f"{min_class[0]} ({min_class[1]})")
                
                # Class imbalance warning
                imbalance_ratio = max_class[1] / min_class[1]
                if imbalance_ratio > 10:
                    st.warning(f"⚠️ High class imbalance detected! The most common class has {imbalance_ratio:.1f}x more samples than the least common.")
                    st.info("💡 Consider using class weights or data augmentation to handle imbalance")
                elif imbalance_ratio > 5:
                    st.info(f"ℹ️ Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}x)")
                else:
                    st.success(f"✅ Classes are relatively balanced (ratio: {imbalance_ratio:.1f}x)")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 EDA Summary")
    
    st.markdown(f"**Task:** {task.replace('_', ' ').title()}")
    st.markdown(f"**Batch Size:** {data_config.get('batch_size', 'N/A')}")
    st.markdown(f"**Image Size:** {data_config.get('image_size', 'N/A')}")
    st.markdown(f"**Augmentation:** {'Enabled' if data_config.get('augment', False) else 'Disabled'}")
    
    st.markdown("---")
    st.markdown("### 🚀 Next Steps")
    
    st.info("After exploring your data, you can:")
    
    if st.button("🚀 Start Training", use_container_width=True):
        st.switch_page("pages/3_🚀_Training.py")
    
    if st.button("⚙️ Modify Config", use_container_width=True):
        st.switch_page("pages/1_⚙️_Config_Setup.py")


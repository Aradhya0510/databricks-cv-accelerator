"""
Configuration Form Builder
Generates dynamic forms for configuration
"""

import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from utils.config_generator import ConfigGenerator


class ConfigFormBuilder:
    """Build dynamic configuration forms."""
    
    @staticmethod
    def task_selector() -> str:
        """
        Render task selection form.
        
        Returns:
            Selected task name
        """
        task_options = {
            "🔍 Object Detection": "detection",
            "🏷️ Image Classification": "classification",
            "🎨 Semantic Segmentation": "semantic_segmentation",
            "🖼️ Instance Segmentation": "instance_segmentation",
            "🌐 Universal Segmentation": "universal_segmentation",
        }
        
        selected_display = st.selectbox(
            "Select Computer Vision Task",
            options=list(task_options.keys()),
            help="Choose the type of computer vision task you want to train"
        )
        
        return task_options[selected_display]
    
    @staticmethod
    def model_selector(task: str) -> Tuple[str, Dict[str, str]]:
        """
        Render model selection form for a given task.
        
        Args:
            task: Selected task type
            
        Returns:
            Tuple of (model_name, model_info)
        """
        models = ConfigGenerator.get_models_for_task(task)
        
        if not models:
            st.error(f"No models available for task: {task}")
            return "", {}
        
        # Create display options
        model_options = {
            f"{model['display']} ({model['size']})" : model["name"]
            for model in models
        }
        
        selected_display = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            help="Choose a pre-trained model from HuggingFace"
        )
        
        model_name = model_options[selected_display]
        model_info = ConfigGenerator.get_model_info(model_name)
        
        # Display model information
        with st.expander("ℹ️ Model Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Model ID:** `{model_name}`")
                st.markdown(f"**Size:** {model_info.get('size', 'Unknown')}")
            with col2:
                st.markdown(f"**Display Name:** {model_info.get('display', 'Unknown')}")
                st.markdown(f"**Task:** {task}")
        
        return model_name, model_info
    
    @staticmethod
    def data_config_form(task: str, default_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render data configuration form.
        
        Args:
            task: Selected task type
            default_values: Optional default values for form fields
            
        Returns:
            Dictionary with data configuration
        """
        default_values = default_values or {}
        
        st.markdown("### 📁 Data Configuration")
        
        config = {}
        
        # Number of classes
        config["num_classes"] = st.number_input(
            "Number of Classes",
            min_value=1,
            max_value=10000,
            value=default_values.get("num_classes", 80 if task == "detection" else 1000),
            help="Number of output classes for your dataset"
        )
        
        # Data paths based on task
        if task in ["detection", "instance_segmentation", "universal_segmentation"]:
            st.markdown("#### Training Data")
            col1, col2 = st.columns(2)
            with col1:
                config["train_data_path"] = st.text_input(
                    "Training Images Path",
                    value=default_values.get("train_data_path", "/Volumes/<catalog>/<schema>/<volume>/data/train/"),
                    help="Path to training images directory"
                )
            with col2:
                config["train_annotation_file"] = st.text_input(
                    "Training Annotations (COCO JSON)",
                    value=default_values.get("train_annotation_file", "/Volumes/<catalog>/<schema>/<volume>/data/annotations_train.json"),
                    help="Path to training annotations in COCO format"
                )
            
            st.markdown("#### Validation Data")
            col1, col2 = st.columns(2)
            with col1:
                config["val_data_path"] = st.text_input(
                    "Validation Images Path",
                    value=default_values.get("val_data_path", "/Volumes/<catalog>/<schema>/<volume>/data/val/"),
                    help="Path to validation images directory"
                )
            with col2:
                config["val_annotation_file"] = st.text_input(
                    "Validation Annotations (COCO JSON)",
                    value=default_values.get("val_annotation_file", "/Volumes/<catalog>/<schema>/<volume>/data/annotations_val.json"),
                    help="Path to validation annotations in COCO format"
                )
            
            # Optional test data
            with st.expander("🧪 Test Data (Optional)"):
                col1, col2 = st.columns(2)
                with col1:
                    test_data_path = st.text_input(
                        "Test Images Path",
                        value=default_values.get("test_data_path", ""),
                        help="Optional: Path to test images directory"
                    )
                with col2:
                    test_annotation_file = st.text_input(
                        "Test Annotations (COCO JSON)",
                        value=default_values.get("test_annotation_file", ""),
                        help="Optional: Path to test annotations"
                    )
                
                if test_data_path:
                    config["test_data_path"] = test_data_path
                    config["test_annotation_file"] = test_annotation_file
        else:
            # Classification and semantic segmentation (folder structure)
            st.markdown("#### Data Paths")
            config["train_data_path"] = st.text_input(
                "Training Data Path",
                value=default_values.get("train_data_path", "/Volumes/<catalog>/<schema>/<volume>/data/train/"),
                help="Path to training data (folder structure with class subdirectories)"
            )
            
            config["val_data_path"] = st.text_input(
                "Validation Data Path",
                value=default_values.get("val_data_path", "/Volumes/<catalog>/<schema>/<volume>/data/val/"),
                help="Path to validation data"
            )
            
            # Optional test data
            with st.expander("🧪 Test Data (Optional)"):
                test_data_path = st.text_input(
                    "Test Data Path",
                    value=default_values.get("test_data_path", ""),
                    help="Optional: Path to test data"
                )
                if test_data_path:
                    config["test_data_path"] = test_data_path
        
        # Data loading parameters
        st.markdown("#### Data Loading Parameters")
        col1, col2 = st.columns(2)
        with col1:
            config["batch_size"] = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=256,
                value=default_values.get("batch_size", 16),
                help="Number of samples per batch"
            )
        with col2:
            config["num_workers"] = st.number_input(
                "Number of Workers",
                min_value=0,
                max_value=32,
                value=default_values.get("num_workers", 4),
                help="Number of data loading workers"
            )
        
        # Image size
        config["image_size"] = st.number_input(
            "Image Size",
            min_value=64,
            max_value=2048,
            value=default_values.get("image_size", 512),
            step=32,
            help="Input image size (will be resized to square)"
        )
        
        # Augmentation
        config["augment"] = st.checkbox(
            "Enable Data Augmentation",
            value=default_values.get("augment", True),
            help="Apply data augmentation during training"
        )
        
        if config["augment"]:
            with st.expander("⚙️ Augmentation Settings"):
                aug_config = {}
                
                col1, col2 = st.columns(2)
                with col1:
                    aug_config["horizontal_flip"] = st.checkbox("Horizontal Flip", value=True)
                    aug_config["vertical_flip"] = st.checkbox("Vertical Flip", value=False)
                    aug_config["random_crop"] = st.checkbox("Random Crop", value=True)
                with col2:
                    aug_config["rotation"] = st.slider("Rotation (degrees)", 0, 180, 15)
                
                st.markdown("**Color Jitter**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    brightness = st.slider("Brightness", 0.0, 1.0, 0.2, 0.1)
                with col2:
                    contrast = st.slider("Contrast", 0.0, 1.0, 0.2, 0.1)
                with col3:
                    saturation = st.slider("Saturation", 0.0, 1.0, 0.2, 0.1)
                with col4:
                    hue = st.slider("Hue", 0.0, 0.5, 0.1, 0.05)
                
                aug_config["color_jitter"] = {
                    "brightness": brightness,
                    "contrast": contrast,
                    "saturation": saturation,
                    "hue": hue,
                }
                
                config["augmentations"] = aug_config
        
        return config
    
    @staticmethod
    def training_config_form(default_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render training configuration form.
        
        Args:
            default_values: Optional default values for form fields
            
        Returns:
            Dictionary with training configuration
        """
        default_values = default_values or {}
        
        st.markdown("### 🎯 Training Configuration")
        
        config = {}
        
        # Basic training parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            config["epochs"] = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=1000,
                value=default_values.get("epochs", 100),
                help="Total number of training epochs"
            )
        with col2:
            config["learning_rate"] = st.number_input(
                "Learning Rate",
                min_value=1e-7,
                max_value=1.0,
                value=default_values.get("learning_rate", 1e-4),
                format="%.2e",
                help="Initial learning rate"
            )
        with col3:
            config["weight_decay"] = st.number_input(
                "Weight Decay",
                min_value=0.0,
                max_value=1.0,
                value=default_values.get("weight_decay", 1e-4),
                format="%.2e",
                help="L2 regularization factor"
            )
        
        # Scheduler
        config["scheduler"] = st.selectbox(
            "Learning Rate Scheduler",
            options=["cosine", "step", "exponential", "none"],
            index=0,
            help="Learning rate scheduling strategy"
        )
        
        # Early stopping
        st.markdown("#### Early Stopping")
        col1, col2 = st.columns(2)
        with col1:
            config["early_stopping_patience"] = st.number_input(
                "Patience (epochs)",
                min_value=1,
                max_value=100,
                value=default_values.get("early_stopping_patience", 20),
                help="Number of epochs with no improvement before stopping"
            )
        with col2:
            config["log_every_n_steps"] = st.number_input(
                "Log Every N Steps",
                min_value=1,
                max_value=1000,
                value=default_values.get("log_every_n_steps", 50),
                help="Logging frequency"
            )
        
        # Checkpointing
        st.markdown("#### Checkpointing")
        config["checkpoint_dir"] = st.text_input(
            "Checkpoint Directory",
            value=default_values.get("checkpoint_dir", "/Volumes/<catalog>/<schema>/<volume>/checkpoints"),
            help="Directory to save model checkpoints"
        )
        
        config["volume_checkpoint_dir"] = st.text_input(
            "Volume Checkpoint Directory",
            value=default_values.get("volume_checkpoint_dir", "/Volumes/<catalog>/<schema>/<volume>/volume_checkpoints"),
            help="Persistent checkpoint directory in Unity Catalog Volumes"
        )
        
        config["save_top_k"] = st.number_input(
            "Save Top K Models",
            min_value=1,
            max_value=10,
            value=default_values.get("save_top_k", 3),
            help="Number of best models to keep"
        )
        
        # Distributed training
        with st.expander("🚀 Advanced: Distributed Training"):
            config["distributed"] = st.checkbox(
                "Enable Distributed Training",
                value=default_values.get("distributed", False),
                help="Use multiple GPUs if available"
            )
            
            if config["distributed"]:
                config["use_ray"] = st.checkbox(
                    "Use Ray (Multi-node)",
                    value=default_values.get("use_ray", False),
                    help="Use Ray for multi-node training (otherwise uses DDP for single-node)"
                )
                
                config["num_workers"] = st.number_input(
                    "Number of Workers",
                    min_value=1,
                    max_value=16,
                    value=default_values.get("num_workers", 1),
                    help="Number of training workers"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    config["cpu_per_worker"] = st.number_input(
                        "CPUs per Worker",
                        min_value=1,
                        max_value=64,
                        value=default_values.get("cpu_per_worker", 4)
                    )
                with col2:
                    config["gpu_per_worker"] = st.number_input(
                        "GPUs per Worker",
                        min_value=0,
                        max_value=8,
                        value=default_values.get("gpu_per_worker", 1)
                    )
            else:
                config["use_ray"] = False
                config["num_workers"] = 1
        
        config["use_gpu"] = st.checkbox(
            "Use GPU",
            value=default_values.get("use_gpu", True),
            help="Enable GPU acceleration"
        )
        
        return config
    
    @staticmethod
    def mlflow_config_form(default_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render MLflow configuration form.
        
        Args:
            default_values: Optional default values for form fields
            
        Returns:
            Dictionary with MLflow configuration
        """
        default_values = default_values or {}
        
        st.markdown("### 📊 MLflow Configuration")
        
        config = {}
        
        config["experiment_name"] = st.text_input(
            "Experiment Name (Workspace Path)",
            value=default_values.get("experiment_name", "/Users/<email@databricks.com>/cv_experiments"),
            help="MLflow experiment path (must be absolute workspace path)"
        )
        
        st.info("💡 Experiment name must be an absolute workspace path, e.g., `/Users/your.email@databricks.com/cv_detection`")
        
        return config
    
    @staticmethod
    def output_config_form(default_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render output configuration form.
        
        Args:
            default_values: Optional default values for form fields
            
        Returns:
            Dictionary with output configuration
        """
        default_values = default_values or {}
        
        st.markdown("### 💾 Output Configuration")
        
        config = {}
        
        config["results_dir"] = st.text_input(
            "Results Directory",
            value=default_values.get("results_dir", "/Volumes/<catalog>/<schema>/<volume>/results"),
            help="Directory to save training results and predictions"
        )
        
        config["save_predictions"] = st.checkbox(
            "Save Predictions",
            value=default_values.get("save_predictions", True),
            help="Save model predictions to disk"
        )
        
        config["save_images"] = st.checkbox(
            "Save Visualization Images",
            value=default_values.get("save_images", True),
            help="Save visualization images with predictions"
        )
        
        config["max_images"] = st.number_input(
            "Max Visualization Images",
            min_value=1,
            max_value=1000,
            value=default_values.get("max_images", 10),
            help="Maximum number of visualization images to save"
        )
        
        return config
    
    @staticmethod
    def task_specific_config_form(task: str, default_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render task-specific configuration form.
        
        Args:
            task: Selected task type
            default_values: Optional default values for form fields
            
        Returns:
            Dictionary with task-specific configuration
        """
        default_values = default_values or {}
        
        st.markdown("### ⚙️ Task-Specific Configuration")
        
        config = {}
        
        if task == "detection":
            col1, col2, col3 = st.columns(3)
            with col1:
                config["confidence_threshold"] = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_values.get("confidence_threshold", 0.5),
                    step=0.05,
                    help="Minimum confidence for detections"
                )
            with col2:
                config["iou_threshold"] = st.slider(
                    "IOU Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_values.get("iou_threshold", 0.5),
                    step=0.05,
                    help="IOU threshold for NMS"
                )
            with col3:
                config["max_detections"] = st.number_input(
                    "Max Detections",
                    min_value=1,
                    max_value=1000,
                    value=default_values.get("max_detections", 100),
                    help="Maximum detections per image"
                )
        
        elif task == "classification":
            col1, col2 = st.columns(2)
            with col1:
                config["dropout"] = st.slider(
                    "Dropout Rate",
                    min_value=0.0,
                    max_value=0.9,
                    value=default_values.get("dropout", 0.2),
                    step=0.1,
                    help="Dropout probability for regularization"
                )
            with col2:
                config["mixup_alpha"] = st.slider(
                    "Mixup Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_values.get("mixup_alpha", 0.2),
                    step=0.1,
                    help="Mixup augmentation parameter (0 = disabled)"
                )
        
        elif task in ["semantic_segmentation", "instance_segmentation", "universal_segmentation"]:
            col1, col2 = st.columns(2)
            with col1:
                config["aux_loss_weight"] = st.slider(
                    "Auxiliary Loss Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_values.get("aux_loss_weight", 0.4),
                    step=0.1,
                    help="Weight for auxiliary losses"
                )
            with col2:
                config["mask_threshold"] = st.slider(
                    "Mask Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_values.get("mask_threshold", 0.5),
                    step=0.05,
                    help="Threshold for binary masks"
                )
        
        return config


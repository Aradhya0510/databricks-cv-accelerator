"""
Configuration Form Builder
Generates Streamlit forms for building training configurations
"""

import streamlit as st
from typing import Dict, Any


# Model configurations for detection
DETECTION_MODELS = {
    "facebook/detr-resnet-50": {
        "display_name": "DETR ResNet-50",
        "description": "DEtection TRansformer with ResNet-50 backbone. Good balance of speed and accuracy.",
        "default_image_size": 800,
        "default_batch_size": 16,
    },
    "facebook/detr-resnet-101": {
        "display_name": "DETR ResNet-101",
        "description": "Larger DETR model with ResNet-101 backbone. Higher accuracy, slower training.",
        "default_image_size": 800,
        "default_batch_size": 8,
    },
    "hustvl/yolos-tiny": {
        "display_name": "YOLOS Tiny",
        "description": "YOLO-style detection with vision transformers. Fast and lightweight.",
        "default_image_size": 512,
        "default_batch_size": 32,
    },
    "hustvl/yolos-small": {
        "display_name": "YOLOS Small",
        "description": "Medium-sized YOLOS model. Good speed/accuracy tradeoff.",
        "default_image_size": 512,
        "default_batch_size": 24,
    },
}


def build_detection_config() -> Dict[str, Any]:
    """Build configuration dictionary from Streamlit form inputs."""

    config = {
        "model": {},
        "data": {},
        "training": {},
        "mlflow": {},
        "output": {}
    }

    # ========== MODEL CONFIGURATION ==========
    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        model_id = st.selectbox(
            "Model Architecture",
            options=list(DETECTION_MODELS.keys()),
            format_func=lambda x: DETECTION_MODELS[x]["display_name"],
            key="model_name",
            help="Select the pre-trained model architecture to use"
        )

    with col2:
        num_classes = st.number_input(
            "Number of Classes",
            min_value=1,
            max_value=1000,
            value=80,
            key="num_classes",
            help="Number of object classes in your dataset (COCO has 80)"
        )

    # Display model description
    st.info(f"{DETECTION_MODELS[model_id]['description']}")

    # Detection-specific settings
    col1, col2, col3 = st.columns(3)

    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="confidence_threshold",
            help="Minimum confidence score for predictions"
        )

    with col2:
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="iou_threshold",
            help="Intersection over Union threshold for NMS"
        )

    with col3:
        max_detections = st.number_input(
            "Max Detections",
            min_value=10,
            max_value=500,
            value=100,
            key="max_detections",
            help="Maximum number of objects to detect per image"
        )

    # Training hyperparameters
    col1, col2 = st.columns(2)

    with col1:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-2,
            value=1e-4,
            format="%.6f",
            key="model_learning_rate",
            help="Initial learning rate for training"
        )

    with col2:
        weight_decay = st.number_input(
            "Weight Decay",
            min_value=0.0,
            max_value=1e-2,
            value=1e-4,
            format="%.6f",
            key="weight_decay",
            help="L2 regularization parameter"
        )

    # Scheduler configuration
    scheduler = st.selectbox(
        "Learning Rate Scheduler",
        options=["cosine", "step", "exponential", "none"],
        index=0,
        key="scheduler",
        help="Learning rate scheduling strategy"
    )

    scheduler_params = {}
    if scheduler == "cosine":
        col1, col2 = st.columns(2)
        with col1:
            scheduler_params["T_max"] = st.number_input("T_max (Cosine Cycles)", value=300, key="t_max")
        with col2:
            scheduler_params["eta_min"] = st.number_input("Eta Min", value=1e-6, format="%.6f", key="eta_min")

    # Populate model config
    config["model"] = {
        "model_name": model_id,
        "task_type": "detection",
        "num_classes": num_classes,
        "pretrained": True,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "max_detections": max_detections,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "scheduler_params": scheduler_params if scheduler != "none" else None,
    }

    # ========== DATA CONFIGURATION ==========
    st.markdown('<div class="section-header">Data Configuration</div>', unsafe_allow_html=True)

    # Unity Catalog paths
    st.markdown("**Unity Catalog Paths**")
    col1, col2, col3 = st.columns(3)

    with col1:
        catalog = st.text_input("Catalog", value="your_catalog", key="catalog")
    with col2:
        schema = st.text_input("Schema", value="your_schema", key="schema")
    with col3:
        volume = st.text_input("Volume", value="your_volume", key="volume")

    base_path = f"/Volumes/{catalog}/{schema}/{volume}"
    st.code(f"Base path: {base_path}", language="text")

    # Data paths
    col1, col2 = st.columns(2)

    with col1:
        train_data_path = st.text_input(
            "Training Data Path",
            value=f"{base_path}/data/train2017/",
            key="train_data_path",
            help="Path to training images"
        )
        train_annotation_file = st.text_input(
            "Training Annotations",
            value=f"{base_path}/data/instances_train2017.json",
            key="train_annotation_file",
            help="Path to training COCO annotations JSON"
        )

    with col2:
        val_data_path = st.text_input(
            "Validation Data Path",
            value=f"{base_path}/data/val2017/",
            key="val_data_path",
            help="Path to validation images"
        )
        val_annotation_file = st.text_input(
            "Validation Annotations",
            value=f"{base_path}/data/instances_val2017.json",
            key="val_annotation_file",
            help="Path to validation COCO annotations JSON"
        )

    # Optional test data
    with st.expander("Test Data (Optional)"):
        include_test = st.checkbox("Include test dataset", value=False, key="include_test")
        if include_test:
            test_data_path = st.text_input(
                "Test Data Path",
                value=f"{base_path}/data/test2017/",
                key="test_data_path"
            )
            test_annotation_file = st.text_input(
                "Test Annotations",
                value=f"{base_path}/data/instances_test2017.json",
                key="test_annotation_file"
            )
        else:
            test_data_path = None
            test_annotation_file = None

    # Data loading parameters
    st.markdown("**Data Loading Parameters**")
    col1, col2, col3 = st.columns(3)

    with col1:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=128,
            value=DETECTION_MODELS[model_id]["default_batch_size"],
            key="batch_size",
            help="Number of samples per batch"
        )

    with col2:
        num_workers = st.number_input(
            "Num Workers",
            min_value=0,
            max_value=32,
            value=4,
            key="num_workers",
            help="Number of data loading workers"
        )

    with col3:
        image_size = st.number_input(
            "Image Size",
            min_value=128,
            max_value=1024,
            value=DETECTION_MODELS[model_id]["default_image_size"],
            step=32,
            key="image_size",
            help="Input image size (square)"
        )

    # Image augmentation
    with st.expander("Image Augmentation (Optional)"):
        augment = st.checkbox("Enable augmentation", value=True, key="augment")

        if augment:
            col1, col2 = st.columns(2)
            with col1:
                horizontal_flip = st.checkbox("Horizontal Flip", value=True, key="h_flip")
                vertical_flip = st.checkbox("Vertical Flip", value=False, key="v_flip")
                rotation = st.slider("Rotation (degrees)", 0, 45, 10, key="rotation")

            with col2:
                brightness = st.slider("Brightness Jitter", 0.0, 0.5, 0.2, key="brightness")
                contrast = st.slider("Contrast Jitter", 0.0, 0.5, 0.2, key="contrast")

            augmentations = {
                "horizontal_flip": horizontal_flip,
                "vertical_flip": vertical_flip,
                "rotation": rotation,
                "brightness_contrast": brightness if brightness > 0 or contrast > 0 else None,
            }
        else:
            augmentations = None

    # Populate data config
    config["data"] = {
        "train_data_path": train_data_path,
        "train_annotation_file": train_annotation_file,
        "val_data_path": val_data_path,
        "val_annotation_file": val_annotation_file,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "model_name": model_id,
        "image_size": [image_size, image_size],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "augment": augment,
        "augmentations": augmentations,
    }

    if include_test:
        config["data"]["test_data_path"] = test_data_path
        config["data"]["test_annotation_file"] = test_annotation_file

    # ========== TRAINING CONFIGURATION ==========
    st.markdown('<div class="section-header">Training Configuration</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        max_epochs = st.number_input(
            "Max Epochs",
            min_value=1,
            max_value=1000,
            value=50,
            key="max_epochs",
            help="Maximum number of training epochs"
        )

    with col2:
        early_stopping_patience = st.number_input(
            "Early Stopping Patience",
            min_value=1,
            max_value=100,
            value=20,
            key="early_stopping_patience",
            help="Epochs to wait before early stopping"
        )

    with col3:
        log_every_n_steps = st.number_input(
            "Log Every N Steps",
            min_value=1,
            max_value=1000,
            value=50,
            key="log_every_n_steps",
            help="Logging frequency"
        )

    # Monitoring
    col1, col2 = st.columns(2)

    with col1:
        monitor_metric = st.selectbox(
            "Monitor Metric",
            options=["val_map", "val_loss", "train_loss"],
            index=0,
            key="monitor_metric",
            help="Metric to monitor for checkpointing"
        )

    with col2:
        monitor_mode = st.selectbox(
            "Monitor Mode",
            options=["max", "min"],
            index=0 if "map" in monitor_metric else 1,
            key="monitor_mode",
            help="Whether to maximize or minimize the metric"
        )

    # Checkpoint configuration
    st.markdown("**Checkpointing**")
    col1, col2 = st.columns(2)

    with col1:
        checkpoint_dir = st.text_input(
            "Checkpoint Directory",
            value=f"{base_path}/checkpoints/detection",
            key="checkpoint_dir",
            help="Directory to save model checkpoints"
        )

    with col2:
        save_top_k = st.number_input(
            "Save Top K Models",
            min_value=1,
            max_value=10,
            value=3,
            key="save_top_k",
            help="Number of best checkpoints to keep"
        )

    # Distributed training
    with st.expander("Distributed Training (Optional)"):
        distributed = st.checkbox("Enable distributed training", value=False, key="distributed")

        if distributed:
            col1, col2 = st.columns(2)
            with col1:
                use_ray = st.checkbox("Use Ray (multi-node)", value=False, key="use_ray")
                use_gpu = st.checkbox("Use GPU", value=True, key="use_gpu")

            with col2:
                num_workers_dist = st.number_input("Number of Workers", value=1, key="num_workers_dist")
                resources_cpu = st.number_input("CPU per Worker", value=4, key="cpu_per_worker")
                resources_gpu = st.number_input("GPU per Worker", value=1, key="gpu_per_worker")

            resources_per_worker = {"CPU": resources_cpu, "GPU": resources_gpu}
        else:
            use_ray = False
            use_gpu = True
            num_workers_dist = 1
            resources_per_worker = {"CPU": 4, "GPU": 1}

    # Populate training config
    config["training"] = {
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "scheduler_params": scheduler_params if scheduler != "none" else None,
        "early_stopping_patience": early_stopping_patience,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "checkpoint_dir": checkpoint_dir,
        "save_top_k": save_top_k,
        "log_every_n_steps": log_every_n_steps,
        "distributed": distributed,
        "use_ray": use_ray,
        "use_gpu": use_gpu,
        "num_workers": num_workers_dist,
        "resources_per_worker": resources_per_worker,
    }

    # ========== MLFLOW CONFIGURATION ==========
    st.markdown('<div class="section-header">MLflow Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        experiment_name = st.text_input(
            "Experiment Name",
            value="detection_training",
            key="experiment_name",
            help="MLflow experiment name"
        )

    with col2:
        run_name = st.text_input(
            "Run Name",
            value="detr-resnet50",
            key="run_name",
            help="MLflow run name"
        )

    log_model = st.checkbox("Log Model to MLflow", value=True, key="log_model")

    # Tags
    with st.expander("Tags (Optional)"):
        st.markdown("Add custom tags for organization and filtering")
        tag_framework = st.text_input("Framework", value="lightning", key="tag_framework")
        tag_model = st.text_input("Model", value=DETECTION_MODELS[model_id]["display_name"].lower().replace(" ", "-"), key="tag_model")
        tag_dataset = st.text_input("Dataset", value="coco", key="tag_dataset")

    # Populate MLflow config
    config["mlflow"] = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "log_model": log_model,
        "tags": {
            "framework": tag_framework,
            "model": tag_model,
            "dataset": tag_dataset,
        }
    }

    # ========== OUTPUT CONFIGURATION ==========
    st.markdown('<div class="section-header">Output Configuration</div>', unsafe_allow_html=True)

    results_dir = st.text_input(
        "Results Directory",
        value=f"{base_path}/results/detection",
        key="results_dir",
        help="Directory to save training results"
    )

    save_predictions = st.checkbox("Save Predictions", value=True, key="save_predictions")

    # Visualization options
    with st.expander("Visualization Options"):
        save_images = st.checkbox("Save Visualizations", value=True, key="save_images")
        vis_confidence = st.slider("Visualization Confidence Threshold", 0.0, 1.0, 0.5, key="vis_confidence")
        max_boxes = st.number_input("Max Boxes to Visualize", 1, 100, 20, key="max_boxes")
        max_images = st.number_input("Max Images to Save", 1, 100, 10, key="max_images")

    # Populate output config
    config["output"] = {
        "results_dir": results_dir,
        "save_predictions": save_predictions,
        "visualization": {
            "save_images": save_images,
            "confidence_threshold": vis_confidence,
            "max_boxes": max_boxes,
            "max_images": max_images,
        }
    }

    return config

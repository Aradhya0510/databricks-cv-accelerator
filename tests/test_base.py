#!/usr/bin/env python3
"""
Base test class for task module testing.
Provides common utilities and patterns for testing all CV task modules.
"""

import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Tuple


class BaseTaskTest(unittest.TestCase):
    """Base class for testing CV task modules."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_temp_dir)
        
        # Common test parameters
        self.batch_size = 2
        self.num_classes = 3
        self.image_size = 224
        self.num_workers = 0  # Avoid multiprocessing issues in tests
        
    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_dummy_image(self, size: Tuple[int, int] = (100, 100)) -> Image.Image:
        """Create a dummy PIL image for testing."""
        return Image.fromarray(
            np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        )
    
    def create_dummy_coco_dataset(self, num_classes: int = 3, images_per_class: int = 2) -> str:
        """Create a dummy COCO-format dataset for testing."""
        dataset_dir = Path(self.temp_dir) / "dummy_dataset"
        dataset_dir.mkdir()
        
        # Create class directories
        for class_idx in range(num_classes):
            class_dir = dataset_dir / f"class_{class_idx}"
            class_dir.mkdir()
            
            # Create images for this class
            for img_idx in range(images_per_class):
                img = self.create_dummy_image((50, 50))
                img_path = class_dir / f"image_{img_idx}.jpg"
                img.save(img_path)
        
        return str(dataset_dir)
    
    def create_dummy_coco_annotations(self, num_images: int = 4, num_classes: int = 3) -> Dict[str, Any]:
        """Create dummy COCO format annotations."""
        annotations = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Create categories
        for class_idx in range(num_classes):
            annotations["categories"].append({
                "id": class_idx + 1,
                "name": f"class_{class_idx}",
                "supercategory": "object"
            })
        
        # Create images and annotations
        annotation_id = 1
        for img_idx in range(num_images):
            # Add image
            annotations["images"].append({
                "id": img_idx,
                "file_name": f"image_{img_idx}.jpg",
                "width": 100,
                "height": 100
            })
            
            # Add annotation (one per image)
            annotations["annotations"].append({
                "id": annotation_id,
                "image_id": img_idx,
                "category_id": (img_idx % num_classes) + 1,
                "bbox": [10, 10, 20, 20],  # [x, y, width, height]
                "area": 400,
                "iscrowd": 0
            })
            annotation_id += 1
        
        return annotations
    
    def mock_transformers(self):
        """Mock transformers to avoid model loading."""
        return patch.multiple(
            'transformers',
            AutoImageProcessor=MagicMock(),
            AutoModelForObjectDetection=MagicMock(),
            AutoModelForImageClassification=MagicMock(),
            AutoModelForSemanticSegmentation=MagicMock(),
            AutoModelForInstanceSegmentation=MagicMock(),
            AutoConfig=MagicMock()
        )
    
    def assert_config_structure(self, config: Any, expected_fields: List[str]):
        """Assert that config has expected structure."""
        for field in expected_fields:
            self.assertTrue(hasattr(config, field), f"Config missing field: {field}")
    
    def assert_model_interface(self, model: Any):
        """Assert that model has required interface."""
        required_methods = ['forward', 'training_step', 'validation_step', 'configure_optimizers']
        for method in required_methods:
            self.assertTrue(hasattr(model, method), f"Model missing method: {method}")
            self.assertTrue(callable(getattr(model, method)), f"Model method not callable: {method}")
    
    def assert_dataset_interface(self, dataset: Any):
        """Assert that dataset has required interface."""
        self.assertTrue(hasattr(dataset, '__len__'))
        self.assertTrue(hasattr(dataset, '__getitem__'))
        self.assertTrue(callable(dataset.__len__))
        self.assertTrue(callable(dataset.__getitem__))
        
        # Test that we can get an item
        if len(dataset) > 0:
            item = dataset[0]
            self.assertIsInstance(item, dict)
    
    def assert_datamodule_interface(self, datamodule: Any):
        """Assert that datamodule has required interface."""
        required_methods = ['setup', 'train_dataloader', 'val_dataloader', 'test_dataloader']
        for method in required_methods:
            self.assertTrue(hasattr(datamodule, method), f"DataModule missing method: {method}")
            self.assertTrue(callable(getattr(datamodule, method)), f"DataModule method not callable: {method}")
    
    def assert_adapter_interface(self, adapter: Any, is_input_adapter: bool = True):
        """Assert that adapter has required interface."""
        if is_input_adapter:
            self.assertTrue(callable(adapter), "Input adapter should be callable")
        else:
            # Output adapter
            required_methods = ['adapt_output', 'format_predictions', 'format_targets']
            for method in required_methods:
                self.assertTrue(hasattr(adapter, method), f"Output adapter missing method: {method}")
                self.assertTrue(callable(getattr(adapter, method)), f"Output adapter method not callable: {method}")
    
    def create_minimal_config(self, task_type: str, **kwargs) -> Dict[str, Any]:
        """Create minimal config for testing."""
        base_config = {
            "model_name": "test/model",
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "image_size": self.image_size,
            "learning_rate": 1e-4,
            "max_epochs": 1,
            "pretrained": False
        }
        
        # Add task-specific config
        if task_type == "classification":
            base_config.update({
                "train_data_path": self.temp_dir,
                "val_data_path": self.temp_dir,
                "test_data_path": None
            })
        elif task_type in ["detection", "semantic_segmentation", "instance_segmentation", "panoptic_segmentation"]:
            base_config.update({
                "train_data_path": self.temp_dir,
                "train_annotation_file": f"{self.temp_dir}/annotations.json",
                "val_data_path": self.temp_dir,
                "val_annotation_file": f"{self.temp_dir}/annotations.json",
                "test_data_path": None,
                "test_annotation_file": None
            })
        
        base_config.update(kwargs)
        return base_config

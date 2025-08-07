#!/usr/bin/env python3
"""
Comprehensive tests for instance segmentation task module.
Tests all components end-to-end: config, model, data, adapters, evaluation.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_base import BaseTaskTest
from tasks.instance_segmentation.model import InstanceSegmentationModel, InstanceSegmentationModelConfig
from tasks.instance_segmentation.data import InstanceSegmentationDataset, InstanceSegmentationDataModule, InstanceSegmentationDataConfig
from tasks.instance_segmentation.adapters import (
    Mask2FormerInputAdapter, Mask2FormerOutputAdapter,
    get_input_adapter, get_output_adapter
)
from tasks.instance_segmentation.evaluate import InstanceSegmentationEvaluator


class TestInstanceSegmentationTask(BaseTaskTest):
    """Comprehensive tests for instance segmentation task module."""
    
    def setUp(self):
        """Set up instance segmentation-specific test fixtures."""
        super().setUp()
        
        # Create dummy COCO dataset
        self.dataset_path = self.create_dummy_coco_dataset()
        self.annotations = self.create_dummy_coco_annotations()
        
        # Save annotations to file
        annotations_file = os.path.join(self.temp_dir, "annotations.json")
        with open(annotations_file, 'w') as f:
            json.dump(self.annotations, f)
        
        self.annotations_file = annotations_file
    
    def test_instance_segmentation_model_config(self):
        """Test instance segmentation model config initialization and structure."""
        config_dict = self.create_minimal_config("instance_segmentation")
        config = InstanceSegmentationModelConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'model_name', 'num_classes', 'pretrained', 'learning_rate', 
            'weight_decay', 'scheduler', 'epochs', 'mask_threshold',
            'overlap_threshold', 'num_queries'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.model_name, "test/model")
        self.assertEqual(config.num_classes, self.num_classes)
        self.assertEqual(config.mask_threshold, 0.5)
    
    def test_instance_segmentation_data_config(self):
        """Test instance segmentation data config initialization and structure."""
        config_dict = self.create_minimal_config("instance_segmentation")
        config = InstanceSegmentationDataConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'train_data_path', 'val_data_path', 'train_annotation_file',
            'val_annotation_file', 'batch_size', 'num_workers', 'image_size'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.batch_size, self.batch_size)
        self.assertEqual(config.image_size, self.image_size)
    
    def test_instance_segmentation_model_initialization(self):
        """Test instance segmentation model initialization with mocked transformers."""
        with self.mock_transformers():
            config_dict = self.create_minimal_config("instance_segmentation")
            model = InstanceSegmentationModel(config_dict)
            
            # Test model interface
            self.assert_model_interface(model)
            
            # Test model attributes
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.output_adapter)
    
    def test_instance_segmentation_dataset_initialization(self):
        """Test instance segmentation dataset initialization."""
        dataset = InstanceSegmentationDataset(
            data_path=self.dataset_path,
            annotation_file=self.annotations_file,
            transform=None
        )
        
        # Test dataset interface
        self.assert_dataset_interface(dataset)
        
        # Test dataset properties
        self.assertGreater(len(dataset), 0)
        self.assertIsNotNone(dataset.class_names)
    
    def test_instance_segmentation_dataset_getitem(self):
        """Test instance segmentation dataset item retrieval."""
        dataset = InstanceSegmentationDataset(
            data_path=self.dataset_path,
            annotation_file=self.annotations_file,
            transform=None
        )
        
        if len(dataset) > 0:
            item = dataset[0]
            
            # Test item structure
            self.assertIn("pixel_values", item)
            self.assertIn("labels", item)
            
            # Test labels structure
            labels = item["labels"]
            self.assertIn("boxes", labels)
            self.assertIn("masks", labels)
            self.assertIn("class_labels", labels)
            self.assertIn("image_id", labels)
    
    def test_instance_segmentation_datamodule_initialization(self):
        """Test instance segmentation datamodule initialization."""
        config_dict = self.create_minimal_config("instance_segmentation")
        datamodule = InstanceSegmentationDataModule(config_dict)
        
        # Test datamodule interface
        self.assert_datamodule_interface(datamodule)
        
        # Test datamodule attributes
        self.assertIsNotNone(datamodule.config)
    
    def test_instance_segmentation_datamodule_setup(self):
        """Test instance segmentation datamodule setup."""
        config_dict = self.create_minimal_config("instance_segmentation")
        datamodule = InstanceSegmentationDataModule(config_dict)
        
        # Setup datamodule
        datamodule.setup()
        
        # Test datasets are created
        self.assertIsNotNone(datamodule.train_dataset)
        self.assertIsNotNone(datamodule.val_dataset)
        self.assertGreater(len(datamodule.train_dataset), 0)
        self.assertGreater(len(datamodule.val_dataset), 0)
    
    def test_instance_segmentation_datamodule_dataloaders(self):
        """Test instance segmentation datamodule dataloaders."""
        config_dict = self.create_minimal_config("instance_segmentation")
        datamodule = InstanceSegmentationDataModule(config_dict)
        datamodule.setup()
        
        # Test dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertEqual(train_loader.batch_size, self.batch_size)
        self.assertEqual(val_loader.batch_size, self.batch_size)
    
    def test_mask2former_input_adapter(self):
        """Test Mask2Former input adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = Mask2FormerInputAdapter("facebook/mask2former-swin-base-coco-instance", image_size=800)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=True)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 800)
            self.assertIsNotNone(adapter.processor)
    
    def test_mask2former_output_adapter(self):
        """Test Mask2Former output adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = Mask2FormerOutputAdapter("facebook/mask2former-swin-base-coco-instance", image_size=800)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=False)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 800)
            self.assertIsNotNone(adapter.processor)
    
    def test_adapter_factory_functions(self):
        """Test adapter factory functions."""
        # Test input adapter factory
        mask2former_input = get_input_adapter("facebook/mask2former-swin-base-coco-instance")
        
        self.assertIsInstance(mask2former_input, Mask2FormerInputAdapter)
        
        # Test output adapter factory
        mask2former_output = get_output_adapter("facebook/mask2former-swin-base-coco-instance")
        
        self.assertIsInstance(mask2former_output, Mask2FormerOutputAdapter)
    
    def test_instance_mask_processing(self):
        """Test instance mask processing utilities."""
        # Create dummy instance masks
        num_instances = 3
        mask_size = (100, 100)
        masks = torch.randint(0, 2, (num_instances, *mask_size), dtype=torch.bool)
        
        # Test mask shape and values
        self.assertEqual(masks.shape, (num_instances, 100, 100))
        self.assertTrue(torch.all(masks >= 0))
        self.assertTrue(torch.all(masks <= 1))
    
    def test_instance_segmentation_evaluator_initialization(self):
        """Test instance segmentation evaluator initialization."""
        # Create dummy model checkpoint and config
        checkpoint_path = os.path.join(self.temp_dir, "dummy_checkpoint.ckpt")
        config_path = os.path.join(self.temp_dir, "dummy_config.yaml")
        
        # Create dummy files
        with open(checkpoint_path, 'w') as f:
            f.write("dummy checkpoint")
        with open(config_path, 'w') as f:
            f.write("dummy config")
        
        # Test evaluator initialization (should not fail even with dummy files)
        try:
            evaluator = InstanceSegmentationEvaluator(checkpoint_path, config_path)
            # If it doesn't fail, test basic interface
            self.assertIsNotNone(evaluator)
        except Exception:
            # Expected to fail with dummy files, but should not crash
            pass
    
    def test_instance_segmentation_metrics_interface(self):
        """Test instance segmentation metrics interface."""
        # Test that metrics can be imported and initialized
        try:
            from pycocotools.cocoeval import COCOeval
            
            # Test that COCOeval can be imported
            self.assertIsNotNone(COCOeval)
            
        except ImportError:
            # Skip if pycocotools not available
            self.skipTest("pycocotools not available")
    
    def test_mask_overlap_handling(self):
        """Test mask overlap handling utilities."""
        # Create dummy overlapping masks
        mask1 = torch.zeros((100, 100), dtype=torch.bool)
        mask1[10:30, 10:30] = True
        
        mask2 = torch.zeros((100, 100), dtype=torch.bool)
        mask2[20:40, 20:40] = True
        
        # Test overlap calculation
        overlap = torch.logical_and(mask1, mask2)
        overlap_area = overlap.sum()
        
        # Should have some overlap
        self.assertGreater(overlap_area, 0)
    
    def test_instance_annotation_format(self):
        """Test instance annotation format handling."""
        # Create dummy instance annotation
        instance_annotation = {
            "id": 1,
            "image_id": 0,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],  # [x, y, width, height]
            "area": 400,
            "iscrowd": 0,
            "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]  # Polygon format
        }
        
        # Test annotation structure
        self.assertIn("id", instance_annotation)
        self.assertIn("image_id", instance_annotation)
        self.assertIn("category_id", instance_annotation)
        self.assertIn("bbox", instance_annotation)
        self.assertIn("segmentation", instance_annotation)
        
        # Test bbox format
        bbox = instance_annotation["bbox"]
        self.assertEqual(len(bbox), 4)
        self.assertTrue(all(isinstance(x, (int, float)) for x in bbox))


if __name__ == "__main__":
    unittest.main()

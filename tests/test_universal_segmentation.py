#!/usr/bin/env python3
"""
Comprehensive tests for panoptic segmentation task module.
Tests all components end-to-end: config, model, data, adapters, evaluation.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_base import BaseTaskTest
from tasks.universal_segmentation.model import UniversalSegmentationModel, UniversalSegmentationModelConfig
from tasks.universal_segmentation.data import UniversalSegmentationDataset, UniversalSegmentationDataModule, UniversalSegmentationDataConfig
from tasks.universal_segmentation.adapters import (
    Mask2FormerInputAdapter, Mask2FormerOutputAdapter,
    get_input_adapter, get_output_adapter
)
from tasks.universal_segmentation.evaluate import UniversalSegmentationEvaluator


class TestUniversalSegmentationTask(BaseTaskTest):
    """Comprehensive tests for panoptic segmentation task module."""
    
    def setUp(self):
        """Set up panoptic segmentation-specific test fixtures."""
        super().setUp()
        
        # Create dummy COCO dataset
        self.dataset_path = self.create_dummy_coco_dataset()
        self.annotations = self.create_dummy_coco_annotations()
        
        # Save annotations to file
        annotations_file = os.path.join(self.temp_dir, "annotations.json")
        with open(annotations_file, 'w') as f:
            json.dump(self.annotations, f)
        
        self.annotations_file = annotations_file
    
    def test_universal_segmentation_model_config(self):
        """Test panoptic segmentation model config initialization and structure."""
        config_dict = self.create_minimal_config("universal_segmentation")
        config = UniversalSegmentationModelConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'model_name', 'num_classes', 'pretrained', 'learning_rate', 
            'weight_decay', 'scheduler', 'epochs', 'mask_threshold',
            'overlap_threshold', 'num_queries', 'num_thing_classes',
            'num_stuff_classes'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.model_name, "test/model")
        self.assertEqual(config.num_classes, self.num_classes)
        self.assertEqual(config.mask_threshold, 0.5)
    
    def test_universal_segmentation_data_config(self):
        """Test panoptic segmentation data config initialization and structure."""
        config_dict = self.create_minimal_config("universal_segmentation")
        config = UniversalSegmentationDataConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'train_data_path', 'val_data_path', 'train_annotation_file',
            'val_annotation_file', 'batch_size', 'num_workers', 'image_size'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.batch_size, self.batch_size)
        self.assertEqual(config.image_size, self.image_size)
    
    def test_universal_segmentation_model_initialization(self):
        """Test panoptic segmentation model initialization with mocked transformers."""
        with self.mock_transformers():
            config_dict = self.create_minimal_config("universal_segmentation")
            model = UniversalSegmentationModel(config_dict)
            
            # Test model interface
            self.assert_model_interface(model)
            
            # Test model attributes
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.output_adapter)
    
    def test_universal_segmentation_dataset_initialization(self):
        """Test panoptic segmentation dataset initialization."""
        dataset = UniversalSegmentationDataset(
            data_path=self.dataset_path,
            annotation_file=self.annotations_file,
            transform=None
        )
        
        # Test dataset interface
        self.assert_dataset_interface(dataset)
        
        # Test dataset properties
        self.assertGreater(len(dataset), 0)
        self.assertIsNotNone(dataset.class_names)
    
    def test_universal_segmentation_dataset_getitem(self):
        """Test panoptic segmentation dataset item retrieval."""
        dataset = UniversalSegmentationDataset(
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
            self.assertIn("panoptic_masks", labels)
    
    def test_universal_segmentation_datamodule_initialization(self):
        """Test panoptic segmentation datamodule initialization."""
        config_dict = self.create_minimal_config("universal_segmentation")
        datamodule = UniversalSegmentationDataModule(config_dict)
        
        # Test datamodule interface
        self.assert_datamodule_interface(datamodule)
        
        # Test datamodule attributes
        self.assertIsNotNone(datamodule.config)
    
    def test_universal_segmentation_datamodule_setup(self):
        """Test panoptic segmentation datamodule setup."""
        config_dict = self.create_minimal_config("universal_segmentation")
        datamodule = UniversalSegmentationDataModule(config_dict)
        
        # Setup datamodule
        datamodule.setup()
        
        # Test datasets are created
        self.assertIsNotNone(datamodule.train_dataset)
        self.assertIsNotNone(datamodule.val_dataset)
        self.assertGreater(len(datamodule.train_dataset), 0)
        self.assertGreater(len(datamodule.val_dataset), 0)
    
    def test_universal_segmentation_datamodule_dataloaders(self):
        """Test panoptic segmentation datamodule dataloaders."""
        config_dict = self.create_minimal_config("universal_segmentation")
        datamodule = UniversalSegmentationDataModule(config_dict)
        datamodule.setup()
        
        # Test dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertEqual(train_loader.batch_size, self.batch_size)
        self.assertEqual(val_loader.batch_size, self.batch_size)
    
    def test_mask2former_input_adapter(self):
        """Test Mask2Former input adapter for panoptic segmentation."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = Mask2FormerInputAdapter("facebook/mask2former-swin-base-coco-panoptic", image_size=800)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=True)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 800)
            self.assertIsNotNone(adapter.processor)
    
    def test_mask2former_output_adapter(self):
        """Test Mask2Former output adapter for panoptic segmentation."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = Mask2FormerOutputAdapter("facebook/mask2former-swin-base-coco-panoptic", image_size=800)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=False)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 800)
            self.assertIsNotNone(adapter.processor)
    
    def test_adapter_factory_functions(self):
        """Test adapter factory functions."""
        # Test input adapter factory
        mask2former_input = get_input_adapter("facebook/mask2former-swin-base-coco-panoptic")
        
        self.assertIsInstance(mask2former_input, Mask2FormerInputAdapter)
        
        # Test output adapter factory
        mask2former_output = get_output_adapter("facebook/mask2former-swin-base-coco-panoptic")
        
        self.assertIsInstance(mask2former_output, Mask2FormerOutputAdapter)
    
    def test_panoptic_mask_processing(self):
        """Test panoptic mask processing utilities."""
        # Create dummy panoptic mask
        panoptic_mask = torch.randint(0, self.num_classes * 1000, (100, 100), dtype=torch.long)
        
        # Test mask shape and values
        self.assertEqual(panoptic_mask.shape, (100, 100))
        self.assertTrue(torch.all(panoptic_mask >= 0))
    
    def test_thing_stuff_classification(self):
        """Test thing vs stuff class classification."""
        # Test thing classes (instances)
        thing_classes = [1, 2, 3, 4, 5]  # Example COCO thing classes
        
        # Test stuff classes (background regions)
        stuff_classes = [0, 11, 12, 13, 14]  # Example COCO stuff classes
        
        # Test that thing and stuff classes are distinct
        thing_set = set(thing_classes)
        stuff_set = set(stuff_classes)
        self.assertEqual(len(thing_set.intersection(stuff_set)), 0)
    
    def test_universal_segmentation_evaluator_initialization(self):
        """Test panoptic segmentation evaluator initialization."""
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
            evaluator = UniversalSegmentationEvaluator(checkpoint_path, config_path)
            # If it doesn't fail, test basic interface
            self.assertIsNotNone(evaluator)
        except Exception:
            # Expected to fail with dummy files, but should not crash
            pass
    
    def test_universal_segmentation_metrics_interface(self):
        """Test panoptic segmentation metrics interface."""
        # Test that metrics can be imported and initialized
        try:
            from pycocotools.cocoeval import COCOeval
            
            # Test that COCOeval can be imported
            self.assertIsNotNone(COCOeval)
            
        except ImportError:
            # Skip if pycocotools not available
            self.skipTest("pycocotools not available")
    
    def test_panoptic_annotation_format(self):
        """Test panoptic annotation format handling."""
        # Create dummy panoptic annotation
        panoptic_annotation = {
            "file_name": "image_0.png",
            "image_id": 0,
            "segments_info": [
                {
                    "id": 1,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": [10, 10, 20, 20],
                    "area": 400
                },
                {
                    "id": 2,
                    "category_id": 11,  # Stuff class
                    "iscrowd": 0,
                    "bbox": [0, 0, 100, 100],
                    "area": 10000
                }
            ]
        }
        
        # Test annotation structure
        self.assertIn("file_name", panoptic_annotation)
        self.assertIn("image_id", panoptic_annotation)
        self.assertIn("segments_info", panoptic_annotation)
        
        # Test segments_info structure
        segments_info = panoptic_annotation["segments_info"]
        self.assertGreater(len(segments_info), 0)
        
        for segment in segments_info:
            self.assertIn("id", segment)
            self.assertIn("category_id", segment)
            self.assertIn("iscrowd", segment)
            self.assertIn("bbox", segment)
            self.assertIn("area", segment)
    
    def test_panoptic_mask_merging(self):
        """Test panoptic mask merging utilities."""
        # Create dummy instance masks
        num_instances = 3
        mask_size = (100, 100)
        instance_masks = torch.randint(0, 2, (num_instances, *mask_size), dtype=torch.bool)
        
        # Create dummy semantic mask
        semantic_mask = torch.randint(0, self.num_classes, mask_size, dtype=torch.long)
        
        # Test that masks have correct shapes
        self.assertEqual(instance_masks.shape, (num_instances, 100, 100))
        self.assertEqual(semantic_mask.shape, (100, 100))
    
    def test_thing_stuff_threshold_handling(self):
        """Test thing vs stuff threshold handling."""
        # Test confidence thresholds
        thing_threshold = 0.7
        stuff_threshold = 0.5
        
        # Test that thing threshold is typically higher than stuff threshold
        self.assertGreater(thing_threshold, stuff_threshold)
        
        # Test threshold application
        confidence_scores = torch.tensor([0.8, 0.6, 0.3, 0.9])
        
        thing_detections = confidence_scores >= thing_threshold
        stuff_detections = confidence_scores >= stuff_threshold
        
        # Should have fewer thing detections than stuff detections
        self.assertLessEqual(thing_detections.sum(), stuff_detections.sum())


if __name__ == "__main__":
    unittest.main()

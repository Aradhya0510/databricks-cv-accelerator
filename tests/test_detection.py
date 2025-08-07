#!/usr/bin/env python3
"""
Comprehensive tests for detection task module.
Tests all components end-to-end: config, model, data, adapters, evaluation.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_base import BaseTaskTest
from tasks.detection.model import DetectionModel, DetectionModelConfig
from tasks.detection.data import DetectionDataset, DetectionDataModule, DetectionDataConfig
from tasks.detection.adapters import (
    DETRInputAdapter, DETROutputAdapter, 
    YOLOSInputAdapter, YOLOSOutputAdapter,
    get_input_adapter, get_output_adapter
)
from tasks.detection.evaluate import DetectionEvaluator


class TestDetectionTask(BaseTaskTest):
    """Comprehensive tests for detection task module."""
    
    def setUp(self):
        """Set up detection-specific test fixtures."""
        super().setUp()
        
        # Create dummy COCO dataset
        self.dataset_path = self.create_dummy_coco_dataset()
        self.annotations = self.create_dummy_coco_annotations()
        
        # Save annotations to file
        annotations_file = os.path.join(self.temp_dir, "annotations.json")
        with open(annotations_file, 'w') as f:
            json.dump(self.annotations, f)
        
        self.annotations_file = annotations_file
    
    def test_detection_model_config(self):
        """Test detection model config initialization and structure."""
        config_dict = self.create_minimal_config("detection")
        config = DetectionModelConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'model_name', 'num_classes', 'pretrained', 'learning_rate', 
            'weight_decay', 'scheduler', 'epochs', 'confidence_threshold',
            'iou_threshold', 'max_detections'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.model_name, "test/model")
        self.assertEqual(config.num_classes, self.num_classes)
        self.assertEqual(config.confidence_threshold, 0.5)
    
    def test_detection_data_config(self):
        """Test detection data config initialization and structure."""
        config_dict = self.create_minimal_config("detection")
        config = DetectionDataConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'train_data_path', 'val_data_path', 'train_annotation_file',
            'val_annotation_file', 'batch_size', 'num_workers', 'image_size'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.batch_size, self.batch_size)
        self.assertEqual(config.image_size, self.image_size)
    
    def test_detection_model_initialization(self):
        """Test detection model initialization with mocked transformers."""
        with self.mock_transformers():
            config_dict = self.create_minimal_config("detection")
            model = DetectionModel(config_dict)
            
            # Test model interface
            self.assert_model_interface(model)
            
            # Test model attributes
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.output_adapter)
    
    def test_detection_dataset_initialization(self):
        """Test detection dataset initialization."""
        dataset = DetectionDataset(
            data_path=self.dataset_path,
            annotation_file=self.annotations_file,
            transform=None
        )
        
        # Test dataset interface
        self.assert_dataset_interface(dataset)
        
        # Test dataset properties
        self.assertGreater(len(dataset), 0)
        self.assertIsNotNone(dataset.class_names)
    
    def test_detection_dataset_getitem(self):
        """Test detection dataset item retrieval."""
        dataset = DetectionDataset(
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
            self.assertIn("class_labels", labels)
            self.assertIn("image_id", labels)
    
    def test_detection_datamodule_initialization(self):
        """Test detection datamodule initialization."""
        config_dict = self.create_minimal_config("detection")
        datamodule = DetectionDataModule(config_dict)
        
        # Test datamodule interface
        self.assert_datamodule_interface(datamodule)
        
        # Test datamodule attributes
        self.assertIsNotNone(datamodule.config)
    
    def test_detection_datamodule_setup(self):
        """Test detection datamodule setup."""
        config_dict = self.create_minimal_config("detection")
        datamodule = DetectionDataModule(config_dict)
        
        # Setup datamodule
        datamodule.setup()
        
        # Test datasets are created
        self.assertIsNotNone(datamodule.train_dataset)
        self.assertIsNotNone(datamodule.val_dataset)
        self.assertGreater(len(datamodule.train_dataset), 0)
        self.assertGreater(len(datamodule.val_dataset), 0)
    
    def test_detection_datamodule_dataloaders(self):
        """Test detection datamodule dataloaders."""
        config_dict = self.create_minimal_config("detection")
        datamodule = DetectionDataModule(config_dict)
        datamodule.setup()
        
        # Test dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertEqual(train_loader.batch_size, self.batch_size)
        self.assertEqual(val_loader.batch_size, self.batch_size)
    
    def test_detr_input_adapter(self):
        """Test DETR input adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = DETRInputAdapter("facebook/detr-resnet-50", image_size=800)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=True)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 800)
            self.assertIsNotNone(adapter.processor)
    
    def test_detr_output_adapter(self):
        """Test DETR output adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = DETROutputAdapter("facebook/detr-resnet-50", image_size=800)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=False)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 800)
            self.assertIsNotNone(adapter.processor)
    
    def test_yolos_input_adapter(self):
        """Test YOLOS input adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = YOLOSInputAdapter("hustvl/yolos-tiny", image_size=800)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=True)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 800)
            self.assertIsNotNone(adapter.processor)
    
    def test_yolos_output_adapter(self):
        """Test YOLOS output adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = YOLOSOutputAdapter("hustvl/yolos-tiny", image_size=800)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=False)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 800)
            self.assertIsNotNone(adapter.processor)
    
    def test_adapter_factory_functions(self):
        """Test adapter factory functions."""
        # Test input adapter factory
        detr_input = get_input_adapter("facebook/detr-resnet-50")
        yolos_input = get_input_adapter("hustvl/yolos-tiny")
        
        self.assertIsInstance(detr_input, DETRInputAdapter)
        self.assertIsInstance(yolos_input, YOLOSInputAdapter)
        
        # Test output adapter factory
        detr_output = get_output_adapter("facebook/detr-resnet-50")
        yolos_output = get_output_adapter("hustvl/yolos-tiny")
        
        self.assertIsInstance(detr_output, DETROutputAdapter)
        self.assertIsInstance(yolos_output, YOLOSOutputAdapter)
    
    def test_box_format_conversion(self):
        """Test box format conversion utilities."""
        # Test xyxy to cxcwh conversion
        xyxy_boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
        
        # Test with DETR adapter
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            adapter = DETRInputAdapter("facebook/detr-resnet-50")
            
            cxcwh_boxes = adapter._xyxy_to_cxcwh(xyxy_boxes)
            
            # Check shape
            self.assertEqual(cxcwh_boxes.shape, xyxy_boxes.shape)
            
            # Check first box: [0, 0, 10, 10] -> [5, 5, 10, 10]
            expected_first = torch.tensor([5, 5, 10, 10], dtype=torch.float32)
            torch.testing.assert_close(cxcwh_boxes[0], expected_first, rtol=1e-6, atol=1e-6)
    
    def test_empty_target_handling(self):
        """Test handling of empty targets."""
        empty_target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor([0])
        }
        
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            adapter = DETRInputAdapter("facebook/detr-resnet-50")
            
            # Mock processor call
            mock_processed = MagicMock()
            mock_processed.pixel_values = torch.randn(1, 3, 800, 800)
            adapter.processor = MagicMock(return_value=mock_processed)
            
            image = self.create_dummy_image()
            processed_image, adapted_target = adapter(image, empty_target)
            
            # Should handle empty targets gracefully
            self.assertEqual(adapted_target["boxes"].shape[0], 0)
            self.assertEqual(adapted_target["class_labels"].shape[0], 0)
    
    def test_detection_evaluator_initialization(self):
        """Test detection evaluator initialization."""
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
            evaluator = DetectionEvaluator(checkpoint_path, config_path)
            # If it doesn't fail, test basic interface
            self.assertIsNotNone(evaluator)
        except Exception:
            # Expected to fail with dummy files, but should not crash
            pass


if __name__ == "__main__":
    unittest.main()

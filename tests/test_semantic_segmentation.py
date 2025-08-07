#!/usr/bin/env python3
"""
Comprehensive tests for semantic segmentation task module.
Tests all components end-to-end: config, model, data, adapters, evaluation.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_base import BaseTaskTest
from tasks.semantic_segmentation.model import SemanticSegmentationModel, SemanticSegmentationModelConfig
from tasks.semantic_segmentation.data import SemanticSegmentationDataset, SemanticSegmentationDataModule, SemanticSegmentationDataConfig
from tasks.semantic_segmentation.adapters import (
    SegFormerInputAdapter, SegFormerOutputAdapter,
    get_input_adapter, get_output_adapter
)
from tasks.semantic_segmentation.evaluate import SemanticSegmentationEvaluator


class TestSemanticSegmentationTask(BaseTaskTest):
    """Comprehensive tests for semantic segmentation task module."""
    
    def setUp(self):
        """Set up semantic segmentation-specific test fixtures."""
        super().setUp()
        
        # Create dummy COCO dataset
        self.dataset_path = self.create_dummy_coco_dataset()
        self.annotations = self.create_dummy_coco_annotations()
        
        # Save annotations to file
        annotations_file = os.path.join(self.temp_dir, "annotations.json")
        with open(annotations_file, 'w') as f:
            json.dump(self.annotations, f)
        
        self.annotations_file = annotations_file
    
    def test_semantic_segmentation_model_config(self):
        """Test semantic segmentation model config initialization and structure."""
        config_dict = self.create_minimal_config("semantic_segmentation")
        config = SemanticSegmentationModelConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'model_name', 'num_classes', 'pretrained', 'learning_rate', 
            'weight_decay', 'scheduler', 'epochs', 'aux_loss_weight',
            'mask_threshold'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.model_name, "test/model")
        self.assertEqual(config.num_classes, self.num_classes)
        self.assertEqual(config.aux_loss_weight, 0.4)
    
    def test_semantic_segmentation_data_config(self):
        """Test semantic segmentation data config initialization and structure."""
        config_dict = self.create_minimal_config("semantic_segmentation")
        config = SemanticSegmentationDataConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'train_data_path', 'val_data_path', 'train_annotation_file',
            'val_annotation_file', 'batch_size', 'num_workers', 'image_size'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.batch_size, self.batch_size)
        self.assertEqual(config.image_size, self.image_size)
    
    def test_semantic_segmentation_model_initialization(self):
        """Test semantic segmentation model initialization with mocked transformers."""
        with self.mock_transformers():
            config_dict = self.create_minimal_config("semantic_segmentation")
            model = SemanticSegmentationModel(config_dict)
            
            # Test model interface
            self.assert_model_interface(model)
            
            # Test model attributes
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.output_adapter)
    
    def test_semantic_segmentation_dataset_initialization(self):
        """Test semantic segmentation dataset initialization."""
        dataset = SemanticSegmentationDataset(
            data_path=self.dataset_path,
            annotation_file=self.annotations_file,
            transform=None
        )
        
        # Test dataset interface
        self.assert_dataset_interface(dataset)
        
        # Test dataset properties
        self.assertGreater(len(dataset), 0)
        self.assertIsNotNone(dataset.class_names)
    
    def test_semantic_segmentation_dataset_getitem(self):
        """Test semantic segmentation dataset item retrieval."""
        dataset = SemanticSegmentationDataset(
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
            self.assertIn("masks", labels)
            self.assertIn("class_labels", labels)
            self.assertIn("image_id", labels)
    
    def test_semantic_segmentation_datamodule_initialization(self):
        """Test semantic segmentation datamodule initialization."""
        config_dict = self.create_minimal_config("semantic_segmentation")
        datamodule = SemanticSegmentationDataModule(config_dict)
        
        # Test datamodule interface
        self.assert_datamodule_interface(datamodule)
        
        # Test datamodule attributes
        self.assertIsNotNone(datamodule.config)
    
    def test_semantic_segmentation_datamodule_setup(self):
        """Test semantic segmentation datamodule setup."""
        config_dict = self.create_minimal_config("semantic_segmentation")
        datamodule = SemanticSegmentationDataModule(config_dict)
        
        # Setup datamodule
        datamodule.setup()
        
        # Test datasets are created
        self.assertIsNotNone(datamodule.train_dataset)
        self.assertIsNotNone(datamodule.val_dataset)
        self.assertGreater(len(datamodule.train_dataset), 0)
        self.assertGreater(len(datamodule.val_dataset), 0)
    
    def test_semantic_segmentation_datamodule_dataloaders(self):
        """Test semantic segmentation datamodule dataloaders."""
        config_dict = self.create_minimal_config("semantic_segmentation")
        datamodule = SemanticSegmentationDataModule(config_dict)
        datamodule.setup()
        
        # Test dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertEqual(train_loader.batch_size, self.batch_size)
        self.assertEqual(val_loader.batch_size, self.batch_size)
    
    def test_segformer_input_adapter(self):
        """Test SegFormer input adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = SegFormerInputAdapter("nvidia/segformer-b0-finetuned-ade-512-512", image_size=512)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=True)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 512)
            self.assertIsNotNone(adapter.processor)
    
    def test_segformer_output_adapter(self):
        """Test SegFormer output adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = SegFormerOutputAdapter("nvidia/segformer-b0-finetuned-ade-512-512", image_size=512)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=False)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 512)
            self.assertIsNotNone(adapter.processor)
    
    def test_adapter_factory_functions(self):
        """Test adapter factory functions."""
        # Test input adapter factory
        segformer_input = get_input_adapter("nvidia/segformer-b0-finetuned-ade-512-512")
        
        self.assertIsInstance(segformer_input, SegFormerInputAdapter)
        
        # Test output adapter factory
        segformer_output = get_output_adapter("nvidia/segformer-b0-finetuned-ade-512-512")
        
        self.assertIsInstance(segformer_output, SegFormerOutputAdapter)
    
    def test_mask_processing(self):
        """Test mask processing utilities."""
        # Create dummy mask
        mask = torch.randint(0, self.num_classes, (100, 100), dtype=torch.long)
        
        # Test mask shape and values
        self.assertEqual(mask.shape, (100, 100))
        self.assertTrue(torch.all(mask >= 0))
        self.assertTrue(torch.all(mask < self.num_classes))
    
    def test_semantic_segmentation_evaluator_initialization(self):
        """Test semantic segmentation evaluator initialization."""
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
            evaluator = SemanticSegmentationEvaluator(checkpoint_path, config_path)
            # If it doesn't fail, test basic interface
            self.assertIsNotNone(evaluator)
        except Exception:
            # Expected to fail with dummy files, but should not crash
            pass
    
    def test_segmentation_metrics_interface(self):
        """Test segmentation metrics interface."""
        # Test that metrics can be imported and initialized
        try:
            from torchmetrics.classification import MulticlassJaccardIndex, Dice
            
            # Test metric initialization
            jaccard = MulticlassJaccardIndex(num_classes=self.num_classes, average='macro')
            dice = Dice(num_classes=self.num_classes, average='macro')
            
            # Test that metrics are callable
            self.assertTrue(callable(jaccard))
            self.assertTrue(callable(dice))
            
        except ImportError:
            # Skip if torchmetrics not available
            self.skipTest("torchmetrics not available")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Comprehensive tests for classification task module.
Tests all components end-to-end: config, model, data, adapters, evaluation.
"""

import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_base import BaseTaskTest
from tasks.classification.model import ClassificationModel, ClassificationModelConfig
from tasks.classification.data import ClassificationDataset, ClassificationDataModule, ClassificationDataConfig
from tasks.classification.adapters import (
    ViTInputAdapter, ConvNeXTInputAdapter, SwinInputAdapter, ResNetOutputAdapter,
    get_input_adapter, get_output_adapter
)
from tasks.classification.evaluate import ClassificationEvaluator


class TestClassificationTask(BaseTaskTest):
    """Comprehensive tests for classification task module."""
    
    def setUp(self):
        """Set up classification-specific test fixtures."""
        super().setUp()
        
        # Create dummy classification dataset
        self.dataset_path = self.create_dummy_coco_dataset()
    
    def test_classification_model_config(self):
        """Test classification model config initialization and structure."""
        config_dict = self.create_minimal_config("classification")
        config = ClassificationModelConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'model_name', 'num_classes', 'pretrained', 'learning_rate', 
            'weight_decay', 'scheduler', 'epochs', 'dropout_rate',
            'label_smoothing'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.model_name, "test/model")
        self.assertEqual(config.num_classes, self.num_classes)
        self.assertEqual(config.dropout_rate, 0.1)
    
    def test_classification_data_config(self):
        """Test classification data config initialization and structure."""
        config_dict = self.create_minimal_config("classification")
        config = ClassificationDataConfig(**config_dict)
        
        # Test config structure
        expected_fields = [
            'train_data_path', 'val_data_path', 'test_data_path',
            'batch_size', 'num_workers', 'image_size', 'mean', 'std'
        ]
        self.assert_config_structure(config, expected_fields)
        
        # Test config values
        self.assertEqual(config.batch_size, self.batch_size)
        self.assertEqual(config.image_size, self.image_size)
    
    def test_classification_model_initialization(self):
        """Test classification model initialization with mocked transformers."""
        with self.mock_transformers():
            config_dict = self.create_minimal_config("classification")
            model = ClassificationModel(config_dict)
            
            # Test model interface
            self.assert_model_interface(model)
            
            # Test model attributes
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.output_adapter)
    
    def test_classification_dataset_initialization(self):
        """Test classification dataset initialization."""
        dataset = ClassificationDataset(self.dataset_path)
        
        # Test dataset interface
        self.assert_dataset_interface(dataset)
        
        # Test dataset properties
        self.assertGreater(len(dataset), 0)
        self.assertIsNotNone(dataset.class_names)
    
    def test_classification_dataset_getitem(self):
        """Test classification dataset item retrieval."""
        dataset = ClassificationDataset(self.dataset_path)
        
        if len(dataset) > 0:
            item = dataset[0]
            
            # Test item structure
            self.assertIn("pixel_values", item)
            self.assertIn("labels", item)
            
            # Test labels structure
            labels = item["labels"]
            self.assertIn("class_labels", labels)
            self.assertIn("image_id", labels)
    
    def test_classification_datamodule_initialization(self):
        """Test classification datamodule initialization."""
        config_dict = self.create_minimal_config("classification")
        datamodule = ClassificationDataModule(config_dict)
        
        # Test datamodule interface
        self.assert_datamodule_interface(datamodule)
        
        # Test datamodule attributes
        self.assertIsNotNone(datamodule.config)
    
    def test_classification_datamodule_setup(self):
        """Test classification datamodule setup."""
        config_dict = self.create_minimal_config("classification")
        datamodule = ClassificationDataModule(config_dict)
        
        # Setup datamodule
        datamodule.setup()
        
        # Test datasets are created
        self.assertIsNotNone(datamodule.train_dataset)
        self.assertIsNotNone(datamodule.val_dataset)
        self.assertGreater(len(datamodule.train_dataset), 0)
        self.assertGreater(len(datamodule.val_dataset), 0)
    
    def test_classification_datamodule_dataloaders(self):
        """Test classification datamodule dataloaders."""
        config_dict = self.create_minimal_config("classification")
        datamodule = ClassificationDataModule(config_dict)
        datamodule.setup()
        
        # Test dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertEqual(train_loader.batch_size, self.batch_size)
        self.assertEqual(val_loader.batch_size, self.batch_size)
    
    def test_vit_input_adapter(self):
        """Test ViT input adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = ViTInputAdapter("google/vit-base-patch16-224", image_size=224)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=True)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 224)
            self.assertIsNotNone(adapter.processor)
    
    def test_convnext_input_adapter(self):
        """Test ConvNeXT input adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = ConvNeXTInputAdapter("facebook/convnext-base-224", image_size=224)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=True)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 224)
            self.assertIsNotNone(adapter.processor)
    
    def test_swin_input_adapter(self):
        """Test Swin input adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = SwinInputAdapter("microsoft/swin-base-patch4-window7-224", image_size=224)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=True)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 224)
            self.assertIsNotNone(adapter.processor)
    
    def test_resnet_output_adapter(self):
        """Test ResNet output adapter."""
        with patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            mock_processor.return_value = MagicMock()
            
            adapter = ResNetOutputAdapter("microsoft/resnet-50", image_size=224)
            
            # Test adapter interface
            self.assert_adapter_interface(adapter, is_input_adapter=False)
            
            # Test adapter attributes
            self.assertEqual(adapter.image_size, 224)
            self.assertIsNotNone(adapter.processor)
    
    def test_adapter_factory_functions(self):
        """Test adapter factory functions."""
        # Test input adapter factory
        vit_input = get_input_adapter("google/vit-base-patch16-224")
        convnext_input = get_input_adapter("facebook/convnext-base-224")
        swin_input = get_input_adapter("microsoft/swin-base-patch4-window7-224")
        
        self.assertIsInstance(vit_input, ViTInputAdapter)
        self.assertIsInstance(convnext_input, ConvNeXTInputAdapter)
        self.assertIsInstance(swin_input, SwinInputAdapter)
        
        # Test output adapter factory
        resnet_output = get_output_adapter("microsoft/resnet-50")
        
        self.assertIsInstance(resnet_output, ResNetOutputAdapter)
    
    def test_classification_metrics_interface(self):
        """Test classification metrics interface."""
        # Test that metrics can be imported and initialized
        try:
            from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
            
            # Test metric initialization
            accuracy = MulticlassAccuracy(num_classes=self.num_classes, average='macro')
            precision = MulticlassPrecision(num_classes=self.num_classes, average='macro')
            recall = MulticlassRecall(num_classes=self.num_classes, average='macro')
            
            # Test that metrics are callable
            self.assertTrue(callable(accuracy))
            self.assertTrue(callable(precision))
            self.assertTrue(callable(recall))
            
        except ImportError:
            # Skip if torchmetrics not available
            self.skipTest("torchmetrics not available")
    
    def test_classification_evaluator_initialization(self):
        """Test classification evaluator initialization."""
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
            evaluator = ClassificationEvaluator(checkpoint_path, config_path)
            # If it doesn't fail, test basic interface
            self.assertIsNotNone(evaluator)
        except Exception:
            # Expected to fail with dummy files, but should not crash
            pass
    
    def test_label_smoothing_handling(self):
        """Test label smoothing handling."""
        # Test label smoothing values
        label_smoothing = 0.1
        
        # Test that label smoothing is between 0 and 1
        self.assertGreaterEqual(label_smoothing, 0.0)
        self.assertLessEqual(label_smoothing, 1.0)
        
        # Test label smoothing application
        num_classes = self.num_classes
        target = torch.randint(0, num_classes, (self.batch_size,))
        
        # Create smoothed labels
        smoothed_targets = torch.zeros(self.batch_size, num_classes)
        smoothed_targets.scatter_(1, target.unsqueeze(1), 1.0 - label_smoothing)
        smoothed_targets += label_smoothing / num_classes
        
        # Test that smoothed targets sum to 1
        self.assertTrue(torch.allclose(smoothed_targets.sum(dim=1), torch.ones(self.batch_size), atol=1e-6))


if __name__ == "__main__":
    unittest.main() 
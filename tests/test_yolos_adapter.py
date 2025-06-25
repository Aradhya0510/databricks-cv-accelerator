#!/usr/bin/env python3
"""
Tests for YOLOS adapter functionality.
"""

import unittest
import torch
from PIL import Image
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tasks.detection.adapters import YOLOSAdapter, YOLOSOutputAdapter, get_adapter, get_output_adapter

class TestYOLOSAdapter(unittest.TestCase):
    """Test cases for YOLOS adapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_name = "hustvl/yolos-tiny"  # Use tiny model for faster tests
        self.image_size = 800
        self.adapter = YOLOSAdapter(self.model_name, self.image_size)
        self.output_adapter = YOLOSOutputAdapter(self.model_name, self.image_size)
        
        # Create a dummy image
        self.image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        
        # Create dummy target
        self.target = {
            "boxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
            "labels": torch.tensor([1, 2]),
            "image_id": torch.tensor([0])
        }
    
    def test_adapter_initialization(self):
        """Test that YOLOS adapter initializes correctly."""
        self.assertIsNotNone(self.adapter.processor)
        self.assertEqual(self.adapter.image_size, self.image_size)
    
    def test_output_adapter_initialization(self):
        """Test that YOLOS output adapter initializes correctly."""
        self.assertIsNotNone(self.output_adapter.processor)
        self.assertEqual(self.output_adapter.image_size, self.image_size)
    
    def test_image_processing(self):
        """Test that images are processed correctly."""
        processed_image, adapted_target = self.adapter(self.image, self.target)
        
        # Check image shape
        self.assertEqual(processed_image.shape, (3, self.image_size, self.image_size))
        self.assertTrue(torch.is_tensor(processed_image))
        
        # Check target structure
        self.assertIn("boxes", adapted_target)
        self.assertIn("class_labels", adapted_target)
        self.assertIn("image_id", adapted_target)
        self.assertIn("size", adapted_target)
        
        # Check box format (should be normalized [cx, cy, w, h])
        boxes = adapted_target["boxes"]
        self.assertEqual(boxes.shape[1], 4)  # 4 coordinates per box
        
        # Check normalization (values should be between 0 and 1)
        self.assertTrue(torch.all(boxes >= 0))
        self.assertTrue(torch.all(boxes <= 1))
    
    def test_empty_target_handling(self):
        """Test handling of empty targets."""
        empty_target = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor([0])
        }
        
        processed_image, adapted_target = self.adapter(self.image, empty_target)
        
        # Should handle empty targets gracefully
        self.assertEqual(adapted_target["boxes"].shape[0], 0)
        self.assertEqual(adapted_target["class_labels"].shape[0], 0)
    
    def test_box_format_conversion(self):
        """Test conversion from xyxy to cxcwh format."""
        # Test the static method
        xyxy_boxes = torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]])
        cxcwh_boxes = self.adapter._xyxy_to_cxcwh(xyxy_boxes)
        
        # Check shape
        self.assertEqual(cxcwh_boxes.shape, xyxy_boxes.shape)
        
        # Check first box: [0, 0, 100, 100] -> [50, 50, 100, 100]
        expected_first = torch.tensor([50, 50, 100, 100])
        torch.testing.assert_close(cxcwh_boxes[0], expected_first, atol=1e-6)
        
        # Check second box: [50, 50, 150, 150] -> [100, 100, 100, 100]
        expected_second = torch.tensor([100, 100, 100, 100])
        torch.testing.assert_close(cxcwh_boxes[1], expected_second, atol=1e-6)
    
    def test_output_adaptation(self):
        """Test output adaptation functionality."""
        # Create dummy model outputs
        batch_size, num_queries, num_classes = 2, 100, 81
        dummy_outputs = type('DummyOutput', (), {
            'loss': torch.tensor(0.5),
            'pred_boxes': torch.randn(batch_size, num_queries, 4),
            'logits': torch.randn(batch_size, num_queries, num_classes),
            'loss_dict': {'classification_loss': torch.tensor(0.3), 'bbox_loss': torch.tensor(0.2)}
        })()
        
        adapted_outputs = self.output_adapter.adapt_output(dummy_outputs)
        
        # Check structure
        self.assertIn("loss", adapted_outputs)
        self.assertIn("pred_boxes", adapted_outputs)
        self.assertIn("pred_logits", adapted_outputs)
        self.assertIn("loss_dict", adapted_outputs)
        
        # Check values
        self.assertEqual(adapted_outputs["loss"], dummy_outputs.loss)
        torch.testing.assert_close(adapted_outputs["pred_boxes"], dummy_outputs.pred_boxes)
        torch.testing.assert_close(adapted_outputs["pred_logits"], dummy_outputs.logits)
    
    def test_target_formatting(self):
        """Test target formatting for metrics."""
        # Create dummy targets in adapter format
        targets = [{
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),  # [cx, cy, w, h] normalized
            "class_labels": torch.tensor([1]),
            "size": torch.tensor([480, 640]),  # [h, w]
            "image_id": torch.tensor([0])
        }]
        
        formatted_targets = self.output_adapter.format_targets(targets)
        
        # Check structure
        self.assertEqual(len(formatted_targets), 1)
        formatted_target = formatted_targets[0]
        
        self.assertIn("boxes", formatted_target)
        self.assertIn("labels", formatted_target)
        self.assertIn("image_id", formatted_target)
        
        # Check box format conversion (should be [x1, y1, x2, y2] in absolute pixels)
        boxes = formatted_target["boxes"]
        self.assertEqual(boxes.shape, (1, 4))
        
        # Convert back: [cx, cy, w, h] normalized -> [x1, y1, x2, y2] absolute
        # cx=0.5*640=320, cy=0.5*480=240, w=0.2*640=128, h=0.3*480=144
        # x1 = cx - w/2 = 320 - 64 = 256, y1 = cy - h/2 = 240 - 72 = 168
        # x2 = cx + w/2 = 320 + 64 = 384, y2 = cy + h/2 = 240 + 72 = 312
        expected_boxes = torch.tensor([[256, 168, 384, 312]])
        torch.testing.assert_close(boxes, expected_boxes, atol=1.0)
    
    def test_adapter_factory_functions(self):
        """Test the adapter factory functions."""
        # Test input adapter factory
        yolos_adapter = get_adapter("hustvl/yolos-base")
        self.assertIsInstance(yolos_adapter, YOLOSAdapter)
        
        detr_adapter = get_adapter("facebook/detr-resnet-50")
        from tasks.detection.adapters import DETRAdapter
        self.assertIsInstance(detr_adapter, DETRAdapter)
        
        # Test output adapter factory
        yolos_output_adapter = get_output_adapter("hustvl/yolos-base")
        self.assertIsInstance(yolos_output_adapter, YOLOSOutputAdapter)
        
        detr_output_adapter = get_output_adapter("facebook/detr-resnet-50")
        from tasks.detection.adapters import DETROutputAdapter
        self.assertIsInstance(detr_output_adapter, DETROutputAdapter)
    
    def test_model_name_detection(self):
        """Test that model name detection works correctly."""
        # Test YOLOS variants
        yolos_variants = [
            "hustvl/yolos-tiny",
            "hustvl/yolos-small", 
            "hustvl/yolos-base",
            "hustvl/yolos-base-22k"
        ]
        
        for variant in yolos_variants:
            adapter = get_adapter(variant)
            self.assertIsInstance(adapter, YOLOSAdapter)
            
            output_adapter = get_output_adapter(variant)
            self.assertIsInstance(output_adapter, YOLOSOutputAdapter)

if __name__ == "__main__":
    unittest.main() 
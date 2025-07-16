#!/usr/bin/env python3
"""
Test script for standalone model wrapper.
This script tests that the standalone wrapper can be created and used without import issues.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForObjectDetection, AutoConfig

def create_standalone_model_wrapper(model, config):
    """Create a standalone model wrapper that doesn't depend on relative imports."""
    
    class StandaloneDetectionModel(nn.Module):
        """Standalone detection model wrapper for serving."""
        
        def __init__(self, model, config):
            super().__init__()
            self.model = model
            self.config = config
            self.confidence_threshold = config.get('confidence_threshold', 0.5)
            self.iou_threshold = config.get('iou_threshold', 0.5)
            self.max_detections = config.get('max_detections', 100)
            
        def forward(self, image):
            """
            Forward pass for serving.
            
            Args:
                image: Input image tensor of shape (batch_size, 3, height, width)
            
            Returns:
                dict: Dictionary containing predictions
                    - boxes: Bounding boxes (batch_size, num_detections, 4)
                    - scores: Confidence scores (batch_size, num_detections)
                    - labels: Class labels (batch_size, num_detections)
            """
            # Ensure model is in eval mode
            self.model.eval()
            
            # Move input to same device as model
            device = next(self.model.parameters()).device
            image = image.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(pixel_values=image)
            
            # Extract predictions
            if hasattr(outputs, 'logits') and hasattr(outputs, 'pred_boxes'):
                # DETR-style outputs
                logits = outputs.logits
                pred_boxes = outputs.pred_boxes
                
                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=-1)
                scores, labels = torch.max(probs, dim=-1)
                
                # Apply confidence threshold
                mask = scores > self.confidence_threshold
                
                # Filter predictions
                filtered_boxes = pred_boxes[mask]
                filtered_scores = scores[mask]
                filtered_labels = labels[mask]
                
                # Limit number of detections
                if len(filtered_boxes) > self.max_detections:
                    # Keep top detections by score
                    top_indices = torch.argsort(filtered_scores, descending=True)[:self.max_detections]
                    filtered_boxes = filtered_boxes[top_indices]
                    filtered_scores = filtered_scores[top_indices]
                    filtered_labels = filtered_labels[top_indices]
                
                return {
                    'boxes': filtered_boxes.unsqueeze(0),  # Add batch dimension
                    'scores': filtered_scores.unsqueeze(0),
                    'labels': filtered_labels.unsqueeze(0)
                }
            else:
                # Fallback for other model types
                return {
                    'boxes': torch.zeros(1, 0, 4),
                    'scores': torch.zeros(1, 0),
                    'labels': torch.zeros(1, 0, dtype=torch.long)
                }
    
    # Create the standalone wrapper
    standalone_model = StandaloneDetectionModel(model, config)
    
    return standalone_model

def test_standalone_wrapper():
    """Test the standalone wrapper functionality."""
    
    print("🧪 Testing standalone model wrapper...")
    
    try:
        # Create a dummy model for testing
        config = {
            'confidence_threshold': 0.5,
            'iou_threshold': 0.5,
            'max_detections': 100
        }
        
        # Create a simple dummy model (in real usage, this would be the trained model)
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = nn.Parameter(torch.randn(1))
                
            def forward(self, pixel_values):
                # Simulate DETR outputs
                batch_size = pixel_values.shape[0]
                return type('Outputs', (), {
                    'logits': torch.randn(batch_size, 100, 91),  # 100 queries, 91 classes
                    'pred_boxes': torch.randn(batch_size, 100, 4)  # 100 queries, 4 coords
                })()
        
        dummy_model = DummyModel()
        
        # Create standalone wrapper
        standalone_model = create_standalone_model_wrapper(dummy_model, config)
        
        # Test forward pass
        test_input = torch.randn(1, 3, 800, 800)
        output = standalone_model(test_input)
        
        print("✅ Standalone wrapper test successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output type: {type(output)}")
        print(f"   Output keys: {list(output.keys()) if isinstance(output, dict) else 'N/A'}")
        
        if isinstance(output, dict):
            for key, value in output.items():
                print(f"   {key} shape: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Standalone wrapper test failed: {e}")
        return False

if __name__ == "__main__":
    test_standalone_wrapper() 
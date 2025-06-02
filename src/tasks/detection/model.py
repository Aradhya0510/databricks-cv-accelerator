from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoModelForObjectDetection, AutoConfig, PreTrainedModel
from .adapters import get_output_adapter, DETROutputAdapter

@dataclass
class DetectionModelConfig:
    """Configuration for detection model."""
    model_name: str
    num_classes: int
    pretrained: bool = True
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_detections: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    scheduler_params: Optional[Dict[str, Any]] = None
    epochs: int = 100
    task_type: str = "detection"
    class_names: Optional[List[str]] = None
    model_kwargs: Optional[Dict[str, Any]] = None

class DetectionModel(pl.LightningModule):
    """Base detection model that can work with any Hugging Face object detection model."""
    
    def __init__(self, config: Union[Dict[str, Any], DetectionModelConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = DetectionModelConfig(**config)
        self.config = config
        self.save_hyperparameters(config.__dict__)
        
        # Initialize model
        self._init_model()
        
        # Initialize metrics
        self._init_metrics()
        
        # Initialize output adapter
        self.output_adapter = get_output_adapter(config.model_name)
    
    def _init_model(self) -> None:
        """Initialize the model architecture."""
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_classes,
                **self.config.model_kwargs or {}
            )
            
            # Initialize model with ignore_mismatched_sizes=True to handle class size differences
            # Force model to CPU during initialization
            with torch.device('cpu'):
                self.model = AutoModelForObjectDetection.from_pretrained(
                    self.config.model_name,
                    config=model_config,
                    ignore_mismatched_sizes=True,  # Allow loading with different class sizes
                    device_map=None,  # Prevent automatic device mapping
                    **self.config.model_kwargs or {}
                )
                # Explicitly move model to CPU
                self.model = self.model.to('cpu')
            
            # Set model parameters
            self.model.config.confidence_threshold = self.config.confidence_threshold
            self.model.config.iou_threshold = self.config.iou_threshold
            self.model.config.max_detections = self.config.max_detections
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def _init_metrics(self) -> None:
        """Initialize metrics for training, validation, and testing."""
        self.train_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True
        )
        self.val_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True
        )
        self.test_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Forward pass of the model.
        
        Args:
            pixel_values: Input images
            pixel_mask: Optional attention mask
            labels: Optional target labels
            
        Returns:
            Dictionary containing model outputs including:
            - loss: Training loss if labels are provided
            - pred_boxes: Predicted bounding boxes
            - pred_logits: Predicted class logits
            - loss_dict: Dictionary of individual loss components
        """
        # Validate input
        if pixel_values.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {pixel_values.dim()}D")
        
        # Adapt targets if needed
        if labels is not None:
            labels = self.output_adapter.adapt_targets(labels)
            
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )
        return self.output_adapter.adapt_output(outputs)
    
    def _format_predictions(self, outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Format model outputs for metric computation.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            List of prediction dictionaries for each image
        """
        return self.output_adapter.format_predictions(outputs)
    
    def _format_targets(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format batch targets for metric computation.
        
        Args:
            batch: Batch dictionary containing targets
            
        Returns:
            List of target dictionaries for each image
        """
        return self.output_adapter.format_targets(batch["labels"])
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask"),
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        preds = self._format_predictions(outputs)
        targets = self._format_targets(batch)
        
        # Update metrics
        self.train_map.update(preds=preds, target=targets)
        
        # Log metrics
        self.log("train_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        for k, v in outputs["loss_dict"].items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True)
        
        return outputs["loss"]
    
    def on_train_epoch_end(self) -> None:
        """Calculate and log mAP metrics at the end of training epoch."""
        map_metrics = self.train_map.compute()
        
        # Log each metric separately
        self.log("train_map", map_metrics["map"], prog_bar=True)
        self.log("train_map_50", map_metrics["map_50"])
        self.log("train_map_75", map_metrics["map_75"])
        self.log("train_map_small", map_metrics["map_small"])
        self.log("train_map_medium", map_metrics["map_medium"])
        self.log("train_map_large", map_metrics["map_large"])
        self.log("train_mar_1", map_metrics["mar_1"])
        self.log("train_mar_10", map_metrics["mar_10"])
        self.log("train_mar_100", map_metrics["mar_100"])
        self.log("train_mar_small", map_metrics["mar_small"])
        self.log("train_mar_medium", map_metrics["mar_medium"])
        self.log("train_mar_large", map_metrics["mar_large"])
        
        # Log per-class metrics
        for i, class_id in enumerate(map_metrics["classes"]):
            if map_metrics["map_per_class"][i] != -1:
                self.log(f"train_map_class_{class_id}", map_metrics["map_per_class"][i])
            if map_metrics["mar_100_per_class"][i] != -1:
                self.log(f"train_mar_100_class_{class_id}", map_metrics["mar_100_per_class"][i])
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask"),
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        preds = self._format_predictions(outputs)
        targets = self._format_targets(batch)
        
        # Update metrics
        self.val_map.update(preds=preds, target=targets)
        
        # Log metrics
        self.log("val_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        """Calculate and log mAP metrics at the end of validation epoch."""
        map_metrics = self.val_map.compute()
        
        # Log each metric separately
        self.log("val_map", map_metrics["map"], prog_bar=True)
        self.log("val_map_50", map_metrics["map_50"])
        self.log("val_map_75", map_metrics["map_75"])
        self.log("val_map_small", map_metrics["map_small"])
        self.log("val_map_medium", map_metrics["map_medium"])
        self.log("val_map_large", map_metrics["map_large"])
        self.log("val_mar_1", map_metrics["mar_1"])
        self.log("val_mar_10", map_metrics["mar_10"])
        self.log("val_mar_100", map_metrics["mar_100"])
        self.log("val_mar_small", map_metrics["mar_small"])
        self.log("val_mar_medium", map_metrics["mar_medium"])
        self.log("val_mar_large", map_metrics["mar_large"])
        
        # Log per-class metrics
        for i, class_id in enumerate(map_metrics["classes"]):
            if map_metrics["map_per_class"][i] != -1:
                self.log(f"val_map_class_{class_id}", map_metrics["map_per_class"][i])
            if map_metrics["mar_100_per_class"][i] != -1:
                self.log(f"val_mar_100_class_{class_id}", map_metrics["mar_100_per_class"][i])
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask"),
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        preds = self._format_predictions(outputs)
        targets = self._format_targets(batch)
        
        # Update metrics
        self.test_map.update(preds=preds, target=targets)
        
        # Log metrics
        self.log("test_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        for k, v in outputs["loss_dict"].items():
            self.log(f"test_{k}", v, on_step=True, on_epoch=True)
    
    def on_test_epoch_end(self) -> None:
        """Calculate and log mAP metrics at the end of test epoch."""
        map_metrics = self.test_map.compute()
        
        # Log each metric separately
        self.log("test_map", map_metrics["map"], prog_bar=True)
        self.log("test_map_50", map_metrics["map_50"])
        self.log("test_map_75", map_metrics["map_75"])
        self.log("test_map_small", map_metrics["map_small"])
        self.log("test_map_medium", map_metrics["map_medium"])
        self.log("test_map_large", map_metrics["map_large"])
        self.log("test_mar_1", map_metrics["mar_1"])
        self.log("test_mar_10", map_metrics["mar_10"])
        self.log("test_mar_100", map_metrics["mar_100"])
        self.log("test_mar_small", map_metrics["mar_small"])
        self.log("test_mar_medium", map_metrics["mar_medium"])
        self.log("test_mar_large", map_metrics["mar_large"])
        
        # Log per-class metrics
        for i, class_id in enumerate(map_metrics["classes"]):
            if map_metrics["map_per_class"][i] != -1:
                self.log(f"test_map_class_{class_id}", map_metrics["map_per_class"][i])
            if map_metrics["mar_100_per_class"][i] != -1:
                self.log(f"test_mar_100_class_{class_id}", map_metrics["mar_100_per_class"][i])
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.scheduler == "cosine":
            scheduler_params = self.config.scheduler_params or {
                "T_max": self.config.epochs,
                "eta_min": 1e-6
            }
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **scheduler_params
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        return optimizer
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_classes: int,
        **kwargs
    ) -> "DetectionModel":
        """Create a model from a pretrained checkpoint.
        
        Args:
            model_name: Name or path of the pretrained model
            num_classes: Number of output classes
            **kwargs: Additional arguments for model configuration
            
        Returns:
            Initialized model
        """
        config = DetectionModelConfig(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
        return cls(config)
    
    def on_train_epoch_start(self) -> None:
        """Reset training metrics at the start of each epoch."""
        self.train_map.reset()
    
    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics at the start of each epoch."""
        self.val_map.reset()
    
    def on_test_epoch_start(self) -> None:
        """Reset test metrics at the start of each epoch."""
        self.test_map.reset()
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save additional state to checkpoint.
        
        Args:
            checkpoint: Dictionary to save state to
        """
        checkpoint["model_config"] = self.config.__dict__
        checkpoint["class_names"] = self.config.class_names
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load additional state from checkpoint.
        
        Args:
            checkpoint: Dictionary to load state from
        """
        if "model_config" in checkpoint:
            self.config = DetectionModelConfig(**checkpoint["model_config"])
        if "class_names" in checkpoint:
            self.config.class_names = checkpoint["class_names"] 
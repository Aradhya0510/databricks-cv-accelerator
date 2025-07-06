from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoModelForObjectDetection, AutoConfig, PreTrainedModel
from .adapters import get_input_adapter, get_output_adapter

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
    image_size: int = 640
    num_workers: int = 1  # Add num_workers parameter
    
    @property
    def sync_dist_flag(self) -> bool:
        """Return True if num_workers > 1 (distributed training), False otherwise."""
        return self.num_workers > 1

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
        
        # Initialize output adapter with image size from config
        self.output_adapter = get_output_adapter(
            config.model_name,
            image_size=config.image_size if hasattr(config, 'image_size') else 640
        )
    
    def _init_model(self) -> None:
        """Initialize the model architecture."""
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_classes,
                **self.config.model_kwargs or {}
            )
            
            # Initialize model without forcing CPU
            self.model = AutoModelForObjectDetection.from_pretrained(
                self.config.model_name,
                config=model_config,
                ignore_mismatched_sizes=True,  # Allow loading with different class sizes
                **self.config.model_kwargs or {}
            )
            
            # Set model parameters
            self.model.config.confidence_threshold = self.config.confidence_threshold
            self.model.config.iou_threshold = self.config.iou_threshold
            self.model.config.max_detections = self.config.max_detections
            
            # Save model config for checkpointing
            self.model_config = model_config
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def _init_metrics(self) -> None:
        """Initialize metrics for training, validation, and testing."""
        # MeanAveragePrecision expects absolute pixel coordinates
        metric_kwargs = {
            "box_format": "xyxy",
            "iou_type": "bbox",
            "class_metrics": True
        }
        
        self.train_map = MeanAveragePrecision(**metric_kwargs)
        self.val_map = MeanAveragePrecision(**metric_kwargs)
        self.test_map = MeanAveragePrecision(**metric_kwargs)
    
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
        
        # No need to adapt targets - they're already in the correct format from the data adapter
            
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )
        
        # Extract and adapt outputs
        adapted_outputs = self.output_adapter.adapt_output(outputs)
        
        # Ensure loss_dict exists
        if "loss_dict" not in adapted_outputs:
            adapted_outputs["loss_dict"] = {}
        
        return adapted_outputs
    
    def _format_predictions(self, outputs: Dict[str, Any], batch: Optional[Dict[str, Any]] = None) -> List[Dict[str, torch.Tensor]]:
        """Format model outputs for metric computation.
        
        Args:
            outputs: Model outputs dictionary
            batch: Optional batch dictionary for getting target sizes
            
        Returns:
            List of prediction dictionaries for each image
        """
        return self.output_adapter.format_predictions(outputs, batch)
    
    def _format_targets(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Format batch targets for metric computation.
        
        Args:
            batch: Dictionary containing targets
            
        Returns:
            List of target dictionaries for each image
        """
        return self.output_adapter.format_targets(batch["labels"])
    
    def _log_metric(self, name: str, value: float, **kwargs) -> None:
        """Log a metric using PyTorch Lightning's logging system."""
        # Let PyTorch Lightning handle the logging and distributed synchronization
        self.log(name, value, **kwargs)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Dictionary containing:
                - pixel_values: Batch of images
                - labels: List of target dictionaries
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # Forward pass
        outputs = self.forward(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        preds = self._format_predictions(outputs, batch)
        targets = self._format_targets(batch)
        
        # Update metrics
        self.train_map.update(preds=preds, target=targets)
        
        # Log metrics with sync_dist flag and batch_size
        self.log("train_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.config.sync_dist_flag)
        for k, v in outputs["loss_dict"].items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, sync_dist=self.config.sync_dist_flag)
        
        return outputs["loss"]
    
    def on_train_epoch_end(self) -> None:
        """Calculate and log mAP metrics at the end of training epoch."""
        map_metrics = self.train_map.compute()
        
        # Log each metric using PyTorch Lightning's logging with sync_dist flag
        self.log("train_map", map_metrics["map"], prog_bar=True, sync_dist=self.config.sync_dist_flag)
        self.log("train_map_50", map_metrics["map_50"], sync_dist=self.config.sync_dist_flag)
        self.log("train_map_75", map_metrics["map_75"], sync_dist=self.config.sync_dist_flag)
        self.log("train_map_small", map_metrics["map_small"], sync_dist=self.config.sync_dist_flag)
        self.log("train_map_medium", map_metrics["map_medium"], sync_dist=self.config.sync_dist_flag)
        self.log("train_map_large", map_metrics["map_large"], sync_dist=self.config.sync_dist_flag)
        self.log("train_mar_1", map_metrics["mar_1"], sync_dist=self.config.sync_dist_flag)
        self.log("train_mar_10", map_metrics["mar_10"], sync_dist=self.config.sync_dist_flag)
        self.log("train_mar_100", map_metrics["mar_100"], sync_dist=self.config.sync_dist_flag)
        self.log("train_mar_small", map_metrics["mar_small"], sync_dist=self.config.sync_dist_flag)
        self.log("train_mar_medium", map_metrics["mar_medium"], sync_dist=self.config.sync_dist_flag)
        self.log("train_mar_large", map_metrics["mar_large"], sync_dist=self.config.sync_dist_flag)
        
        # Log per-class metrics
        for i, class_id in enumerate(map_metrics["classes"]):
            if map_metrics["map_per_class"][i] != -1:
                self.log(f"train_map_class_{class_id}", map_metrics["map_per_class"][i], sync_dist=self.config.sync_dist_flag)
            if map_metrics["mar_100_per_class"][i] != -1:
                self.log(f"train_mar_100_class_{class_id}", map_metrics["mar_100_per_class"][i], sync_dist=self.config.sync_dist_flag)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Dictionary containing:
                - pixel_values: Batch of images
                - labels: List of target dictionaries
            batch_idx: Batch index
        """
        # Forward pass
        outputs = self.forward(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        preds = self._format_predictions(outputs, batch)
        targets = self._format_targets(batch)
        
        # Update metrics
        self.val_map.update(preds=preds, target=targets)
        
        # Log metrics with sync_dist flag and batch_size
        self.log("val_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.config.sync_dist_flag)
        for k, v in outputs["loss_dict"].items():
            self.log(f"val_{k}", v, on_step=True, on_epoch=True, sync_dist=self.config.sync_dist_flag)
    
    def on_validation_epoch_end(self) -> None:
        """Calculate and log mAP metrics at the end of validation epoch."""
        map_metrics = self.val_map.compute()
        
        # Log each metric separately with sync_dist flag
        self.log("val_map", map_metrics["map"], prog_bar=True, sync_dist=self.config.sync_dist_flag)
        self.log("val_map_50", map_metrics["map_50"], sync_dist=self.config.sync_dist_flag)
        self.log("val_map_75", map_metrics["map_75"], sync_dist=self.config.sync_dist_flag)
        self.log("val_map_small", map_metrics["map_small"], sync_dist=self.config.sync_dist_flag)
        self.log("val_map_medium", map_metrics["map_medium"], sync_dist=self.config.sync_dist_flag)
        self.log("val_map_large", map_metrics["map_large"], sync_dist=self.config.sync_dist_flag)
        self.log("val_mar_1", map_metrics["mar_1"], sync_dist=self.config.sync_dist_flag)
        self.log("val_mar_10", map_metrics["mar_10"], sync_dist=self.config.sync_dist_flag)
        self.log("val_mar_100", map_metrics["mar_100"], sync_dist=self.config.sync_dist_flag)
        self.log("val_mar_small", map_metrics["mar_small"], sync_dist=self.config.sync_dist_flag)
        self.log("val_mar_medium", map_metrics["mar_medium"], sync_dist=self.config.sync_dist_flag)
        self.log("val_mar_large", map_metrics["mar_large"], sync_dist=self.config.sync_dist_flag)
        
        # Log per-class metrics
        for i, class_id in enumerate(map_metrics["classes"]):
            if map_metrics["map_per_class"][i] != -1:
                self.log(f"val_map_class_{class_id}", map_metrics["map_per_class"][i], sync_dist=self.config.sync_dist_flag)
            if map_metrics["mar_100_per_class"][i] != -1:
                self.log(f"val_mar_100_class_{class_id}", map_metrics["mar_100_per_class"][i], sync_dist=self.config.sync_dist_flag)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.
        
        Args:
            batch: Dictionary containing:
                - pixel_values: Batch of images
                - labels: List of target dictionaries
            batch_idx: Batch index
        """
        # Forward pass
        outputs = self.forward(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        preds = self._format_predictions(outputs, batch)
        targets = self._format_targets(batch)
        
        # Update metrics
        self.test_map.update(preds=preds, target=targets)
        
        # Log metrics with sync_dist flag and batch_size
        self.log("test_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.config.sync_dist_flag)
        for k, v in outputs["loss_dict"].items():
            self.log(f"test_{k}", v, on_step=True, on_epoch=True, sync_dist=self.config.sync_dist_flag)
    
    def on_test_epoch_end(self) -> None:
        """Calculate and log mAP metrics at the end of test epoch."""
        map_metrics = self.test_map.compute()
        
        # Log each metric separately with sync_dist flag
        self.log("test_map", map_metrics["map"], prog_bar=True, sync_dist=self.config.sync_dist_flag)
        self.log("test_map_50", map_metrics["map_50"], sync_dist=self.config.sync_dist_flag)
        self.log("test_map_75", map_metrics["map_75"], sync_dist=self.config.sync_dist_flag)
        self.log("test_map_small", map_metrics["map_small"], sync_dist=self.config.sync_dist_flag)
        self.log("test_map_medium", map_metrics["map_medium"], sync_dist=self.config.sync_dist_flag)
        self.log("test_map_large", map_metrics["map_large"], sync_dist=self.config.sync_dist_flag)
        self.log("test_mar_1", map_metrics["mar_1"], sync_dist=self.config.sync_dist_flag)
        self.log("test_mar_10", map_metrics["mar_10"], sync_dist=self.config.sync_dist_flag)
        self.log("test_mar_100", map_metrics["mar_100"], sync_dist=self.config.sync_dist_flag)
        self.log("test_mar_small", map_metrics["mar_small"], sync_dist=self.config.sync_dist_flag)
        self.log("test_mar_medium", map_metrics["mar_medium"], sync_dist=self.config.sync_dist_flag)
        self.log("test_mar_large", map_metrics["mar_large"], sync_dist=self.config.sync_dist_flag)
        
        # Log per-class metrics
        for i, class_id in enumerate(map_metrics["classes"]):
            if map_metrics["map_per_class"][i] != -1:
                self.log(f"test_map_class_{class_id}", map_metrics["map_per_class"][i], sync_dist=self.config.sync_dist_flag)
            if map_metrics["mar_100_per_class"][i] != -1:
                self.log(f"test_mar_100_class_{class_id}", map_metrics["mar_100_per_class"][i], sync_dist=self.config.sync_dist_flag)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer parameters from config
        optimizer_params = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay
        }
        
        # Check if model has a backbone (common in vision models)
        if hasattr(self.model, 'backbone'):
            # Separate parameters for backbone and other parts
            backbone_params = []
            other_params = []
            
            for name, param in self.named_parameters():
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            
            # Create parameter groups with different learning rates
            param_groups = [
                {
                    "params": backbone_params,
                    "lr": self.config.learning_rate * 0.1,  # Lower learning rate for backbone
                    "weight_decay": self.config.weight_decay
                },
                {
                    "params": other_params,
                    "lr": self.config.learning_rate,  # Higher learning rate for task-specific parts
                    "weight_decay": self.config.weight_decay
                }
            ]
            
            # Create optimizer with parameter groups
            optimizer = torch.optim.AdamW(param_groups)
        else:
            # If no backbone, use standard optimizer
            optimizer = torch.optim.AdamW(self.parameters(), **optimizer_params)
        
        # Configure scheduler with warmup
        if self.config.scheduler == "cosine":
            # Calculate total steps and warmup steps
            total_steps = self.config.epochs * len(self.trainer.datamodule.train_dataloader())
            warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warmup
            
            # Create warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            # Create cosine scheduler
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-6
            )
            
            # Combine schedulers
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        else:
            # Default to no scheduler
            return {"optimizer": optimizer}
    
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
        checkpoint["model_config"] = self.model_config.to_dict()
        checkpoint["class_names"] = self.config.class_names
        checkpoint["optimizer_params"] = {
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "scheduler": self.config.scheduler,
            "scheduler_params": self.config.scheduler_params
        }
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load additional state from checkpoint.
        
        Args:
            checkpoint: Dictionary to load state from
        """
        if "model_config" in checkpoint:
            self.model_config = AutoConfig.from_dict(checkpoint["model_config"])
        if "class_names" in checkpoint:
            self.config.class_names = checkpoint["class_names"]
        if "optimizer_params" in checkpoint:
            params = checkpoint["optimizer_params"]
            self.config.learning_rate = params["learning_rate"]
            self.config.weight_decay = params["weight_decay"]
            self.config.scheduler = params["scheduler"]
            self.config.scheduler_params = params["scheduler_params"] 
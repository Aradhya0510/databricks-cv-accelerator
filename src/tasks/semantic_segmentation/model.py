from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Dice, JaccardIndex, Accuracy, Precision, Recall
from transformers import (
    AutoModelForSemanticSegmentation,
    AutoConfig,
    PreTrainedModel
)
from .adapters import get_semantic_adapter, SemanticSegmentationOutputAdapter

@dataclass
class SemanticSegmentationModelConfig:
    """Configuration for semantic segmentation model."""
    model_name: str
    num_classes: int
    pretrained: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    epochs: int = 10
    class_names: Optional[List[str]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    image_size: int = 512

class SemanticSegmentationModel(pl.LightningModule):
    """Semantic segmentation model that works with Hugging Face semantic segmentation models."""
    
    def __init__(self, config: Union[Dict[str, Any], SemanticSegmentationModelConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = SemanticSegmentationModelConfig(**config)
        self.config = config
        self.save_hyperparameters(config.__dict__)
        
        # Initialize model
        self._init_model()
        
        # Initialize metrics
        self._init_metrics()
        
        # Initialize output adapter
        self.output_adapter = SemanticSegmentationOutputAdapter()
    
    def _init_model(self) -> None:
        """Initialize the model architecture."""
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_classes,
                **self.config.model_kwargs or {}
            )
            
            # Initialize semantic segmentation model
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                self.config.model_name,
                config=model_config,
                ignore_mismatched_sizes=True,
                **self.config.model_kwargs or {}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def _init_metrics(self) -> None:
        """Initialize metrics for training, validation, and testing."""
        # Semantic segmentation metrics
        self.train_dice = Dice(num_classes=self.config.num_classes)
        self.train_iou = JaccardIndex(task="multiclass", num_classes=self.config.num_classes)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.train_precision = Precision(task="multiclass", num_classes=self.config.num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=self.config.num_classes)
        
        self.val_dice = Dice(num_classes=self.config.num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=self.config.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=self.config.num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=self.config.num_classes)
        
        self.test_dice = Dice(num_classes=self.config.num_classes)
        self.test_iou = JaccardIndex(task="multiclass", num_classes=self.config.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=self.config.num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=self.config.num_classes)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Forward pass of the model.
        
        Args:
            pixel_values: Input images
            labels: Optional target labels
            
        Returns:
            Dictionary containing model outputs including:
            - loss: Training loss if labels are provided
            - logits: Predicted class logits
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
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        predictions = self._format_predictions(outputs)
        targets = self._format_targets(batch)
        
        # Update metrics
        for pred, target in zip(predictions, targets):
            pred_mask = pred["semantic_mask"]
            target_mask = target["semantic_mask"]
            
            self.train_dice.update(pred_mask, target_mask)
            self.train_iou.update(pred_mask, target_mask)
            self.train_accuracy.update(pred_mask, target_mask)
            self.train_precision.update(pred_mask, target_mask)
            self.train_recall.update(pred_mask, target_mask)
        
        # Log loss
        self.log("train_loss", outputs["loss"], prog_bar=True)
        
        return outputs["loss"]
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log metrics
        self.log("train_dice", self.train_dice.compute(), prog_bar=True)
        self.log("train_iou", self.train_iou.compute(), prog_bar=True)
        self.log("train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute(), prog_bar=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True)
        
        # Reset metrics
        self.train_dice.reset()
        self.train_iou.reset()
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        predictions = self._format_predictions(outputs)
        targets = self._format_targets(batch)
        
        # Update metrics
        for pred, target in zip(predictions, targets):
            pred_mask = pred["semantic_mask"]
            target_mask = target["semantic_mask"]
            
            self.val_dice.update(pred_mask, target_mask)
            self.val_iou.update(pred_mask, target_mask)
            self.val_accuracy.update(pred_mask, target_mask)
            self.val_precision.update(pred_mask, target_mask)
            self.val_recall.update(pred_mask, target_mask)
        
        # Log loss
        self.log("val_loss", outputs["loss"], prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Log metrics
        self.log("val_dice", self.val_dice.compute(), prog_bar=True)
        self.log("val_iou", self.val_iou.compute(), prog_bar=True)
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute(), prog_bar=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True)
        
        # Reset metrics
        self.val_dice.reset()
        self.val_iou.reset()
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        # Format predictions and targets for metrics
        predictions = self._format_predictions(outputs)
        targets = self._format_targets(batch)
        
        # Update metrics
        for pred, target in zip(predictions, targets):
            pred_mask = pred["semantic_mask"]
            target_mask = target["semantic_mask"]
            
            self.test_dice.update(pred_mask, target_mask)
            self.test_iou.update(pred_mask, target_mask)
            self.test_accuracy.update(pred_mask, target_mask)
            self.test_precision.update(pred_mask, target_mask)
            self.test_recall.update(pred_mask, target_mask)
        
        # Log loss
        self.log("test_loss", outputs["loss"], prog_bar=True)
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Log metrics
        self.log("test_dice", self.test_dice.compute(), prog_bar=True)
        self.log("test_iou", self.test_iou.compute(), prog_bar=True)
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)
        
        # Reset metrics
        self.test_dice.reset()
        self.test_iou.reset()
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_classes: int,
        **kwargs
    ) -> "SemanticSegmentationModel":
        """Create a model from a pretrained checkpoint.
        
        Args:
            model_name: Name of the pretrained model
            num_classes: Number of classes
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        config = SemanticSegmentationModelConfig(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
        return cls(config)
    
    def on_train_epoch_start(self) -> None:
        """Called at the start of training epoch."""
        # Set model to training mode
        self.model.train()
    
    def on_validation_epoch_start(self) -> None:
        """Called at the start of validation epoch."""
        # Set model to evaluation mode
        self.model.eval()
    
    def on_test_epoch_start(self) -> None:
        """Called at the start of test epoch."""
        # Set model to evaluation mode
        self.model.eval()
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving a checkpoint."""
        # Add model configuration to checkpoint
        checkpoint["model_config"] = self.config.__dict__
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        # Load model configuration from checkpoint
        if "model_config" in checkpoint:
            self.config = SemanticSegmentationModelConfig(**checkpoint["model_config"]) 
"""Detection task — registers itself with the central TaskRegistry."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoConfig, AutoModelForObjectDetection

from ...config.schema import PipelineConfig, ModelConfig
from ...registry import TaskRegistry
from .adapters import get_input_adapter, get_output_adapter
from .collate import detection_collate_fn
from .data import COCODetectionDataset


@TaskRegistry.register("detection")
class DetectionTask:
    """Provides everything the TrainingEngine needs for detection training."""

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def get_model(self, model_cfg: ModelConfig) -> AutoModelForObjectDetection:
        hf_config = AutoConfig.from_pretrained(
            model_cfg.model_name,
            num_labels=model_cfg.num_classes,
        )
        model = AutoModelForObjectDetection.from_pretrained(
            model_cfg.model_name,
            config=hf_config,
            ignore_mismatched_sizes=True,
        )
        model.config.confidence_threshold = model_cfg.confidence_threshold
        model.config.iou_threshold = model_cfg.iou_threshold
        model.config.max_detections = model_cfg.max_detections
        return model

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def get_train_dataset(self, config: PipelineConfig) -> COCODetectionDataset:
        adapter = self.get_input_adapter(config.model)
        return COCODetectionDataset(
            root_dir=config.data.train_data_path,
            annotation_file=config.data.train_annotation_file,
            transform=adapter,
        )

    def get_val_dataset(self, config: PipelineConfig) -> COCODetectionDataset:
        adapter = self.get_input_adapter(config.model)
        return COCODetectionDataset(
            root_dir=config.data.val_data_path,
            annotation_file=config.data.val_annotation_file,
            transform=adapter,
        )

    # ------------------------------------------------------------------
    # Adapters
    # ------------------------------------------------------------------
    def get_input_adapter(self, model_cfg: ModelConfig):
        return get_input_adapter(model_cfg.model_name, image_size=model_cfg.image_size_scalar)

    def get_output_adapter(self, model_cfg: ModelConfig):
        return get_output_adapter(model_cfg.model_name, image_size=model_cfg.image_size_scalar)

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------
    def get_collate_fn(self):
        return detection_collate_fn

    # ------------------------------------------------------------------
    # Optimizer + scheduler (mirrors model.py:configure_optimizers)
    # ------------------------------------------------------------------
    def create_optimizer_and_scheduler(
        self,
        model: torch.nn.Module,
        config: PipelineConfig,
        num_training_steps: int,
    ) -> Tuple[AdamW, Optional[Any]]:
        lr = config.model.learning_rate
        wd = config.model.weight_decay

        # Separate backbone vs head params
        if hasattr(model, "backbone") or hasattr(model, "model") and hasattr(model.model, "backbone"):
            backbone_params = []
            other_params = []
            for name, param in model.named_parameters():
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)

            param_groups = [
                {"params": backbone_params, "lr": lr * 0.1, "weight_decay": wd},
                {"params": other_params, "lr": lr, "weight_decay": wd},
            ]
        else:
            param_groups = [{"params": model.parameters(), "lr": lr, "weight_decay": wd}]

        optimizer = AdamW(param_groups)

        if config.model.scheduler == "cosine":
            warmup_steps = int(num_training_steps * 0.1)

            warmup = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(warmup_steps, 1),
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=max(num_training_steps - warmup_steps, 1), eta_min=1e-6,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
            return optimizer, scheduler

        return optimizer, None

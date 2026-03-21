"""Segmentation task — registers itself with the central TaskRegistry."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor
from transformers.trainer_utils import EvalLoopOutput

from ...config.schema import PipelineConfig, ModelConfig
from ...registry import TaskRegistry
from ...utils.coco import COCOAnnotationType, detect_annotation_type
from .adapters import (
    detect_segmentation_family,
    get_input_adapter,
    postprocess_semantic,
)
from .collate import segmentation_collate_fn
from .data import (
    COCOInstanceSegmentationDataset,
    COCOPanopticSegmentationDataset,
    SemanticSegmentationDataset,
)


@TaskRegistry.register("segmentation")
class SegmentationTask:
    """Provides everything the TrainingEngine needs for segmentation training.

    Supports two model families transparently:
      - **Semantic** (SegFormer, UperNet, BEiT, DPT): labels are ``(H, W)``
        class-index tensors; model uses cross-entropy loss.
      - **Universal** (Mask2Former, MaskFormer, OneFormer): labels are
        per-instance ``mask_labels`` + ``class_labels``; model uses Hungarian
        matching + dice + cross-entropy loss.
    """

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def get_model(self, model_cfg: ModelConfig) -> torch.nn.Module:
        _, family_cfg = detect_segmentation_family(model_cfg.model_name)

        hf_config = AutoConfig.from_pretrained(
            model_cfg.model_name,
            num_labels=model_cfg.num_classes,
        )

        if family_cfg.model_type == "universal":
            from transformers import AutoModelForUniversalSegmentation

            model = AutoModelForUniversalSegmentation.from_pretrained(
                model_cfg.model_name,
                config=hf_config,
                ignore_mismatched_sizes=True,
            )
        else:
            from transformers import AutoModelForSemanticSegmentation

            model = AutoModelForSemanticSegmentation.from_pretrained(
                model_cfg.model_name,
                config=hf_config,
                ignore_mismatched_sizes=True,
            )
        return model

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def get_train_dataset(self, config: PipelineConfig):
        return self._build_dataset(
            data_path=config.data.train_data_path,
            annotation_file=config.data.train_annotation_file,
            model_name=config.model.model_name,
            image_size=config.model.image_size_scalar,
        )

    def get_val_dataset(self, config: PipelineConfig):
        return self._build_dataset(
            data_path=config.data.val_data_path,
            annotation_file=config.data.val_annotation_file,
            model_name=config.model.model_name,
            image_size=config.model.image_size_scalar,
        )

    def _build_dataset(
        self,
        data_path: str,
        annotation_file: Optional[str],
        model_name: str,
        image_size: int,
    ):
        """Select dataset format by auto-detecting the annotation type.

        Resolution order (COCO-first):
          1. ``instances_*.json`` → :class:`COCOInstanceSegmentationDataset`
             (uses ``pycocotools.annToMask``; same file used for detection)
          2. ``panoptic_*.json``  → :class:`COCOPanopticSegmentationDataset`
             (RGB panoptic PNGs + segments_info)
          3. No annotation file   → :class:`SemanticSegmentationDataset`
             (ADE20K-style images + mask PNGs fallback)
        """
        adapter = get_input_adapter(model_name, image_size=image_size)

        if annotation_file:
            ann_type = detect_annotation_type(annotation_file)

            if ann_type == COCOAnnotationType.INSTANCES:
                return COCOInstanceSegmentationDataset(
                    image_dir=data_path,
                    annotation_file=annotation_file,
                    transform=adapter,
                )
            else:
                return COCOPanopticSegmentationDataset(
                    image_dir=data_path,
                    annotation_file=annotation_file,
                    transform=adapter,
                )

        return SemanticSegmentationDataset(
            root_dir=data_path,
            transform=adapter,
        )

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------
    def get_collate_fn(self) -> Callable:
        return segmentation_collate_fn

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    def create_optimizer_and_scheduler(
        self,
        model: torch.nn.Module,
        config: PipelineConfig,
        num_training_steps: int,
    ) -> Tuple[AdamW, Optional[Any]]:
        lr = config.model.learning_rate
        wd = config.model.weight_decay

        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if "backbone" in name or "encoder" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * 0.1, "weight_decay": wd},
            {"params": head_params, "lr": lr, "weight_decay": wd},
        ]

        optimizer = AdamW(param_groups)

        if config.model.scheduler == "cosine":
            warmup_steps = int(num_training_steps * 0.1)
            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=max(warmup_steps, 1),
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(num_training_steps - warmup_steps, 1),
                eta_min=1e-6,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
            return optimizer, scheduler

        return optimizer, None

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(model, inputs, return_outputs=False):
        if "mask_labels" in inputs:
            # Universal segmentation (Mask2Former, MaskFormer, OneFormer)
            outputs = model(
                pixel_values=inputs["pixel_values"],
                mask_labels=inputs["mask_labels"],
                class_labels=inputs["class_labels"],
            )
        else:
            # Semantic segmentation (SegFormer, UperNet, BEiT, DPT)
            outputs = model(
                pixel_values=inputs["pixel_values"],
                labels=inputs["labels"],
            )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def get_eval_fn(self, model_cfg: ModelConfig) -> Callable:
        """Return a closure that computes mIoU metrics."""
        _, family_cfg = detect_segmentation_family(model_cfg.model_name)
        processor = AutoImageProcessor.from_pretrained(model_cfg.model_name)

        def _eval_fn(
            *,
            model,
            dataloader: DataLoader,
            args,
            metric_key_prefix: str = "eval",
        ) -> EvalLoopOutput:
            device = args.device
            num_classes = model_cfg.num_classes

            total_loss = 0.0
            num_batches = 0

            # Per-class intersection and union accumulators
            intersection = torch.zeros(num_classes, dtype=torch.long)
            union = torch.zeros(num_classes, dtype=torch.long)

            for inputs in dataloader:
                pixel_values = inputs["pixel_values"].to(device)

                # Build forward kwargs based on model type
                fwd_kwargs: Dict[str, Any] = {"pixel_values": pixel_values}
                if "mask_labels" in inputs:
                    fwd_kwargs["mask_labels"] = [
                        m.to(device) for m in inputs["mask_labels"]
                    ]
                    fwd_kwargs["class_labels"] = [
                        c.to(device) for c in inputs["class_labels"]
                    ]
                elif "labels" in inputs:
                    fwd_kwargs["labels"] = inputs["labels"].to(device)

                with torch.no_grad():
                    outputs = model(**fwd_kwargs)

                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                num_batches += 1

                # Produce per-pixel predictions
                if family_cfg.model_type == "universal":
                    pred_maps = processor.post_process_semantic_segmentation(
                        outputs,
                        target_sizes=[
                            (pixel_values.shape[2], pixel_values.shape[3])
                        ] * pixel_values.shape[0],
                    )
                    pred_maps = [p.cpu() for p in pred_maps]
                else:
                    logits = outputs.logits.cpu()
                    target_h, target_w = pixel_values.shape[2], pixel_values.shape[3]
                    pred_maps = postprocess_semantic(
                        logits, [(target_h, target_w)],
                    )

                # Ground-truth masks
                if "labels" in inputs:
                    gt_masks = inputs["labels"]
                    if isinstance(gt_masks, torch.Tensor):
                        gt_list = [gt_masks[i].cpu() for i in range(gt_masks.shape[0])]
                    else:
                        gt_list = [g.cpu() if isinstance(g, torch.Tensor) else g for g in gt_masks]
                elif "mask_labels" in inputs:
                    gt_list = _universal_labels_to_semantic(
                        inputs["mask_labels"], inputs["class_labels"], num_classes,
                    )
                else:
                    continue

                for pred, gt in zip(pred_maps, gt_list):
                    pred_flat = pred.long().flatten()
                    gt_flat = gt.long().flatten()
                    valid = gt_flat < num_classes
                    pred_flat = pred_flat[valid]
                    gt_flat = gt_flat[valid]

                    for cls in range(num_classes):
                        pred_mask = pred_flat == cls
                        gt_mask = gt_flat == cls
                        intersection[cls] += (pred_mask & gt_mask).sum().item()
                        union[cls] += (pred_mask | gt_mask).sum().item()

            # Compute IoU per class
            iou_per_class = torch.zeros(num_classes)
            active_classes = 0
            for cls in range(num_classes):
                if union[cls] > 0:
                    iou_per_class[cls] = intersection[cls].float() / union[cls].float()
                    active_classes += 1

            mean_iou = iou_per_class.sum().item() / max(active_classes, 1)

            metrics: Dict[str, float] = {}
            metrics[f"{metric_key_prefix}_loss"] = total_loss / max(num_batches, 1)
            metrics[f"{metric_key_prefix}_miou"] = mean_iou

            # Per-class IoU
            for cls in range(num_classes):
                if union[cls] > 0:
                    metrics[f"{metric_key_prefix}_iou_class_{cls}"] = float(
                        iou_per_class[cls].item()
                    )

            # Pixel accuracy
            total_correct = intersection.sum().item()
            total_pixels = union.sum().item()
            if total_pixels > 0:
                pixel_acc = total_correct / total_pixels
                metrics[f"{metric_key_prefix}_pixel_accuracy"] = pixel_acc

            return EvalLoopOutput(
                predictions=None,
                label_ids=None,
                metrics=metrics,
                num_samples=num_batches * (dataloader.batch_size or 1),
            )

        return _eval_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _universal_labels_to_semantic(
    mask_labels: List[torch.Tensor],
    class_labels: List[torch.Tensor],
    num_classes: int,
) -> List[torch.Tensor]:
    """Convert per-instance masks + class labels to semantic class maps."""
    results = []
    for masks, classes in zip(mask_labels, class_labels):
        if masks.dim() == 3:
            h, w = masks.shape[1], masks.shape[2]
        else:
            h, w = masks.shape[-2], masks.shape[-1]

        semantic_map = torch.full((h, w), fill_value=num_classes, dtype=torch.long)
        for i, cls_id in enumerate(classes):
            cls_int = int(cls_id.item()) if isinstance(cls_id, torch.Tensor) else int(cls_id)
            if cls_int < num_classes:
                semantic_map[masks[i].bool()] = cls_int
        results.append(semantic_map)
    return results

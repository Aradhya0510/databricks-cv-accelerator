"""Custom HF Trainer for object detection with torchmetrics evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput

from torchmetrics.detection import MeanAveragePrecision


class DetectionTrainer(Trainer):
    """``transformers.Trainer`` subclass with detection-specific loss and eval.

    Attributes:
        output_adapter: An adapter instance (e.g. ``DETROutputAdapter``) set
            by the engine after construction.
    """

    output_adapter: Any = None  # set externally by TrainingEngine

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Evaluation loop override
    # ------------------------------------------------------------------
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Override to compute detection mAP metrics using torchmetrics."""

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        metric = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", class_metrics=True,
        )
        device = self.args.device

        total_loss = 0.0
        num_batches = 0

        for step, inputs in enumerate(dataloader):
            # Move pixel_values to device; labels are list-of-dicts
            pixel_values = inputs["pixel_values"].to(device)
            labels_on_device = []
            for lbl in inputs["labels"]:
                labels_on_device.append(
                    {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in lbl.items()}
                )

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, labels=labels_on_device)

            if outputs.loss is not None:
                total_loss += outputs.loss.item()
            num_batches += 1

            # Adapt outputs and format for metrics
            adapted = self.output_adapter.adapt_output(outputs)
            batch_dict = {"pixel_values": pixel_values, "labels": labels_on_device}
            preds = self.output_adapter.format_predictions(adapted, batch_dict)
            targets = self.output_adapter.format_targets(labels_on_device)

            # Move to CPU for torchmetrics
            preds_cpu = [
                {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in p.items()}
                for p in preds
            ]
            targets_cpu = [
                {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
            metric.update(preds=preds_cpu, target=targets_cpu)

        # Compute aggregate metrics.
        # torchmetrics tries distributed all_gather during compute() when
        # torch.distributed is initialized (DDP).  The metric lives on CPU
        # and there is no CPU distributed backend, so we temporarily patch
        # is_initialized to return False for the compute() call.
        _orig_is_init = torch.distributed.is_initialized
        torch.distributed.is_initialized = lambda: False
        try:
            map_metrics = metric.compute()
        finally:
            torch.distributed.is_initialized = _orig_is_init

        metrics: Dict[str, float] = {}
        avg_loss = total_loss / max(num_batches, 1)
        metrics[f"{metric_key_prefix}_loss"] = avg_loss

        # Standard COCO metrics
        for key in [
            "map", "map_50", "map_75",
            "map_small", "map_medium", "map_large",
            "mar_1", "mar_10", "mar_100",
            "mar_small", "mar_medium", "mar_large",
        ]:
            val = map_metrics.get(key)
            if val is not None:
                metrics[f"{metric_key_prefix}_{key}"] = float(val.item()) if isinstance(val, torch.Tensor) else float(val)

        # Per-class metrics
        if "classes" in map_metrics:
            for i, class_id in enumerate(map_metrics["classes"]):
                cid = int(class_id)
                if map_metrics["map_per_class"][i] != -1:
                    metrics[f"{metric_key_prefix}_map_class_{cid}"] = float(map_metrics["map_per_class"][i])
                if map_metrics["mar_100_per_class"][i] != -1:
                    metrics[f"{metric_key_prefix}_mar_100_class_{cid}"] = float(map_metrics["mar_100_per_class"][i])

        self.log(metrics)

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_batches * dataloader.batch_size if dataloader.batch_size else num_batches,
        )

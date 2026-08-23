"""Detection task — registers itself with the central TaskRegistry."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForObjectDetection
from transformers.trainer_utils import EvalLoopOutput

from ...config.schema import PipelineConfig, ModelConfig
from ...registry import TaskRegistry
from ..augmentation import build_augmentations
from ...utils.distributed import all_gather_objects
from ...utils.hf import MODEL_DTYPE
from ...utils.labels import apply_label_names
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
            dtype=MODEL_DTYPE,
        )
        model.config.confidence_threshold = model_cfg.confidence_threshold
        model.config.iou_threshold = model_cfg.iou_threshold
        model.config.max_detections = model_cfg.max_detections
        apply_label_names(model, model_cfg.class_names, model_cfg.num_classes)
        return model

    def get_processor(self, model_cfg: ModelConfig):
        """The image processor that must be logged alongside the model.

        Serving has to preprocess exactly the way training did, so this is the
        same processor the input adapter builds.
        """
        return self.get_input_adapter(model_cfg).processor

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def get_train_dataset(self, config: PipelineConfig) -> COCODetectionDataset:
        adapter = self.get_input_adapter(config.model)
        return COCODetectionDataset(
            root_dir=config.data.train_data_path,
            annotation_file=config.data.train_annotation_file,
            transform=adapter,
            # Augmentation applies to the training split only — the validation
            # set must stay fixed for metrics to be comparable across epochs.
            augmentations=build_augmentations(
                config.data.augment,
                config.data.augmentations,
                task_type="detection",
                image_size=config.model.image_size_scalar,
            ),
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

    def get_output_adapter(self, model_cfg: ModelConfig, score_threshold: float = 0.0):
        """Output adapter for this model.

        Defaults to an unfiltered score threshold because that is what mAP
        needs; callers doing inference should pass
        ``model_cfg.confidence_threshold``.
        """
        return get_output_adapter(
            model_cfg.model_name,
            image_size=model_cfg.image_size_scalar,
            score_threshold=score_threshold,
        )

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------
    def get_collate_fn(self) -> Callable:
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

    # ------------------------------------------------------------------
    # Loss (called by CVTrainer.compute_loss)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(model, inputs, return_outputs=False):
        outputs = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Evaluation (called by CVTrainer.evaluation_loop)
    # ------------------------------------------------------------------
    def get_eval_fn(self, model_cfg: ModelConfig) -> Callable:
        """Return a closure that computes detection mAP metrics.

        The returned callable has signature::

            (model, dataloader, args, metric_key_prefix) -> EvalLoopOutput
        """
        output_adapter = self.get_output_adapter(model_cfg)

        def _eval_fn(
            *,
            model,
            dataloader: DataLoader,
            args,
            metric_key_prefix: str = "eval",
        ) -> EvalLoopOutput:
            from torchmetrics.detection import MeanAveragePrecision

            # sync_on_compute=False is the supported way to keep this CPU-side
            # metric from attempting a distributed all-gather.  The previous
            # approach reassigned torch.distributed.is_initialized globally,
            # which is unsafe under any concurrency.
            metric = MeanAveragePrecision(
                box_format="xyxy", iou_type="bbox", class_metrics=True,
                sync_on_compute=False,
            )
            device = args.device

            total_loss = 0.0
            num_batches = 0

            for step, inputs in enumerate(dataloader):
                pixel_values = inputs["pixel_values"].to(device)
                labels_on_device = [
                    {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in lbl.items()}
                    for lbl in inputs["labels"]
                ]

                with torch.no_grad():
                    outputs = model(pixel_values=pixel_values, labels=labels_on_device)

                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                num_batches += 1

                adapted = output_adapter.adapt_output(outputs)
                batch_dict = {"pixel_values": pixel_values, "labels": labels_on_device}
                preds = output_adapter.format_predictions(adapted, batch_dict)
                targets = output_adapter.format_targets(labels_on_device)

                preds_cpu = [
                    {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in p.items()}
                    for p in preds
                ]
                targets_cpu = [
                    {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                    for t in targets
                ]
                # Under DDP each rank holds a shard of the validation set, so
                # predictions are gathered before the metric sees them --
                # otherwise mAP would cover only ~1/world_size of the data.
                gathered_preds = [
                    p for chunk in all_gather_objects(preds_cpu) for p in chunk
                ]
                gathered_targets = [
                    t for chunk in all_gather_objects(targets_cpu) for t in chunk
                ]
                metric.update(preds=gathered_preds, target=gathered_targets)

            map_metrics = metric.compute()

            metrics: Dict[str, float] = {}
            metrics[f"{metric_key_prefix}_loss"] = total_loss / max(num_batches, 1)

            for key in [
                "map", "map_50", "map_75",
                "map_small", "map_medium", "map_large",
                "mar_1", "mar_10", "mar_100",
                "mar_small", "mar_medium", "mar_large",
            ]:
                val = map_metrics.get(key)
                if val is not None:
                    metrics[f"{metric_key_prefix}_{key}"] = (
                        float(val.item()) if isinstance(val, torch.Tensor) else float(val)
                    )

            if "classes" in map_metrics:
                for i, class_id in enumerate(map_metrics["classes"]):
                    cid = int(class_id)
                    if map_metrics["map_per_class"][i] != -1:
                        metrics[f"{metric_key_prefix}_map_class_{cid}"] = float(
                            map_metrics["map_per_class"][i]
                        )
                    if map_metrics["mar_100_per_class"][i] != -1:
                        metrics[f"{metric_key_prefix}_mar_100_class_{cid}"] = float(
                            map_metrics["mar_100_per_class"][i]
                        )

            return EvalLoopOutput(
                predictions=None,
                label_ids=None,
                metrics=metrics,
                num_samples=num_batches * dataloader.batch_size if dataloader.batch_size else num_batches,
            )

        return _eval_fn

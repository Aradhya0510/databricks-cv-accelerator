"""Classification task — registers itself with the central TaskRegistry."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
from transformers.trainer_utils import EvalLoopOutput

from ...config.schema import PipelineConfig, ModelConfig
from ...registry import TaskRegistry
from .data import ImageFolderClassificationDataset
from .collate import classification_collate_fn


@TaskRegistry.register("classification")
class ClassificationTask:
    """Provides everything the TrainingEngine needs for image classification."""

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def get_model(self, model_cfg: ModelConfig) -> AutoModelForImageClassification:
        hf_config = AutoConfig.from_pretrained(
            model_cfg.model_name,
            num_labels=model_cfg.num_classes,
        )
        # Apply classification-specific config
        if hasattr(hf_config, "hidden_dropout_prob"):
            hf_config.hidden_dropout_prob = model_cfg.dropout
        if hasattr(hf_config, "classifier_dropout"):
            hf_config.classifier_dropout = model_cfg.dropout

        model = AutoModelForImageClassification.from_pretrained(
            model_cfg.model_name,
            config=hf_config,
            ignore_mismatched_sizes=True,
        )
        return model

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def get_train_dataset(self, config: PipelineConfig) -> ImageFolderClassificationDataset:
        processor = self._get_processor(config.model)
        return ImageFolderClassificationDataset(
            root_dir=config.data.train_data_path,
            processor=processor,
            class_names=config.model.class_names,
        )

    def get_val_dataset(self, config: PipelineConfig) -> ImageFolderClassificationDataset:
        processor = self._get_processor(config.model)
        return ImageFolderClassificationDataset(
            root_dir=config.data.val_data_path,
            processor=processor,
            class_names=config.model.class_names,
        )

    # ------------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------------
    def _get_processor(self, model_cfg: ModelConfig) -> AutoImageProcessor:
        image_size = model_cfg.image_size_scalar
        return AutoImageProcessor.from_pretrained(
            model_cfg.model_name,
            size={"height": image_size, "width": image_size},
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
        )

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------
    def get_collate_fn(self) -> Callable:
        return classification_collate_fn

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

        # Separate backbone vs classifier params for differential LR
        backbone_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if "classifier" in name or "head" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * 0.1, "weight_decay": wd},
            {"params": classifier_params, "lr": lr, "weight_decay": wd},
        ]

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
    # Loss
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
    # Evaluation
    # ------------------------------------------------------------------
    def get_eval_fn(self, model_cfg: ModelConfig) -> Callable:
        """Return a closure that computes classification metrics."""

        def _eval_fn(
            *,
            model,
            dataloader: DataLoader,
            args,
            metric_key_prefix: str = "eval",
        ) -> EvalLoopOutput:
            device = args.device

            total_loss = 0.0
            num_batches = 0
            all_preds = []
            all_labels = []

            for inputs in dataloader:
                pixel_values = inputs["pixel_values"].to(device)
                labels = inputs["labels"].to(device)

                with torch.no_grad():
                    outputs = model(pixel_values=pixel_values, labels=labels)

                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                num_batches += 1

                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            num_samples = len(all_preds)

            # Accuracy
            correct = (all_preds == all_labels).sum().item()
            accuracy = correct / max(num_samples, 1)

            # Per-class precision, recall, F1
            num_classes = model_cfg.num_classes
            metrics: Dict[str, float] = {}
            metrics[f"{metric_key_prefix}_loss"] = total_loss / max(num_batches, 1)
            metrics[f"{metric_key_prefix}_accuracy"] = accuracy

            precisions = []
            recalls = []
            f1s = []

            for c in range(num_classes):
                tp = ((all_preds == c) & (all_labels == c)).sum().item()
                fp = ((all_preds == c) & (all_labels != c)).sum().item()
                fn = ((all_preds != c) & (all_labels == c)).sum().item()

                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

                # Only store per-class metrics if this class exists in the data
                if tp + fp + fn > 0:
                    metrics[f"{metric_key_prefix}_precision_class_{c}"] = precision
                    metrics[f"{metric_key_prefix}_recall_class_{c}"] = recall
                    metrics[f"{metric_key_prefix}_f1_class_{c}"] = f1

            # Macro averages
            active_classes = [i for i in range(num_classes) if
                             ((all_preds == i) | (all_labels == i)).any()]
            if active_classes:
                metrics[f"{metric_key_prefix}_precision"] = sum(
                    precisions[i] for i in active_classes
                ) / len(active_classes)
                metrics[f"{metric_key_prefix}_recall"] = sum(
                    recalls[i] for i in active_classes
                ) / len(active_classes)
                metrics[f"{metric_key_prefix}_f1"] = sum(
                    f1s[i] for i in active_classes
                ) / len(active_classes)

            return EvalLoopOutput(
                predictions=all_preds.numpy(),
                label_ids=all_labels.numpy(),
                metrics=metrics,
                num_samples=num_samples,
            )

        return _eval_fn

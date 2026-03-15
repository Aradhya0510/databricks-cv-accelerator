"""EvaluationEngine — wraps task eval_fn, adds error analysis + benchmarking."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..config.schema import PipelineConfig
from ..registry import TaskRegistry


class EvaluationEngine:
    """Standalone evaluation: mAP metrics, error analysis, and latency benchmarks."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        # Ensure tasks are registered
        import src.tasks.detection  # noqa: F401
        import src.tasks.classification  # noqa: F401

        self.task = TaskRegistry.get(config.model.task_type)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(
        self,
        model_path: Optional[str] = None,
        run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
    ) -> torch.nn.Module:
        """Load a model from a local path, model URI, or MLflow run.

        Preferred resolution order: *model_uri* → *run_id* → *model_path*.
        """
        if model_uri is not None:
            import mlflow

            return mlflow.transformers.load_model(model_uri)

        if run_id is not None:
            import mlflow

            # Try model URI stored as a run param (MLflow 3 path)
            try:
                client = mlflow.MlflowClient()
                run = client.get_run(run_id)
                stored_uri = run.data.params.get("logged_model_uri")
                if stored_uri:
                    return mlflow.transformers.load_model(stored_uri)
            except Exception:
                pass

            # Fallback: runs:/ URI (deprecated in MLflow 3 but still functional)
            return mlflow.transformers.load_model(f"runs:/{run_id}/model")

        if model_path is not None:
            task_type = self.config.model.task_type
            if task_type == "detection":
                from transformers import AutoModelForObjectDetection

                model = AutoModelForObjectDetection.from_pretrained(model_path)
                model.config.confidence_threshold = self.config.model.confidence_threshold
                model.config.iou_threshold = self.config.model.iou_threshold
                model.config.max_detections = self.config.model.max_detections
            elif task_type == "classification":
                from transformers import AutoModelForImageClassification

                model = AutoModelForImageClassification.from_pretrained(model_path)
            else:
                model = self.task.get_model(self.config.model)
            return model

        # Fall back to creating a fresh model from config (useful for checkpoint dirs)
        return self.task.get_model(self.config.model)

    def _get_val_dataloader(self) -> DataLoader:
        val_ds = self.task.get_val_dataset(self.config)
        return DataLoader(
            val_ds,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=self.task.get_collate_fn(),
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # evaluate()
    # ------------------------------------------------------------------
    def evaluate(
        self,
        model_path: Optional[str] = None,
        run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
        max_batches: Optional[int] = None,
    ) -> dict:
        """Compute mAP metrics on the validation set.

        Reuses the task's ``get_eval_fn()`` for metric computation.
        """
        model = self._load_model(model_path, run_id, model_uri=model_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        dataloader = self._get_val_dataloader()
        if max_batches is not None:
            dataloader = _limit_dataloader(dataloader, max_batches)

        # Build a lightweight args namespace for the eval_fn
        args = _SimpleNamespace(device=device)
        eval_fn = self.task.get_eval_fn(self.config.model)

        output = eval_fn(
            model=model,
            dataloader=dataloader,
            args=args,
            metric_key_prefix="eval",
        )

        metrics = output.metrics
        self._save_results(metrics, "evaluation_metrics.json")
        return metrics

    # ------------------------------------------------------------------
    # error_analysis()
    # ------------------------------------------------------------------
    def error_analysis(
        self,
        model_path: Optional[str] = None,
        run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
        max_batches: int = 100,
    ) -> dict:
        """Categorise predictions into error types.

        For detection: TP, FP (background / confusion / localisation), FN.
        For classification: correct, misclassified (with confusion matrix).
        """
        task_type = self.config.model.task_type
        if task_type == "classification":
            return self._error_analysis_classification(model_path, run_id, max_batches, model_uri=model_uri)
        return self._error_analysis_detection(model_path, run_id, max_batches, model_uri=model_uri)

    def _error_analysis_classification(
        self,
        model_path: Optional[str],
        run_id: Optional[str],
        max_batches: int,
        model_uri: Optional[str] = None,
    ) -> dict:
        """Classification error analysis: confusion matrix + per-class accuracy."""
        model = self._load_model(model_path, run_id, model_uri=model_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        dataloader = self._get_val_dataloader()
        num_classes = self.config.model.num_classes

        all_preds = []
        all_labels = []

        for step, inputs in enumerate(dataloader):
            if step >= max_batches:
                break

            pixel_values = inputs["pixel_values"].to(device)
            labels = inputs["labels"].to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)

            preds = outputs.logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        total = len(all_preds)

        correct = (all_preds == all_labels).sum().item()
        misclassified = total - correct

        # Confusion matrix
        confusion: Dict[str, Dict[str, int]] = {}
        for pred, gt in zip(all_preds.tolist(), all_labels.tolist()):
            gt_key = str(gt)
            pred_key = str(pred)
            if gt_key not in confusion:
                confusion[gt_key] = {}
            confusion[gt_key][pred_key] = confusion[gt_key].get(pred_key, 0) + 1

        # Most confused pairs
        confused_pairs = []
        for gt_class, pred_counts in confusion.items():
            for pred_class, count in pred_counts.items():
                if gt_class != pred_class and count > 0:
                    confused_pairs.append({
                        "true_class": int(gt_class),
                        "predicted_class": int(pred_class),
                        "count": count,
                    })
        confused_pairs.sort(key=lambda x: x["count"], reverse=True)

        result = {
            "summary": {
                "total_samples": total,
                "correct": correct,
                "misclassified": misclassified,
                "accuracy": correct / max(total, 1),
            },
            "confusion_matrix": confusion,
            "most_confused_pairs": confused_pairs[:20],
        }
        self._save_results(result, "error_analysis.json")
        return result

    def _error_analysis_detection(
        self,
        model_path: Optional[str],
        run_id: Optional[str],
        max_batches: int,
        model_uri: Optional[str] = None,
    ) -> dict:
        """Detection error analysis: TP, FP (background/confusion/localisation), FN."""
        model = self._load_model(model_path, run_id, model_uri=model_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        dataloader = self._get_val_dataloader()
        output_adapter = self.task.get_output_adapter(self.config.model)

        iou_threshold = self.config.model.iou_threshold
        stats: Dict[str, int] = {
            "true_positives": 0,
            "false_positives_background": 0,
            "false_positives_confusion": 0,
            "false_positives_localisation": 0,
            "false_negatives": 0,
            "total_predictions": 0,
            "total_ground_truths": 0,
        }
        per_class_errors: Dict[int, Dict[str, int]] = {}

        for step, inputs in enumerate(dataloader):
            if step >= max_batches:
                break

            pixel_values = inputs["pixel_values"].to(device)
            labels_on_device = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in lbl.items()}
                for lbl in inputs["labels"]
            ]

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, labels=labels_on_device)

            adapted = output_adapter.adapt_output(outputs)
            batch_dict = {"pixel_values": pixel_values, "labels": labels_on_device}
            preds = output_adapter.format_predictions(adapted, batch_dict)
            targets = output_adapter.format_targets(labels_on_device)

            for pred, target in zip(preds, targets):
                pred_boxes = pred["boxes"].cpu()
                pred_labels = pred["labels"].cpu()
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                stats["total_predictions"] += len(pred_boxes)
                stats["total_ground_truths"] += len(gt_boxes)

                matched_gt = set()

                for pi in range(len(pred_boxes)):
                    if len(gt_boxes) == 0:
                        stats["false_positives_background"] += 1
                        continue

                    ious = _box_iou(pred_boxes[pi].unsqueeze(0), gt_boxes).squeeze(0)
                    best_iou, best_idx = ious.max(0) if ious.numel() > 0 else (torch.tensor(0.0), torch.tensor(0))
                    best_idx = int(best_idx)

                    if best_iou >= iou_threshold and best_idx not in matched_gt:
                        if pred_labels[pi] == gt_labels[best_idx]:
                            stats["true_positives"] += 1
                            matched_gt.add(best_idx)
                        else:
                            stats["false_positives_confusion"] += 1
                            _inc(per_class_errors, int(gt_labels[best_idx]), "confusion")
                    elif best_iou >= 0.1:
                        stats["false_positives_localisation"] += 1
                    else:
                        stats["false_positives_background"] += 1

                stats["false_negatives"] += len(gt_boxes) - len(matched_gt)

        result = {
            "summary": stats,
            "per_class_errors": per_class_errors,
        }
        self._save_results(result, "error_analysis.json")
        return result

    # ------------------------------------------------------------------
    # benchmark()
    # ------------------------------------------------------------------
    def benchmark(
        self,
        model_path: Optional[str] = None,
        run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
        num_warmup: int = 10,
        num_batches: int = 100,
    ) -> dict:
        """Measure inference throughput and latency."""
        model = self._load_model(model_path, run_id, model_uri=model_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        dataloader = self._get_val_dataloader()
        batch_iter = iter(dataloader)

        # Warm up
        for _ in range(num_warmup):
            try:
                inputs = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                inputs = next(batch_iter)

            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                model(pixel_values=pixel_values)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed run
        latencies = []
        total_images = 0

        for _ in range(num_batches):
            try:
                inputs = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                inputs = next(batch_iter)

            pixel_values = inputs["pixel_values"].to(device)
            batch_size = pixel_values.shape[0]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                model(pixel_values=pixel_values)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append(t1 - t0)
            total_images += batch_size

        latencies_sorted = sorted(latencies)
        total_time = sum(latencies)

        result = {
            "fps": total_images / total_time if total_time > 0 else 0,
            "total_images": total_images,
            "total_time_s": total_time,
            "latency_per_batch_ms": {
                "mean": (total_time / len(latencies)) * 1000,
                "p50": latencies_sorted[len(latencies) // 2] * 1000,
                "p95": latencies_sorted[int(len(latencies) * 0.95)] * 1000,
                "p99": latencies_sorted[int(len(latencies) * 0.99)] * 1000,
            },
            "device": str(device),
        }

        if device.type == "cuda":
            result["gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        self._save_results(result, "benchmark.json")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _save_results(self, data: dict, filename: str) -> None:
        results_dir = self.config.output.results_dir
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Results saved to {path}")


class _SimpleNamespace:
    """Lightweight stand-in for HF TrainingArguments — only carries ``device``."""

    def __init__(self, device: torch.device):
        self.device = device


def _limit_dataloader(dataloader: DataLoader, max_batches: int):
    """Yield at most *max_batches* from *dataloader*."""

    class _Limited:
        def __init__(self, dl, n):
            self._dl = dl
            self._n = n
            self.batch_size = dl.batch_size

        def __iter__(self):
            for i, batch in enumerate(self._dl):
                if i >= self._n:
                    break
                yield batch

    return _Limited(dataloader, max_batches)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes in xyxy format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def _inc(d: dict, class_id: int, error_type: str) -> None:
    if class_id not in d:
        d[class_id] = {}
    d[class_id][error_type] = d[class_id].get(error_type, 0) + 1

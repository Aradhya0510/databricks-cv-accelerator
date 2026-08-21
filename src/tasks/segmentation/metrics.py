"""Segmentation metric accumulation.

Extracted from the evaluation closure so the arithmetic can be checked against
hand-computed expectations — which is how the pixel-accuracy denominator bug
below was found.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch


class SegmentationMetrics:
    """Accumulate per-class intersection/union plus a true pixel count.

    Pixel accuracy needs its own numerator and denominator.  Deriving it from
    the IoU accumulators as ``intersection.sum() / union.sum()`` is wrong: the
    per-class unions count every *correct* pixel once but every *incorrect*
    pixel twice — once in the predicted class's union and once in the true
    class's — so the denominator is ``correct + 2 x incorrect`` rather than the
    number of pixels, and accuracy is under-reported by a margin that grows as
    the model gets worse.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes, dtype=torch.long)
        self.union = torch.zeros(num_classes, dtype=torch.long)
        self.correct_pixels = 0
        self.total_pixels = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Add one prediction/target pair of per-pixel class-index maps."""
        pred_flat = pred.reshape(-1).long()
        target_flat = target.reshape(-1).long()

        # Drop ignore/void pixels — anything outside the label range.
        valid = (target_flat >= 0) & (target_flat < self.num_classes)
        pred_flat = pred_flat[valid]
        target_flat = target_flat[valid]

        if pred_flat.numel() == 0:
            return

        self.correct_pixels += int((pred_flat == target_flat).sum().item())
        self.total_pixels += int(pred_flat.numel())

        for cls in range(self.num_classes):
            pred_mask = pred_flat == cls
            target_mask = target_flat == cls
            self.intersection[cls] += int((pred_mask & target_mask).sum().item())
            self.union[cls] += int((pred_mask | target_mask).sum().item())

    def reduce_across_ranks(self) -> None:
        """Sum the accumulators over every rank.

        Under DDP each rank sees a shard of the validation set, so without
        this the reported mIoU covers only ~1/world_size of the data.
        """
        from ...utils.distributed import all_reduce_sum, is_distributed

        if not is_distributed():
            return

        self.intersection = all_reduce_sum(self.intersection)
        self.union = all_reduce_sum(self.union)

        counts = torch.tensor([self.correct_pixels, self.total_pixels], dtype=torch.long)
        counts = all_reduce_sum(counts)
        self.correct_pixels = int(counts[0].item())
        self.total_pixels = int(counts[1].item())

    def compute(self, prefix: str = "eval") -> Dict[str, float]:
        """Return mIoU, per-class IoU, and pixel accuracy."""
        self.reduce_across_ranks()

        metrics: Dict[str, float] = {}

        iou_sum = 0.0
        active = 0
        for cls in range(self.num_classes):
            if self.union[cls] > 0:
                iou = float(self.intersection[cls]) / float(self.union[cls])
                metrics[f"{prefix}_iou_class_{cls}"] = iou
                iou_sum += iou
                active += 1

        metrics[f"{prefix}_miou"] = iou_sum / active if active else 0.0

        if self.total_pixels > 0:
            metrics[f"{prefix}_pixel_accuracy"] = self.correct_pixels / self.total_pixels

        return metrics

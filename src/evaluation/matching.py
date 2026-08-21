"""Greedy prediction/ground-truth matching for detection error analysis.

Extracted from ``EvaluationEngine`` so the assignment rules can be checked
against hand-built boxes.  Standard COCO-style assignment consumes predictions
in descending score order, so the most confident detection claims each ground
truth; iterating in raw model-output order let a weak box steal a match from a
better one and shifted counts between the TP / localisation / confusion
buckets that the whole report is about.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

# Below this IoU a false positive is "background" rather than a near miss.
LOCALISATION_IOU_FLOOR = 0.1


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of xyxy boxes -> ``[len(b1), len(b2)]``."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: Optional[torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[Dict[str, int], List[Tuple[int, str]]]:
    """Classify each prediction as TP or a flavour of FP, and count FNs.

    Returns ``(counts, confusions)`` where *counts* has the keys
    ``true_positives``, ``false_positives_background``,
    ``false_positives_confusion``, ``false_positives_localisation`` and
    ``false_negatives``, and *confusions* lists ``(true_class, "confusion")``
    pairs for per-class reporting.
    """
    counts = {
        "true_positives": 0,
        "false_positives_background": 0,
        "false_positives_confusion": 0,
        "false_positives_localisation": 0,
        "false_negatives": 0,
    }
    confusions: List[Tuple[int, str]] = []

    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]

    if num_preds == 0:
        counts["false_negatives"] = num_gts
        return counts, confusions

    # Highest-confidence predictions get first claim on each ground truth.
    if pred_scores is not None and pred_scores.numel() == num_preds:
        order = torch.argsort(pred_scores, descending=True).tolist()
    else:
        order = list(range(num_preds))

    if num_gts == 0:
        counts["false_positives_background"] = num_preds
        return counts, confusions

    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()

    for pi in order:
        row = ious[pi]
        best_iou, best_idx = row.max(0)
        best_iou = float(best_iou)
        best_idx = int(best_idx)

        if best_iou >= iou_threshold and best_idx not in matched_gt:
            if int(pred_labels[pi]) == int(gt_labels[best_idx]):
                counts["true_positives"] += 1
                matched_gt.add(best_idx)
            else:
                counts["false_positives_confusion"] += 1
                confusions.append((int(gt_labels[best_idx]), "confusion"))
        elif best_iou >= LOCALISATION_IOU_FLOOR:
            counts["false_positives_localisation"] += 1
        else:
            counts["false_positives_background"] += 1

    counts["false_negatives"] = num_gts - len(matched_gt)
    return counts, confusions

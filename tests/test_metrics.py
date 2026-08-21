"""Metric arithmetic, checked against hand-computed values.

Every expectation here is worked out by hand in the comments so a future
change that shifts a denominator shows up as a concrete number mismatch
rather than a plausible-looking metric.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.matching import box_iou, match_predictions  # noqa: E402
from src.tasks.segmentation.metrics import SegmentationMetrics  # noqa: E402


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_perfect_prediction_scores_one():
    m = SegmentationMetrics(num_classes=2)
    target = torch.tensor([[0, 0], [1, 1]])
    m.update(target.clone(), target)

    out = m.compute()
    assert out["eval_miou"] == pytest.approx(1.0)
    assert out["eval_pixel_accuracy"] == pytest.approx(1.0)


def test_pixel_accuracy_counts_pixels_not_class_unions():
    """The bug: pixel accuracy was intersection.sum() / union.sum().

    pred = [0, 0, 0, 1]
    gt   = [0, 0, 1, 0]
    Positions 0 and 1 agree; 2 and 3 do not.  True accuracy is 2/4 = 0.5.

    The old formula gave:
      class 0: pred={0,1,2}, gt={0,1,3} -> inter=2, union={0,1,2,3}=4
      class 1: pred={3},     gt={2}     -> inter=0, union={2,3}=2
      intersection.sum()=2, union.sum()=6  ->  2/6 = 0.333
    because each wrong pixel is counted in *both* class unions, inflating
    the denominator to correct + 2 x incorrect.
    """
    m = SegmentationMetrics(num_classes=2)
    pred = torch.tensor([0, 0, 0, 1])
    gt = torch.tensor([0, 0, 1, 0])
    m.update(pred, gt)

    out = m.compute()
    assert out["eval_pixel_accuracy"] == pytest.approx(0.5)
    # The old, wrong denominator would have reported this instead.
    assert out["eval_pixel_accuracy"] != pytest.approx(2 / 6)


def test_iou_per_class_is_correct():
    # class 0: pred {0,1,2}, gt {0,1,3} -> inter 2, union 4 -> 0.5
    # class 1: pred {3},     gt {2}     -> inter 0, union 2 -> 0.0
    # miou = (0.5 + 0.0) / 2 = 0.25
    m = SegmentationMetrics(num_classes=2)
    m.update(torch.tensor([0, 0, 0, 1]), torch.tensor([0, 0, 1, 0]))

    out = m.compute()
    assert out["eval_iou_class_0"] == pytest.approx(0.5)
    assert out["eval_iou_class_1"] == pytest.approx(0.0)
    assert out["eval_miou"] == pytest.approx(0.25)


def test_classes_absent_from_the_data_are_excluded_from_miou():
    """A class with an empty union must not drag mIoU toward zero."""
    m = SegmentationMetrics(num_classes=5)
    target = torch.tensor([0, 0, 1, 1])
    m.update(target.clone(), target)

    out = m.compute()
    assert out["eval_miou"] == pytest.approx(1.0)
    assert "eval_iou_class_4" not in out


def test_ignore_index_pixels_are_dropped():
    """255 (void) must not count as a class or as a wrong pixel."""
    m = SegmentationMetrics(num_classes=2)
    m.update(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 255, 255]))

    out = m.compute()
    assert m.total_pixels == 2
    assert out["eval_pixel_accuracy"] == pytest.approx(1.0)


def test_accumulates_across_batches():
    m = SegmentationMetrics(num_classes=2)
    m.update(torch.tensor([0, 0]), torch.tensor([0, 0]))   # 2/2 correct
    m.update(torch.tensor([1, 1]), torch.tensor([0, 0]))   # 0/2 correct

    assert m.compute()["eval_pixel_accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Detection matching
# ---------------------------------------------------------------------------

def test_iou_of_identical_boxes_is_one():
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert box_iou(b, b)[0, 0] == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    assert box_iou(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_half_overlap():
    # a: 10x10 = 100, b: 10x10 = 100, intersection 5x10 = 50, union = 150
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
    assert box_iou(a, b)[0, 0] == pytest.approx(50 / 150)


def test_exact_match_is_a_true_positive():
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    counts, _ = match_predictions(
        box, torch.tensor([1]), torch.tensor([0.9]), box.clone(), torch.tensor([1]),
    )
    assert counts["true_positives"] == 1
    assert counts["false_negatives"] == 0


def test_right_place_wrong_class_is_a_confusion():
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    counts, confusions = match_predictions(
        box, torch.tensor([2]), torch.tensor([0.9]), box.clone(), torch.tensor([1]),
    )
    assert counts["false_positives_confusion"] == 1
    assert counts["true_positives"] == 0
    # The ground truth stays unmatched, so it is also a miss.
    assert counts["false_negatives"] == 1
    assert confusions == [(1, "confusion")]


def test_near_miss_is_a_localisation_error():
    # IoU 50/150 = 0.33: above the 0.1 localisation floor, below 0.5.
    pred = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    counts, _ = match_predictions(pred, torch.tensor([1]), torch.tensor([0.9]), gt, torch.tensor([1]))
    assert counts["false_positives_localisation"] == 1


def test_far_away_prediction_is_a_background_error():
    pred = torch.tensor([[90.0, 90.0, 100.0, 100.0]])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    counts, _ = match_predictions(pred, torch.tensor([1]), torch.tensor([0.9]), gt, torch.tensor([1]))
    assert counts["false_positives_background"] == 1


def test_highest_scoring_prediction_claims_the_ground_truth():
    """The ordering bug: without sorting, the first-listed box wins.

    Two predictions for one ground truth.  The exact match scores 0.4 and is
    listed second; the sloppy box scores 0.9 and is listed first.  Sorting by
    score means the 0.9 box is assigned first — and since it only reaches
    IoU 0.33 it is a localisation error, leaving the exact box to match.
    """
    preds = torch.tensor([
        [5.0, 0.0, 15.0, 10.0],   # sloppy, high score
        [0.0, 0.0, 10.0, 10.0],   # exact, lower score
    ])
    scores = torch.tensor([0.9, 0.4])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

    counts, _ = match_predictions(
        preds, torch.tensor([1, 1]), scores, gt, torch.tensor([1]),
    )
    assert counts["true_positives"] == 1
    assert counts["false_positives_localisation"] == 1
    assert counts["false_negatives"] == 0


def test_one_ground_truth_cannot_be_matched_twice():
    """Two perfect duplicates: one TP, one FP — never two TPs."""
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    preds = torch.cat([box, box])
    counts, _ = match_predictions(
        preds, torch.tensor([1, 1]), torch.tensor([0.9, 0.8]), box, torch.tensor([1]),
    )
    assert counts["true_positives"] == 1
    assert sum(v for k, v in counts.items() if k.startswith("false_positives")) == 1


def test_no_predictions_makes_every_ground_truth_a_miss():
    counts, _ = match_predictions(
        torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long), torch.zeros(0),
        torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        torch.tensor([1, 2]),
    )
    assert counts["false_negatives"] == 2
    assert counts["true_positives"] == 0


def test_no_ground_truth_makes_every_prediction_a_background_error():
    counts, _ = match_predictions(
        torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([1]), torch.tensor([0.9]),
        torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long),
    )
    assert counts["false_positives_background"] == 1
    assert counts["false_negatives"] == 0

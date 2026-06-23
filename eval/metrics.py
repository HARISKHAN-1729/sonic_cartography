"""
eval/metrics.py
───────────────
Evaluation metrics for the Predictive Sonic Cartography pipeline.

Three primary metrics (as proposed in the IROS LBR abstract):

1. Object Hit Rate (OHR)
   ─────────────────────
   For each predicted frame at horizon H, we check: did the object
   predicted to be there (at azimuth θ and distance d) actually appear
   in the real ground-truth frame at approximately the same location?
   Measured as precision and recall over all (clip, horizon) pairs.

2. Distance MAE (Mean Absolute Error)
   ─────────────────────────────────
   For matched object pairs (predicted ↔ GT), compute |d_pred - d_gt|.
   Reported in metres, per-class and aggregated.

3. Accuracy vs. Lead Time curve
   ─────────────────────────────
   OHR and MAE computed separately at each horizon (0.5, 1.0, ..., 2.5s).
   The curve shows the operating window where prediction is reliable
   enough to be sonically useful.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from models.object_detector import Detection


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class MatchedPair:
    label: str
    dist_pred_m: float
    dist_gt_m: float
    azimuth_pred_deg: float
    azimuth_gt_deg: float
    horizon_sec: float


@dataclass
class HorizonMetrics:
    horizon_sec: float
    n_samples: int
    precision: float          # TP / (TP + FP)
    recall: float             # TP / (TP + FN)
    f1: float                 # harmonic mean of P & R
    distance_mae_m: float     # mean |d_pred - d_gt| over matched pairs
    azimuth_mae_deg: float    # mean |az_pred - az_gt| over matched pairs
    per_class_mae: dict = field(default_factory=dict)


# ── Core metric functions ──────────────────────────────────────────────────

def compute_object_hit_rate(
    pred_detections: List[Detection],
    gt_detections: List[Detection],
    iou_threshold: float = 0.5,
    azimuth_tolerance_deg: float = 20.0,
    depth_tolerance_m: float = 2.0,
) -> tuple[float, float, List[MatchedPair]]:
    """
    Compute precision & recall between predicted and GT detections.

    Matching criterion (since we don't have GT bboxes in predicted frames):
    A prediction TP matches a GT object if:
      - Same label (or close synonym)
      - |azimuth_pred - azimuth_gt| ≤ azimuth_tolerance_deg
      - |depth_pred - depth_gt| ≤ depth_tolerance_m

    Parameters
    ----------
    pred_detections : detections from the predicted frame
    gt_detections   : detections from the real future frame
    iou_threshold   : (legacy, kept for API compat; spatial tolerances used)
    azimuth_tolerance_deg : max angular error for a match
    depth_tolerance_m     : max depth error for a match

    Returns
    -------
    precision, recall, list of matched pairs
    """
    if not gt_detections:
        precision = 1.0 if not pred_detections else 0.0
        return precision, 1.0, []

    if not pred_detections:
        return 0.0, 0.0, []

    matched_gt = set()
    matched_pairs = []
    tp = 0

    # Greedy match: for each prediction, find best GT
    for pred in pred_detections:
        best_gt = None
        best_score = float("inf")

        for j, gt in enumerate(gt_detections):
            if j in matched_gt:
                continue
            if not _labels_match(pred.label, gt.label):
                continue

            az_err = abs(pred.azimuth_deg - gt.azimuth_deg)
            d_err  = abs(pred.depth_m - gt.depth_m)

            if (az_err <= azimuth_tolerance_deg and
                    d_err <= depth_tolerance_m):
                score = az_err + d_err  # combined error for tie-breaking
                if score < best_score:
                    best_score = score
                    best_gt = (j, gt)

        if best_gt is not None:
            j, gt = best_gt
            matched_gt.add(j)
            tp += 1
            matched_pairs.append(MatchedPair(
                label=pred.label,
                dist_pred_m=pred.depth_m,
                dist_gt_m=gt.depth_m,
                azimuth_pred_deg=pred.azimuth_deg,
                azimuth_gt_deg=gt.azimuth_deg,
                horizon_sec=0.0,  # filled in by caller
            ))

    precision = tp / len(pred_detections) if pred_detections else 0.0
    recall    = tp / len(gt_detections)   if gt_detections   else 1.0
    return float(precision), float(recall), matched_pairs


def compute_distance_mae(matched_pairs: List[MatchedPair]) -> float:
    """Mean absolute error of depth predictions in metres."""
    if not matched_pairs:
        return float("nan")
    errors = [abs(p.dist_pred_m - p.dist_gt_m) for p in matched_pairs]
    return float(np.mean(errors))


def compute_azimuth_mae(matched_pairs: List[MatchedPair]) -> float:
    """Mean absolute error of azimuth predictions in degrees."""
    if not matched_pairs:
        return float("nan")
    errors = [abs(p.azimuth_pred_deg - p.azimuth_gt_deg) for p in matched_pairs]
    return float(np.mean(errors))


def compute_per_class_mae(
    matched_pairs: List[MatchedPair],
) -> dict[str, float]:
    """Distance MAE broken down by object class."""
    from collections import defaultdict
    class_errors: dict = defaultdict(list)
    for p in matched_pairs:
        class_errors[p.label].append(abs(p.dist_pred_m - p.dist_gt_m))
    return {
        label: float(np.mean(errs))
        for label, errs in class_errors.items()
    }


def aggregate_horizon_metrics(
    horizon_sec: float,
    all_precisions: List[float],
    all_recalls: List[float],
    all_matched_pairs: List[MatchedPair],
) -> HorizonMetrics:
    """Aggregate per-clip metrics into a single HorizonMetrics object."""
    mean_p   = float(np.nanmean(all_precisions)) if all_precisions else 0.0
    mean_r   = float(np.nanmean(all_recalls))    if all_recalls    else 0.0
    f1 = (2 * mean_p * mean_r / (mean_p + mean_r + 1e-9))

    for pair in all_matched_pairs:
        pair.horizon_sec = horizon_sec

    d_mae  = compute_distance_mae(all_matched_pairs)
    az_mae = compute_azimuth_mae(all_matched_pairs)
    per_cls = compute_per_class_mae(all_matched_pairs)

    return HorizonMetrics(
        horizon_sec=horizon_sec,
        n_samples=len(all_precisions),
        precision=mean_p,
        recall=mean_r,
        f1=f1,
        distance_mae_m=d_mae,
        azimuth_mae_deg=az_mae,
        per_class_mae=per_cls,
    )


# ── Utility ────────────────────────────────────────────────────────────────

# Label synonyms for matching (handles spaces vs underscores, plurals, etc.)
_LABEL_SYNONYMS = {
    "traffic_light": {"traffic light", "stoplight", "signal"},
    "stop_sign":     {"stop sign"},
    "fire_hydrant":  {"fire hydrant", "hydrant"},
    "trash_can":     {"trash can", "garbage can", "waste bin", "dustbin"},
}


def _labels_match(a: str, b: str) -> bool:
    """Check if two label strings refer to the same category."""
    a = a.lower().strip()
    b = b.lower().strip()
    if a == b:
        return True
    # Check synonym groups
    for canonical, synonyms in _LABEL_SYNONYMS.items():
        group = synonyms | {canonical}
        if a in group and b in group:
            return True
    return False


def print_metrics_table(metrics_by_horizon: List[HorizonMetrics]) -> None:
    """Pretty-print the accuracy-vs-lead-time table."""
    print(f"\n{'─'*72}")
    print(f"{'Horizon':>10}  {'N':>6}  {'Prec':>6}  {'Rec':>6}  "
          f"{'F1':>6}  {'DepthMAE(m)':>12}  {'AzMAE(°)':>10}")
    print(f"{'─'*72}")
    for m in sorted(metrics_by_horizon, key=lambda x: x.horizon_sec):
        print(
            f"{m.horizon_sec:>9.1f}s  "
            f"{m.n_samples:>6}  "
            f"{m.precision:>6.3f}  "
            f"{m.recall:>6.3f}  "
            f"{m.f1:>6.3f}  "
            f"{m.distance_mae_m:>12.3f}  "
            f"{m.azimuth_mae_deg:>10.2f}"
        )
    print(f"{'─'*72}\n")

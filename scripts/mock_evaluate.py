"""
scripts/mock_evaluate.py
─────────────────────────
Runs the FULL evaluation pipeline logic using synthetic (mock) data —
no GPU, no pretrained models, no SANPO dataset needed.

Useful for:
  - Validating the evaluation loop and metric aggregation on any machine
  - Debugging the pipeline without waiting for model downloads
  - CI / unit-level integration testing

The mock models produce plausible outputs:
  - NWMWrapper  → random RGB frames (with controlled degradation over horizon)
  - DepthEstimator → synthetic depth maps (gaussian distance field)
  - ObjectDetector → random detection lists drawn from a fixed scene prior

Usage:
    python scripts/mock_evaluate.py
    python scripts/mock_evaluate.py --n_clips 50 --plot
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.object_detector import Detection
from audio.scene_sonifier import SceneSonifier
from audio.earcon_bank import EarconBank
from eval.metrics import (
    compute_object_hit_rate,
    aggregate_horizon_metrics,
    print_metrics_table,
)


# ── Mock model definitions ────────────────────────────────────────────────

class MockNWM:
    """
    Fake NWM: returns slightly-noised versions of the last context frame.
    Noise increases with horizon to simulate degrading prediction quality.
    """
    def predict_at_horizons(
        self,
        context_frames: list,
        horizon_times_sec: list,
        action: str = "walk_forward",
    ) -> dict:
        base = context_frames[-1].astype(np.float32)
        result = {}
        for h in horizon_times_sec:
            # Noise sigma increases with horizon: 0→5, 2.5→40
            sigma = 5.0 + h * 14.0
            noise = np.random.normal(0, sigma, base.shape).astype(np.float32)
            pred = np.clip(base + noise, 0, 255).astype(np.uint8)
            result[h] = pred
        return result


class MockDepthEstimator:
    """
    Fake depth estimator: generates a plausible depth field.
    Objects lower in the frame (higher y) are treated as closer.
    """
    def estimate(self, frame) -> tuple:
        h, w = frame.shape[:2] if hasattr(frame, 'shape') else (256, 256)
        # Depth increases toward top of frame (far = sky)
        y_coords = np.linspace(10.0, 0.5, h)[:, None] * np.ones((1, w))
        # Add small random perturbation
        noise = np.random.uniform(0.8, 1.2, (h, w))
        depth_m = (y_coords * noise).astype(np.float32)
        depth_m = np.clip(depth_m, 0.1, 10.0)
        depth_norm = 1.0 - depth_m / depth_m.max()
        return depth_m, depth_norm.astype(np.float32)


OBJECT_POOL = [
    ("person",   2.0,  15.0),
    ("car",      5.0, -20.0),
    ("curb",     1.0,   0.0),
    ("bench",    3.5,  30.0),
    ("pole",     1.5, -10.0),
    ("stairs",   2.5,   5.0),
    ("bicycle",  4.0, -35.0),
    ("wall",     6.0,  45.0),
]


class MockObjectDetector:
    """
    Fake detector: returns a subset of the OBJECT_POOL with added noise.
    horizon_noise controls how much the detections drift (simulates
    NWM prediction degradation translating to detection error).
    """
    def __init__(self, horizon_noise_m: float = 0.0):
        self.horizon_noise_m = horizon_noise_m

    def detect(self, frame, depth_m=None) -> list[Detection]:
        # Randomly drop ~30% of objects (simulates missed detections)
        n_keep = max(1, int(len(OBJECT_POOL) * np.random.uniform(0.6, 1.0)))
        selected = np.random.choice(len(OBJECT_POOL), n_keep, replace=False)

        dets = []
        for i in selected:
            label, base_depth, base_az = OBJECT_POOL[i]
            # Add noise proportional to horizon
            d = base_depth + np.random.normal(0, 0.3 + self.horizon_noise_m)
            az = base_az  + np.random.normal(0, 3.0 + self.horizon_noise_m * 5)
            d = max(0.2, d)
            dets.append(Detection(
                label=label,
                confidence=float(np.random.uniform(0.5, 0.95)),
                bbox=(10, 10, 110, 110),
                depth_m=float(d),
                azimuth_deg=float(az),
            ))
        dets.sort(key=lambda x: x.depth_m)
        return dets


# ── Mock SANPO clip ────────────────────────────────────────────────────────

def make_mock_clip(clip_id: str, n_frames: int = 60):
    """Return a list of synthetic RGB frames."""
    frames = [
        np.random.randint(60, 200, (256, 320, 3), dtype=np.uint8)
        for _ in range(n_frames)
    ]
    return frames


# ── Main mock evaluation loop ─────────────────────────────────────────────

def run_mock_eval(
    n_clips: int = 100,
    horizons_sec: list = None,
    seed: int = 42,
) -> list:
    """
    Run mock evaluation and return list of HorizonMetrics.
    """
    if horizons_sec is None:
        horizons_sec = [0.5, 1.0, 1.5, 2.0, 2.5]

    np.random.seed(seed)

    nwm       = MockNWM()
    depth_est = MockDepthEstimator()
    sonifier  = SceneSonifier()

    # Horizon noise model: prediction degrades linearly with horizon
    # At 0.5s: noise=0.2m, at 2.5s: noise=1.5m
    def horizon_noise(h):
        return 0.1 + h * 0.56

    horizon_accum = {
        h: {"precisions": [], "recalls": [], "pairs": []}
        for h in horizons_sec
    }

    t0 = time.time()

    for clip_idx in range(n_clips):
        # Create synthetic clip
        context_frames = make_mock_clip(f"clip_{clip_idx:04d}")[:8]

        # Predict future frames
        predicted = nwm.predict_at_horizons(
            context_frames=context_frames,
            horizon_times_sec=horizons_sec,
        )

        for h_sec in horizons_sec:
            pred_frame = predicted[h_sec]

            # Parse predicted frame (with horizon-dependent noise)
            pred_depth_m, _ = depth_est.estimate(pred_frame)
            pred_detector    = MockObjectDetector(horizon_noise_m=horizon_noise(h_sec))
            pred_dets        = pred_detector.detect(pred_frame, depth_m=pred_depth_m)

            # Parse GT frame (clean: noise=0)
            gt_frame         = make_mock_clip(f"clip_{clip_idx:04d}_gt")[0]
            gt_depth_m, _    = depth_est.estimate(gt_frame)
            gt_detector      = MockObjectDetector(horizon_noise_m=0.0)
            gt_dets          = gt_detector.detect(gt_frame, depth_m=gt_depth_m)

            # Metrics
            prec, rec, pairs = compute_object_hit_rate(pred_dets, gt_dets)
            horizon_accum[h_sec]["precisions"].append(prec)
            horizon_accum[h_sec]["recalls"].append(rec)
            horizon_accum[h_sec]["pairs"].extend(pairs)

            # Sanity check: also sonify (tests audio pipeline end-to-end)
            _ = sonifier.sonify(pred_dets)

    elapsed = time.time() - t0

    # Aggregate
    horizon_metrics = []
    for h in horizons_sec:
        acc = horizon_accum[h]
        m = aggregate_horizon_metrics(
            horizon_sec=h,
            all_precisions=acc["precisions"],
            all_recalls=acc["recalls"],
            all_matched_pairs=acc["pairs"],
        )
        horizon_metrics.append(m)

    print(f"\nMock evaluation: {n_clips} clips × {len(horizons_sec)} horizons "
          f"in {elapsed:.2f}s ({elapsed/n_clips*1000:.1f}ms/clip)")
    print_metrics_table(horizon_metrics)

    return horizon_metrics


def plot_mock_results(horizon_metrics: list, out_dir: Path):
    """Quick matplotlib plot of mock results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not installed; skipping plot.")
        return

    horizons = [m.horizon_sec for m in horizon_metrics]
    f1s      = [m.f1           for m in horizon_metrics]
    d_maes   = [m.distance_mae_m for m in horizon_metrics]
    az_maes  = [m.azimuth_mae_deg for m in horizon_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: F1 vs horizon
    ax = axes[0]
    ax.plot(horizons, f1s, "o-", color="#534AB7", lw=2.2, ms=7, label="F1")
    ax.set_xlabel("Prediction horizon (s)")
    ax.set_ylabel("Object hit rate (F1)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(horizons)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fs"))
    ax.set_title("Accuracy vs. lead time (mock data)")
    ax2 = ax.twinx()
    ax2.plot(horizons, d_maes, "D-.", color="#D85A30", lw=1.8, ms=5,
             label="Depth MAE")
    ax2.set_ylabel("Distance MAE (m)", color="#D85A30")
    ax2.tick_params(axis="y", labelcolor="#D85A30")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    # Right: Azimuth MAE vs horizon
    ax = axes[1]
    ax.plot(horizons, az_maes, "s-", color="#185FA5", lw=2.0, ms=7)
    ax.fill_between(horizons, az_maes, alpha=0.12, color="#185FA5")
    ax.axhline(15.0, color="#E24B4A", ls="--", lw=1.2,
               label="15° threshold (spatial audio)")
    ax.set_xlabel("Prediction horizon (s)")
    ax.set_ylabel("Azimuth MAE (°)")
    ax.set_title("Directional accuracy vs. lead time")
    ax.set_xticks(horizons)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fs"))
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "mock_eval_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_clips", type=int, default=100)
    p.add_argument("--horizons", type=float, nargs="+",
                   default=[0.5, 1.0, 1.5, 2.0, 2.5])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--output", type=str, default="eval/mock_results.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("\nPredictive Sonic Cartography — Mock Evaluation")
    print("=" * 55)
    print(f"  n_clips  : {args.n_clips}")
    print(f"  horizons : {args.horizons}s")
    print("=" * 55)

    metrics = run_mock_eval(
        n_clips=args.n_clips,
        horizons_sec=args.horizons,
        seed=args.seed,
    )

    # Save results JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "mode": "mock",
        "n_clips": args.n_clips,
        "horizons": [
            {
                "horizon_sec": m.horizon_sec,
                "n_samples": m.n_samples,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "distance_mae_m": m.distance_mae_m,
                "azimuth_mae_deg": m.azimuth_mae_deg,
                "per_class_mae": m.per_class_mae,
            }
            for m in metrics
        ],
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    if args.plot:
        plot_mock_results(metrics, out_path.parent)

    # Final sanity assertions
    f1s = [m.f1 for m in metrics]
    assert f1s[0] >= f1s[-1], \
        "F1 should degrade with horizon (mock model is noisy)"
    d_maes = [m.distance_mae_m for m in metrics]
    assert d_maes[-1] >= d_maes[0], \
        "Distance MAE should increase with horizon"
    print("\n  ✓ Sanity checks passed (F1 degrades, MAE grows with horizon)")


if __name__ == "__main__":
    main()

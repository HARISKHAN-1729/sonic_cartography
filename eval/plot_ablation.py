"""
eval/plot_ablation.py
──────────────────────
Overlay F1-vs-lead-time (and depth MAE) for multiple backends on one figure —
the core ablation plot for the abstract.

Usage
─────
    python eval/plot_ablation.py \
        --results eval/results_last_frame.json eval/results_flow.json \
                  eval/results_nwm.json eval/results_vjepa.json \
        --labels last_frame flow nwm vjepa \
        --output_dir eval/figures/

Any subset works (e.g. just flow + vjepa).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLORS = ["#9AA0A6", "#0F6E56", "#534AB7", "#D85A30", "#185FA5", "#BA7517"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load(path):
    with open(path) as f:
        data = json.load(f)
    hs = sorted(data["horizons"], key=lambda x: x["horizon_sec"])
    return ([h["horizon_sec"] for h in hs],
            [h["f1"] for h in hs],
            [h["distance_mae_m"] for h in hs],
            [h["azimuth_mae_deg"] for h in hs])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output_dir", default="eval/figures/")
    args = p.parse_args()
    assert len(args.results) == len(args.labels), "results and labels must match"

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ── F1 vs horizon (main ablation) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (res, lab) in enumerate(zip(args.results, args.labels)):
        x, f1, _, _ = load(res)
        ax.plot(x, f1, marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)], linewidth=2.2, markersize=7, label=lab)
    ax.set_xlabel("Prediction horizon (seconds)")
    ax.set_ylabel("Object hit rate (F1)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fs"))
    ax.axhspan(0.0, 0.5, alpha=0.05, color="red")
    ax.set_title("World-model ablation: predictive accuracy vs. lead time")
    ax.legend(title="backend", fontsize=10)
    plt.tight_layout()
    f1_path = out / "ablation_f1_vs_horizon.png"
    fig.savefig(f1_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(f1_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {f1_path}")

    # ── Depth MAE vs horizon ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (res, lab) in enumerate(zip(args.results, args.labels)):
        x, _, dmae, _ = load(res)
        ax.plot(x, dmae, marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)], linewidth=2.0, markersize=6, label=lab)
    ax.set_xlabel("Prediction horizon (seconds)")
    ax.set_ylabel("Distance MAE (m)")
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fs"))
    ax.set_title("World-model ablation: depth error vs. lead time")
    ax.legend(title="backend", fontsize=10)
    plt.tight_layout()
    d_path = out / "ablation_depth_mae.png"
    fig.savefig(d_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(d_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {d_path}")
    print("\n  Ablation figures done.")


if __name__ == "__main__":
    main()

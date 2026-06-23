"""
eval/plot_results.py
─────────────────────
Generates the key figures for the IROS LBR abstract:

  Figure 1: Accuracy vs. Lead Time (main result curve)
    - Object Hit Rate (F1) vs. horizon (0.5 → 2.5s)
    - Distance MAE vs. horizon (second y-axis)

  Figure 2: Per-class Distance MAE bar chart
    - Error breakdown by object class

  Figure 3: Precision-Recall scatter by horizon

Usage:
    python eval/plot_results.py \
        --results eval/results.json \
        --output_dir eval/figures/
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
COLORS = {
    "f1":           "#534AB7",   # purple — main metric
    "mae":          "#D85A30",   # coral — depth error
    "precision":    "#0F6E56",   # teal
    "recall":       "#BA7517",   # amber
}
FIG_SIZE = (8, 4)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    horizons_data = results["horizons"]
    horizons_data.sort(key=lambda x: x["horizon_sec"])

    horizons = [h["horizon_sec"] for h in horizons_data]
    f1s       = [h["f1"]              for h in horizons_data]
    precs     = [h["precision"]       for h in horizons_data]
    recs      = [h["recall"]          for h in horizons_data]
    d_maes    = [h["distance_mae_m"]  for h in horizons_data]
    az_maes   = [h["azimuth_mae_deg"] for h in horizons_data]

    # ── Figure 1: Accuracy vs Lead Time (main paper figure) ───────────────
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)

    ax1.plot(horizons, f1s, "o-", color=COLORS["f1"],
             linewidth=2.2, markersize=7, label="F1 (object hit rate)", zorder=3)
    ax1.plot(horizons, precs, "s--", color=COLORS["precision"],
             linewidth=1.4, markersize=5, label="Precision", alpha=0.8)
    ax1.plot(horizons, recs, "^--", color=COLORS["recall"],
             linewidth=1.4, markersize=5, label="Recall", alpha=0.8)

    ax1.set_xlabel("Prediction horizon (seconds)", fontsize=12)
    ax1.set_ylabel("Object hit rate", color=COLORS["f1"], fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLORS["f1"])
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(horizons)
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fs"))

    # Second y-axis: distance MAE
    ax2 = ax1.twinx()
    ax2.plot(horizons, d_maes, "D-.", color=COLORS["mae"],
             linewidth=2.0, markersize=6, label="Depth MAE (m)", zorder=2)
    ax2.set_ylabel("Distance MAE (m)", color=COLORS["mae"], fontsize=12)
    ax2.tick_params(axis="y", labelcolor=COLORS["mae"])
    ax2.set_ylim(0, max(d_maes) * 1.4 if d_maes else 4)

    # Shade "useful zone" where F1 > 0.5
    useful_start = next(
        (h for h, f in zip(horizons, f1s) if f >= 0.5), None
    )
    useful_end = next(
        (h for h, f in zip(reversed(horizons), reversed(f1s)) if f >= 0.5),
        None
    )
    if useful_start is not None and useful_end is not None:
        ax1.axvspan(useful_start, useful_end, alpha=0.08,
                    color=COLORS["f1"], label="Useful zone (F1 ≥ 0.5)")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=9, framealpha=0.9)

    ax1.set_title("Predictive accuracy vs. lead time (SANPO test set)",
                  fontsize=12, pad=10)
    plt.tight_layout()

    fig1_path = out_dir / "fig1_accuracy_vs_lead_time.pdf"
    fig.savefig(fig1_path, bbox_inches="tight", dpi=300)
    fig.savefig(str(fig1_path).replace(".pdf", ".png"),
                bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {fig1_path}")

    # ── Figure 2: Per-class Distance MAE bar chart ────────────────────────
    # Use the 1.5s horizon data (default operating point)
    target_horizon = 1.5
    horizon_1_5 = next(
        (h for h in horizons_data if abs(h["horizon_sec"] - target_horizon) < 0.1),
        horizons_data[len(horizons_data) // 2],  # fallback: middle
    )
    per_class = horizon_1_5.get("per_class_mae", {})

    if per_class:
        labels_sorted = sorted(per_class, key=per_class.get)
        maes_sorted   = [per_class[l] for l in labels_sorted]

        fig2, ax = plt.subplots(figsize=(max(6, len(labels_sorted) * 0.55), 4))
        bars = ax.barh(labels_sorted, maes_sorted, color=COLORS["mae"],
                       alpha=0.85, edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, maes_sorted):
            ax.text(
                val + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}m", va="center", ha="left", fontsize=9,
            )

        ax.set_xlabel("Distance MAE (metres)", fontsize=12)
        ax.set_title(
            f"Per-class depth prediction error at {target_horizon}s horizon",
            fontsize=12, pad=10
        )
        ax.set_xlim(0, max(maes_sorted) * 1.25 if maes_sorted else 4)
        ax.grid(axis="x", alpha=0.4)
        plt.tight_layout()

        fig2_path = out_dir / "fig2_per_class_mae.pdf"
        fig2.savefig(fig2_path, bbox_inches="tight", dpi=300)
        fig2.savefig(str(fig2_path).replace(".pdf", ".png"),
                     bbox_inches="tight", dpi=200)
        plt.close(fig2)
        print(f"  Saved: {fig2_path}")

    # ── Figure 3: Azimuth MAE vs Lead Time ────────────────────────────────
    fig3, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(horizons, az_maes, "o-", color="#185FA5",
            linewidth=2.0, markersize=7)
    ax.fill_between(horizons, az_maes, alpha=0.12, color="#185FA5")
    ax.set_xlabel("Prediction horizon (seconds)", fontsize=12)
    ax.set_ylabel("Azimuth MAE (degrees)", fontsize=12)
    ax.set_title("Directional prediction accuracy vs. lead time", fontsize=12, pad=10)
    ax.set_xticks(horizons)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fs"))
    ax.set_ylim(0, max(az_maes) * 1.3 if az_maes else 40)

    # Draw human FoV reference line (±15° = reliable spatial audio zone)
    ax.axhline(15.0, color="#E24B4A", linestyle="--", linewidth=1.2,
               label="15° azimuth threshold\n(reliable spatial audio)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    fig3_path = out_dir / "fig3_azimuth_mae.pdf"
    fig3.savefig(fig3_path, bbox_inches="tight", dpi=300)
    fig3.savefig(str(fig3_path).replace(".pdf", ".png"),
                 bbox_inches="tight", dpi=200)
    plt.close(fig3)
    print(f"  Saved: {fig3_path}")

    print("\n  All figures saved.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="eval/results.json")
    p.add_argument("--output_dir", type=str, default="eval/figures/")
    return p.parse_args()


if __name__ == "__main__":
    main()

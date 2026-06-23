"""
eval/evaluate.py
─────────────────
Main offline evaluation script for Predictive Sonic Cartography.

Runs the full pipeline on the SANPO dataset:
  1. For each clip: sample a context window + future ground-truth frames
  2. Feed context → NWM → predicted frames at each horizon
  3. Parse predicted frames with depth estimator + object detector
  4. Parse real future frames with the same models (GT)
  5. Compute object hit rate, distance MAE, azimuth MAE per horizon
  6. Save results to JSON

Usage:
    python eval/evaluate.py \
        --sanpo_root data/sanpo/ \
        --config configs/default.yaml \
        --output eval/results.json \
        [--max_clips 100]
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

# Make sure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.world_model import build_world_model
from models.depth_estimator import DepthEstimator
from models.object_detector import ObjectDetector
from audio.scene_sonifier import SceneSonifier
from data.sanpo_loader import SANPOLoader
from eval.metrics import (
    compute_object_hit_rate,
    aggregate_horizon_metrics,
    print_metrics_table,
    HorizonMetrics,
    MatchedPair,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline evaluation of Predictive Sonic Cartography"
    )
    p.add_argument("--sanpo_root", type=str, required=True,
                   help="Root directory of SANPO dataset")
    p.add_argument("--config", type=str,
                   default="configs/default.yaml",
                   help="Path to config YAML")
    p.add_argument("--output", type=str,
                   default="eval/results.json",
                   help="Output path for results JSON")
    p.add_argument("--max_clips", type=int, default=-1,
                   help="Max clips to evaluate (-1 = all)")
    p.add_argument("--save_audio", action="store_true",
                   help="Save example audio files alongside results")
    p.add_argument("--horizons", type=float, nargs="+",
                   default=None,
                   help="Override horizon times (seconds)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load config ────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    if args.horizons:
        cfg.prediction.horizons_sec = args.horizons
    if args.max_clips > 0:
        cfg.evaluation.max_clips = args.max_clips

    horizons_sec = list(cfg.prediction.horizons_sec)
    fps_wm = cfg.world_model.prediction_fps
    horizon_frames = {
        sec: int(round(sec * fps_wm))
        for sec in horizons_sec
    }

    print(f"\n{'='*60}")
    print("Predictive Sonic Cartography — Offline Evaluation")
    print(f"{'='*60}")
    print(f"  SANPO root  : {args.sanpo_root}")
    print(f"  Horizons    : {horizons_sec}")
    print(f"  Max clips   : {cfg.evaluation.max_clips}")
    print(f"  Output      : {args.output}")
    print(f"{'='*60}\n")

    # ── Load models ────────────────────────────────────────────────────────
    print("Loading models...")

    world_model = build_world_model(cfg)   # backend chosen by cfg.world_model.backend
    depth_est = DepthEstimator.from_config(cfg.model.depth)
    detector  = ObjectDetector.from_config(cfg.model.detector)
    sonifier  = SceneSonifier.from_config(cfg.audio)

    print(f"  ✓ World model ({cfg.world_model.backend})")
    print("  ✓ Depth Anything V2")
    print("  ✓ YOLO-World")
    print("  ✓ Scene Sonifier\n")

    # ── Load dataset ───────────────────────────────────────────────────────
    loader = SANPOLoader(
        sanpo_root=args.sanpo_root,
        split=cfg.evaluation.sanpo_split,
        max_clips=cfg.evaluation.max_clips,
    )
    print(f"Loaded {len(loader)} clips from SANPO ({cfg.evaluation.sanpo_split} split)\n")

    # ── Accumulate metrics per horizon ────────────────────────────────────
    # Structure: horizon_sec → {precisions, recalls, matched_pairs}
    horizon_accum: dict = {
        h: {"precisions": [], "recalls": [], "pairs": []}
        for h in horizons_sec
    }

    n_audio_saved = 0
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for clip in tqdm(loader.iter_clips(), total=len(loader), desc="Clips"):
        try:
            _evaluate_clip(
                clip=clip,
                loader=loader,
                world_model=world_model,
                depth_est=depth_est,
                detector=detector,
                sonifier=sonifier,
                horizon_frames=horizon_frames,
                horizons_sec=horizons_sec,
                horizon_accum=horizon_accum,
                save_audio=args.save_audio,
                n_audio_saved=n_audio_saved,
                max_audio_saves=cfg.evaluation.save_audio_examples,
                output_dir=output_dir,
            )
        except Exception as e:
            tqdm.write(f"  [SKIP] clip {clip.clip_id}: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete in {elapsed:.1f}s")

    # ── Aggregate metrics ─────────────────────────────────────────────────
    horizon_metrics: List[HorizonMetrics] = []
    for h in horizons_sec:
        acc = horizon_accum[h]
        m = aggregate_horizon_metrics(
            horizon_sec=h,
            all_precisions=acc["precisions"],
            all_recalls=acc["recalls"],
            all_matched_pairs=acc["pairs"],
        )
        horizon_metrics.append(m)

    print_metrics_table(horizon_metrics)

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "n_clips": len(loader),
        "elapsed_sec": elapsed,
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
            for m in horizon_metrics
        ],
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


def _evaluate_clip(
    clip,
    loader,
    world_model,
    depth_est,
    detector,
    sonifier,
    horizon_frames: dict,
    horizons_sec: list,
    horizon_accum: dict,
    save_audio: bool,
    n_audio_saved: int,
    max_audio_saves: int,
    output_dir: Path,
) -> None:
    """Evaluate a single SANPO clip."""
    # Sample context + future ground-truth
    context_sanpo, future_gt = loader.get_context_and_future(
        clip,
        context_len=cfg_context_len(),
        horizon_frames=list(horizon_frames.values()),
    )
    context_rgb = [f.rgb for f in context_sanpo]

    # ── Predict future frames at all horizons ──────────────────────────
    predicted_frames = world_model.predict_at_horizons(
        context_frames=context_rgb,
        horizon_times_sec=horizons_sec,
        action="walk_forward",
    )

    for h_sec in horizons_sec:
        h_frames = horizon_frames[h_sec]

        if h_frames not in future_gt:
            continue  # GT not available (clip too short)

        gt_frame = future_gt[h_frames]
        pred_frame = predicted_frames[h_sec]

        # ── Parse predicted frame ───────────────────────────────────────
        pred_depth_m, _ = depth_est.estimate(pred_frame)
        pred_dets = detector.detect(pred_frame, depth_m=pred_depth_m)

        # ── Parse GT frame ──────────────────────────────────────────────
        gt_depth_m = gt_frame.depth_m  # use SANPO's depth (ground truth)
        if gt_depth_m.max() < 0.01:   # fallback: estimate from model
            gt_depth_m, _ = depth_est.estimate(gt_frame.rgb)
        gt_dets = detector.detect(gt_frame.rgb, depth_m=gt_depth_m)

        # ── Compute metrics ─────────────────────────────────────────────
        prec, rec, pairs = compute_object_hit_rate(
            pred_dets, gt_dets
        )
        horizon_accum[h_sec]["precisions"].append(prec)
        horizon_accum[h_sec]["recalls"].append(rec)
        horizon_accum[h_sec]["pairs"].extend(pairs)

        # ── Optionally save example audio ───────────────────────────────
        if save_audio and n_audio_saved < max_audio_saves and h_sec == 1.5:
            audio = sonifier.sonify(pred_dets)
            audio_path = output_dir / "audio_examples" / \
                f"{clip.clip_id}_h{h_sec:.1f}s.wav"
            sonifier.save(audio, audio_path)
            n_audio_saved += 1


def cfg_context_len() -> int:
    """Return context length from global cfg (set at load time)."""
    return 8  # matches NWM default; parameterise if needed


if __name__ == "__main__":
    main()

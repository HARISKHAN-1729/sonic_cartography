"""
eval/evaluate_unified.py
─────────────────────────
ONE evaluator for every backend. It calls a ScenePredictor (which hides
whether the scene came from pixels or latents) and scores the predicted
future layout against the real future frame.

Works for:
  scene_predictor.type=pixel  + world_model.backend in {flow,nwm,last_frame}
  scene_predictor.type=latent + V-JEPA 2 + a trained probe

Ground truth at each horizon comes from the pixel teacher (Depth Anything +
YOLO-World) applied to the REAL future frame — identical target for every
backend, so the comparison is apples-to-apples.

Usage
─────
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python eval/evaluate_unified.py \
        --config configs/vjepa.yaml \
        --sanpo_root data/sanpo/ \
        --output eval/results_vjepa.json \
        --max_clips 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.scene_predictor import build_scene_predictor
from models.depth_estimator import DepthEstimator
from models.object_detector import ObjectDetector
from data.sanpo_loader import SANPOLoader
from eval.metrics import (
    compute_object_hit_rate, aggregate_horizon_metrics,
    print_metrics_table, HorizonMetrics,
)


def parse_args():
    p = argparse.ArgumentParser(description="Unified PSC evaluation (pixel|latent)")
    p.add_argument("--config", default="configs/vjepa.yaml")
    p.add_argument("--sanpo_root", required=True)
    p.add_argument("--output", default="eval/results_unified.json")
    p.add_argument("--max_clips", type=int, default=-1)
    p.add_argument("--horizons", type=float, nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.horizons:
        cfg.prediction.horizons_sec = args.horizons
    if args.max_clips > 0:
        cfg.evaluation.max_clips = args.max_clips

    horizons = list(cfg.prediction.horizons_sec)
    fps = cfg.world_model.prediction_fps if "world_model" in cfg \
        else cfg.model.vjepa.prediction_fps
    horizon_frames = {h: int(round(h * fps)) for h in horizons}

    backend_name = (cfg.scene_predictor.get("type", "pixel"))
    if backend_name == "pixel":
        backend_name += f"/{cfg.world_model.backend}"

    print(f"\n{'='*60}\nUnified PSC evaluation — backend: {backend_name}\n{'='*60}")
    print(f"  Horizons : {horizons}\n  SANPO    : {args.sanpo_root}\n{'='*60}\n")

    # Predictor under test (pixel or latent) + the shared GT teacher.
    print("Building scene predictor + GT teacher...")
    predictor = build_scene_predictor(cfg)
    gt_depth = DepthEstimator.from_config(cfg.model.depth)
    gt_detector = ObjectDetector.from_config(cfg.model.detector)

    loader = SANPOLoader(sanpo_root=args.sanpo_root,
                         split=cfg.evaluation.get("sanpo_split", "test"),
                         max_clips=cfg.evaluation.max_clips)
    print(f"Loaded {len(loader)} clips.\n")

    accum = {h: {"precisions": [], "recalls": [], "pairs": []} for h in horizons}
    context_len = int(cfg.model.vjepa.get("context_frames", 8)) \
        if "vjepa" in cfg.model else 8

    t0 = time.time()
    for clip in tqdm(loader.iter_clips(), total=len(loader), desc="Clips"):
        try:
            context_sanpo, future_gt = loader.get_context_and_future(
                clip, context_len=context_len,
                horizon_frames=list(horizon_frames.values()),
            )
            context_rgb = [f.rgb for f in context_sanpo]

            pred_scene = predictor.predict_scene_at_horizons(context_rgb, horizons)

            for h in horizons:
                hf = horizon_frames[h]
                if hf not in future_gt:
                    continue
                gt_frame = future_gt[hf]
                gdepth = gt_frame.depth_m
                if gdepth is None or gdepth.max() < 0.01:
                    gdepth, _ = gt_depth.estimate(gt_frame.rgb)
                gt_dets = gt_detector.detect(gt_frame.rgb, depth_m=gdepth)

                prec, rec, pairs = compute_object_hit_rate(pred_scene[h], gt_dets)
                accum[h]["precisions"].append(prec)
                accum[h]["recalls"].append(rec)
                accum[h]["pairs"].extend(pairs)
        except Exception as e:
            tqdm.write(f"  [SKIP] {clip.clip_id}: {e}")
            continue

    elapsed = time.time() - t0
    metrics = [
        aggregate_horizon_metrics(h, accum[h]["precisions"],
                                  accum[h]["recalls"], accum[h]["pairs"])
        for h in horizons
    ]
    print_metrics_table(metrics)

    out = {
        "backend": backend_name,
        "n_clips": len(loader),
        "elapsed_sec": elapsed,
        "horizons": [
            {"horizon_sec": m.horizon_sec, "n_samples": m.n_samples,
             "precision": m.precision, "recall": m.recall, "f1": m.f1,
             "distance_mae_m": m.distance_mae_m, "azimuth_mae_deg": m.azimuth_mae_deg,
             "per_class_mae": m.per_class_mae}
            for m in metrics
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved -> {args.output}")


if __name__ == "__main__":
    main()

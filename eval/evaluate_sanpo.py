"""
eval/evaluate_sanpo.py
───────────────────────
ONE evaluator, THREE backends, mask-grounded ground truth.

Every predicted layout and every ground-truth layout comes from the SAME
extractor — SANPO's pixel-perfect panoptic masks + metric depth via
MaskTeacher — so the comparison isolates exactly one thing: the DYNAMICS
assumption of each backend.

  --backend last_frame : layout(t+Δ) := layout(t)             (no dynamics)
  --backend flow       : warp mask+depth of t forward by k·flow (constant
                         velocity physics), then extract layout
  --backend vjepa      : V-JEPA 2 predicted latent at t+Δ -> trained probe
                         -> layout                              (learned)

  GT at t+Δ            : extract(mask_{t+Δ}, depth_{t+Δ})

No YOLO-World, no Depth Anything anywhere — those caused the domain-gap
starvation (~1.3 objects/frame) that broke the previous evaluation.

Usage
─────
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python eval/evaluate_sanpo.py --config configs/vjepa.yaml \
        --sanpo_root data/sanpo/ --split eval \
        --backend vjepa --output eval/results_vjepa.json
    # then --backend flow / --backend last_frame with matching --output
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.sanpo_loader import SANPOLoader
from models.mask_teacher import MaskTeacher, warp_layout_forward, compute_flow
from eval.metrics import (
    compute_object_hit_rate, aggregate_horizon_metrics, print_metrics_table,
)


def parse_args():
    p = argparse.ArgumentParser(description="Mask-grounded PSC evaluation")
    p.add_argument("--config", default="configs/vjepa.yaml")
    p.add_argument("--sanpo_root", required=True)
    p.add_argument("--backend", required=True,
                   choices=["vjepa", "flow", "last_frame"])
    p.add_argument("--output", default=None)
    p.add_argument("--split", type=str, default="eval")
    p.add_argument("--max_clips", type=int, default=-1)
    p.add_argument("--horizons", type=float, nargs="+", default=None)
    p.add_argument("--probe", type=str, default="checkpoints/probe.pt")
    p.add_argument("--conf_threshold", type=float, default=0.3,
                   help="Probe objectness threshold (vjepa backend).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    horizons = args.horizons or list(cfg.prediction.horizons_sec)
    fps = int(cfg.model.vjepa.get("prediction_fps", 8)) if "vjepa" in cfg.model \
        else int(cfg.get("world_model", {}).get("prediction_fps", 8))
    horizon_frames = {h: max(1, int(round(h * fps))) for h in horizons}
    context_len = int(cfg.model.vjepa.get("context_frames", 8)) \
        if "vjepa" in cfg.model else 8
    out_path = Path(args.output or f"eval/results_{args.backend}.json")

    print(f"\n{'='*62}\nMask-grounded PSC evaluation — backend: {args.backend}"
          f"\n{'='*62}")
    print(f"  Horizons : {horizons}\n  Split    : {args.split}"
          f"\n  Output   : {out_path}\n{'='*62}\n")

    # ── Data (masks required) ────────────────────────────────────────────────
    loader = SANPOLoader(sanpo_root=args.sanpo_root, split=args.split,
                         max_clips=args.max_clips)
    clips = [c for c in loader.iter_clips() if getattr(c, "mask_frames", None)]
    if not clips:
        raise SystemExit(
            f"No clips with panoptic masks (split={args.split}). Backfill:\n"
            "  python data/get_sanpo.py --path a --panoptic --max_sessions <N>")
    print(f"Loaded {len(clips)} clips with masks.")

    # ── Teacher (identical for GT and every backend) ────────────────────────
    vjepa = probe = None
    classes = None
    if args.backend == "vjepa":
        import torch
        from models.vjepa_world_model import VJEPA2WorldModel
        from models.latent_probe import LatentSpatialProbe
        print("Loading V-JEPA 2 + probe...")
        vjepa = VJEPA2WorldModel.from_config(cfg.model.vjepa)
        ck = torch.load(args.probe, map_location="cpu", weights_only=False)
        meta = ck.get("meta", {})
        if "teacher_state" not in meta:
            raise SystemExit(
                "Probe checkpoint has no teacher_state — retrain with the "
                "mask-teacher train_probe.py so eval uses the same vocabulary.")
        teacher = MaskTeacher.from_state(meta["teacher_state"])
        classes = teacher.class_names
        probe = LatentSpatialProbe.load(args.probe, embed_dim=vjepa.embed_dim)
        probe.eval().to(vjepa.device)
        print(f"  probe classes: {len(classes)}  "
              f"(train self-check F1 was {meta.get('train_selfcheck_f1','?')})")
    else:
        # calibrate a fresh teacher on this split's masks
        teacher = MaskTeacher(
            min_px=int(cfg.get("mask_teacher", {}).get("min_px", 100)),
            max_area_frac=float(cfg.get("mask_teacher", {}).get("max_area_frac", 0.25)),
            max_range_m=float(cfg.get("mask_teacher", {}).get("max_range_m", 20.0)))
        sample = []
        for c in clips[:12]:
            mp = loader.load_mask_path(c, 0)
            if mp is not None:
                sample.append(mp)
        teacher.calibrate(sample)

    image_size = int(cfg.model.vjepa.get("image_size", 256))

    # ── Evaluation loop ──────────────────────────────────────────────────────
    accum = {h: {"precisions": [], "recalls": [], "pairs": []} for h in horizons}
    t0 = time.time()

    for clip in tqdm(clips, desc="Clips", unit="clip"):
        try:
            ctx, future_gt = loader.get_context_and_future(
                clip, context_len=context_len,
                horizon_frames=list(horizon_frames.values()))
            last = ctx[-1]
            last_idx = last.frame_idx

            # Context masks/depth for the baselines
            last_mask_p = loader.load_mask_path(clip, last_idx)
            if last_mask_p is None:
                continue
            last_lab = MaskTeacher._decode(last_mask_p)
            last_depth = last.depth_m

            # Predicted layout per horizon
            pred_layouts = {}
            if args.backend == "last_frame":
                dets_now = teacher.extract(last_lab, last_depth)
                for h in horizons:
                    pred_layouts[h] = dets_now

            elif args.backend == "flow":
                flow = compute_flow(ctx[-2].rgb, last.rgb)
                for h in horizons:
                    k = float(horizon_frames[h])
                    wlab, wdep = warp_layout_forward(last_lab, last_depth, flow, k)
                    pred_layouts[h] = teacher.extract(wlab, wdep)

            else:  # vjepa
                import torch
                from models.latent_probe import probe_maps_to_detections
                grids = vjepa.predict_latent_at_horizons(
                    [f.rgb for f in ctx], horizons)
                for h, grid in grids.items():
                    g = torch.from_numpy(grid).unsqueeze(0).to(
                        next(probe.parameters()).device).float()
                    with torch.no_grad():
                        pr = probe(g)
                        obj = torch.sigmoid(pr["obj"])[0].cpu().numpy()
                        cls = torch.softmax(pr["cls"], -1)[0].cpu().numpy()
                        dpt = pr["depth"][0].cpu().numpy()
                    pred_layouts[h] = probe_maps_to_detections(
                        obj, cls, dpt, classes=classes,
                        image_size=image_size,
                        conf_threshold=args.conf_threshold)

            # Score vs mask-GT of the real future frame
            for h in horizons:
                hf = horizon_frames[h]
                if hf not in future_gt:
                    continue
                gtf = future_gt[hf]
                gt_mask_p = loader.load_mask_path(clip, gtf.frame_idx)
                if gt_mask_p is None:
                    continue
                gt_dets = teacher.extract(gt_mask_p, gtf.depth_m)
                prec, rec, pairs = compute_object_hit_rate(
                    pred_layouts[h], gt_dets)
                accum[h]["precisions"].append(prec)
                accum[h]["recalls"].append(rec)
                accum[h]["pairs"].extend(pairs)

        except Exception as e:
            tqdm.write(f"  [SKIP] {clip.clip_id}: {e}")
            continue

    elapsed = time.time() - t0
    metrics = [aggregate_horizon_metrics(h, accum[h]["precisions"],
                                         accum[h]["recalls"], accum[h]["pairs"])
               for h in horizons]
    print_metrics_table(metrics)

    out = {"backend": args.backend, "split": args.split,
           "n_clips": len(clips), "elapsed_sec": elapsed,
           "gt_source": "sanpo_panoptic_masks+gt_depth",
           "horizons": [
               {"horizon_sec": m.horizon_sec, "n_samples": m.n_samples,
                "precision": m.precision, "recall": m.recall, "f1": m.f1,
                "distance_mae_m": m.distance_mae_m,
                "azimuth_mae_deg": m.azimuth_mae_deg,
                "per_class_mae": m.per_class_mae} for m in metrics]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()

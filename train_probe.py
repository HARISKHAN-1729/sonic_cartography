"""
train_probe.py
───────────────
Train the latent spatial probe (models/latent_probe.py) on top of FROZEN
V-JEPA 2 features.

Idea — distill the pixel pipeline into the latent
──────────────────────────────────────────────────
For each training frame:
  1. TEACHER (pixel)  : run Depth Anything V2 + YOLO-World on the REAL frame
                        -> per-cell objectness / class / depth target maps.
  2. STUDENT (latent) : encode the SAME frame with frozen V-JEPA 2
                        -> token grid -> probe -> predicted maps.
  3. Loss             : BCE(objectness) + CE(class) + L1(depth), the class and
                        depth terms masked to object cells.

Because V-JEPA 2's predictor is trained so predicted future tokens live in the
SAME representation space as encoder tokens, a probe trained on encoder tokens
of real frames transfers to PREDICTED future tokens at eval time. This keeps
training cheap (frozen backbone, single GPU) and the eval task faithful.

Usage
─────
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python train_probe.py \
        --config configs/vjepa.yaml \
        --sanpo_root data/sanpo/ \
        --out checkpoints/probe.pt \
        --steps 4000 --batch_size 8

Single RTX 4090: ~20–40 min for 4000 steps with a frozen ViT-L backbone.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description="Train the V-JEPA 2 latent probe")
    p.add_argument("--config", default="configs/vjepa.yaml")
    p.add_argument("--sanpo_root", required=True)
    p.add_argument("--out", default="checkpoints/probe.pt")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_clips", type=int, default=-1)
    return p.parse_args()


def main():
    import torch
    import torch.nn.functional as F

    from models.vjepa_world_model import VJEPA2WorldModel
    from models.latent_probe import (
        LatentSpatialProbe, detections_to_target_maps, DEFAULT_CLASSES,
    )
    from models.depth_estimator import DepthEstimator
    from models.object_detector import ObjectDetector
    from data.sanpo_loader import SANPOLoader

    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = cfg.model.vjepa.get("device", "cuda")
    image_size = int(cfg.model.vjepa.get("image_size", 256))
    classes = list(cfg.model.detector.get("categories", DEFAULT_CLASSES))

    print("Loading frozen V-JEPA 2 + teacher models (Depth Anything, YOLO-World)...")
    vjepa = VJEPA2WorldModel.from_config(cfg.model.vjepa)
    for p_ in vjepa.model.parameters():
        p_.requires_grad_(False)
    depth_est = DepthEstimator.from_config(cfg.model.depth)
    detector = ObjectDetector.from_config(cfg.model.detector)

    Hg, Wg = vjepa.grid_shape
    probe = LatentSpatialProbe(
        embed_dim=vjepa.embed_dim, num_classes=len(classes),
        max_depth_m=float(cfg.model.depth.get("max_distance_m", 10.0)),
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Gather a flat pool of training frames from SANPO ────────────────────
    loader = SANPOLoader(sanpo_root=args.sanpo_root,
                         split=cfg.evaluation.get("sanpo_split", "test"),
                         max_clips=args.max_clips)
    clips = list(loader.iter_clips())
    print(f"Loaded {len(clips)} clips; sampling frames for probe training.")

    def sample_frame():
        clip = clips[np.random.randint(len(clips))]
        idx = np.random.randint(len(clip.rgb_frames))
        return loader.load_frame(clip, idx)

    # ── Build one training example: (latent grid, teacher maps) ─────────────
    def make_example():
        fr = sample_frame()
        rgb = fr.rgb
        # Teacher labels from the pixel pipeline on the REAL frame.
        depth_m, _ = depth_est.estimate(rgb)
        dets = detector.detect(rgb, depth_m=depth_m)
        obj, cls_idx, depth_t, valid = detections_to_target_maps(
            dets, Hg, Wg, image_size=image_size, classes=classes
        )
        # Student input: frozen V-JEPA encoder grid for the same frame.
        grid = vjepa.encode_frame_grid(rgb)           # (Hg, Wg, D) numpy
        return grid, obj, cls_idx, depth_t, valid

    def make_batch(bs):
        grids, objs, clss, depths, valids = [], [], [], [], []
        for _ in range(bs):
            g, o, c, d, v = make_example()
            grids.append(g); objs.append(o); clss.append(c)
            depths.append(d); valids.append(v)
        to_t = lambda arr, dt: torch.from_numpy(np.stack(arr)).to(device, dt)
        return (to_t(grids, torch.float32), to_t(objs, torch.float32),
                to_t(clss, torch.long), to_t(depths, torch.float32),
                to_t(valids, torch.bool))

    # ── Training loop ───────────────────────────────────────────────────────
    print(f"Training probe for {args.steps} steps (grid {Hg}x{Wg}, dim {vjepa.embed_dim})")
    probe.train()
    t0 = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        grid, obj_t, cls_t, depth_t, valid = make_batch(args.batch_size)
        pred = probe(grid)

        loss_obj = F.binary_cross_entropy_with_logits(pred["obj"], obj_t)

        if valid.any():
            cls_logits = pred["cls"][valid]              # (Nvalid, C)
            cls_target = cls_t[valid].clamp(min=0)       # (Nvalid,)
            loss_cls = F.cross_entropy(cls_logits, cls_target)
            loss_depth = F.l1_loss(pred["depth"][valid], depth_t[valid])
        else:
            loss_cls = torch.zeros((), device=device)
            loss_depth = torch.zeros((), device=device)

        loss = loss_obj + loss_cls + 0.5 * loss_depth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running += float(loss)
        if step % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            rate = step / (time.time() - t0)
            print(f"  step {step:5d}/{args.steps}  loss {avg:.4f}  "
                  f"(obj {float(loss_obj):.3f} cls {float(loss_cls):.3f} "
                  f"depth {float(loss_depth):.3f})  {rate:.1f} it/s")

    # ── Save ────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    probe.save(args.out, meta={"classes": classes, "grid": [Hg, Wg],
                               "embed_dim": vjepa.embed_dim,
                               "vjepa_checkpoint": cfg.model.vjepa.get("checkpoint")})
    print(f"\n✓ Probe saved -> {args.out}")
    print(f"  Now set scene_predictor.type=latent and probe_path={args.out}, "
          f"then run eval/evaluate_unified.py")


if __name__ == "__main__":
    main()

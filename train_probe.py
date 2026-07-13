"""
train_probe.py
───────────────
Train the latent spatial probe on FROZEN V-JEPA 2 features, supervised by
SANPO's own ground truth (panoptic masks + metric depth) via MaskTeacher.

WHY MASK TEACHER (v2 — replaces YOLO-World + Depth Anything)
─────────────────────────────────────────────────────────────
YOLO-World fired on only ~1.3 objects/frame on SANPO-Synthetic (rendered
imagery is out-of-domain for a real-photo detector), starving the probe.
SANPO-Synthetic ships pixel-perfect panoptic masks and metric depth, so the
teacher is now the dataset's own ground truth: dense labels, zero domain gap,
and the class vocabulary is discovered from the masks themselves (stored in
the probe checkpoint so eval uses the identical mapping).

SPEED (fixes the 1.5-hour cache)
────────────────────────────────
- Teacher labels come from masks+GT depth: no Depth Anything, no YOLO — the
  per-frame cost is now just the V-JEPA encode.
- V-JEPA encoding is BATCHED (encode_frames_batch): ~4 s/frame -> <1 s/frame.
- The whole cache (features + targets + teacher state) is saved to disk;
  re-running with the same settings skips caching entirely.

Usage
─────
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python train_probe.py --config configs/vjepa.yaml \
        --sanpo_root data/sanpo/ --split train \
        --out checkpoints/probe.pt \
        --cache_frames 1500 --steps 4000 --batch_size 512
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description="Train the V-JEPA 2 latent probe (mask teacher)")
    p.add_argument("--config", default="configs/vjepa.yaml")
    p.add_argument("--sanpo_root", required=True)
    p.add_argument("--out", default="checkpoints/probe.pt")
    p.add_argument("--split", type=str, default=None,
                   help="Split name -> reads <split>_session_ids.txt (e.g. 'train').")
    p.add_argument("--cache_frames", type=int, default=1500)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_clips", type=int, default=-1)
    p.add_argument("--frames_per_clip_cap", type=int, default=40)
    p.add_argument("--encode_batch", type=int, default=8,
                   help="Frames per batched V-JEPA forward during caching.")
    p.add_argument("--cache_path", type=str, default=None,
                   help="Where to store/reuse the feature cache "
                        "(default: checkpoints/cache_<fingerprint>.pt).")
    p.add_argument("--rebuild_cache", action="store_true")
    return p.parse_args()


def main():
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    from models.vjepa_world_model import VJEPA2WorldModel
    from models.latent_probe import LatentSpatialProbe, detections_to_target_maps
    from models.mask_teacher import MaskTeacher
    from data.sanpo_loader import SANPOLoader

    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = cfg.model.vjepa.get("device", "cuda")
    image_size = int(cfg.model.vjepa.get("image_size", 256))
    split = args.split or cfg.evaluation.get("sanpo_split", "test")

    # ── Load clips (need masks!) ─────────────────────────────────────────────
    loader = SANPOLoader(sanpo_root=args.sanpo_root, split=split,
                         max_clips=args.max_clips)
    clips = [c for c in loader.iter_clips()
             if getattr(c, "mask_frames", None)]
    n_all = len(list(SANPOLoader(sanpo_root=args.sanpo_root, split=split,
                                 max_clips=args.max_clips).iter_clips()))
    if not clips:
        raise SystemExit(
            f"No clips with panoptic masks under '{args.sanpo_root}' (split={split}).\n"
            "The mask teacher needs segmentation masks. Backfill them with:\n"
            "  python data/get_sanpo.py --path a --panoptic --max_sessions <N>\n"
            "(existing rgb/depth are skipped; only masks are fetched)")
    print(f"Loaded {len(clips)} clips with masks (of {n_all} total in split '{split}').")

    # ── Frame pool ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    pool = []
    for ci, clip in enumerate(clips):
        n = min(len(clip.rgb_frames), len(clip.mask_frames),
                len(clip.depth_frames))
        k = min(n, args.frames_per_clip_cap)
        for fi in rng.choice(n, size=k, replace=False):
            pool.append((ci, int(fi)))
    rng.shuffle(pool)
    pool = pool[: min(args.cache_frames, len(pool))]
    n_cache = len(pool)

    # ── Teacher: calibrate on a sample of masks ─────────────────────────────
    mt_cfg = cfg.get("mask_teacher", {})
    teacher = MaskTeacher(
        min_px=int(mt_cfg.get("min_px", 100)),
        max_area_frac=float(mt_cfg.get("max_area_frac", 0.25)),
        max_range_m=float(mt_cfg.get("max_range_m", 20.0)),
        taxonomy_path=mt_cfg.get("taxonomy_path", None),
    )
    calib_masks = [loader.load_mask_path(clips[ci], fi)
                   for ci, fi in pool[: min(24, n_cache)]]
    teacher.calibrate([m for m in calib_masks if m is not None])
    classes = teacher.class_names
    print(f"Class vocabulary ({len(classes)}): {classes}")

    # ── Cache fingerprint (so re-runs skip the expensive phase) ─────────────
    fp = hashlib.md5(
        f"{args.sanpo_root}|{split}|{n_cache}|{args.frames_per_clip_cap}|"
        f"{cfg.model.vjepa.checkpoint}|{teacher.divisor}|{len(classes)}"
        .encode()).hexdigest()[:10]
    cache_path = Path(args.cache_path or f"checkpoints/cache_{fp}.pt")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not args.rebuild_cache:
        print(f"Reusing feature cache: {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        G, O, C, D, Vd = blob["G"], blob["O"], blob["C"], blob["D"], blob["V"]
        teacher = MaskTeacher.from_state(blob["teacher_state"])
        classes = teacher.class_names
    else:
        # ── PHASE 1a: teacher targets (fast, CPU) ────────────────────────────
        print("Building V-JEPA world model...")
        vjepa = VJEPA2WorldModel.from_config(cfg.model.vjepa)
        for p_ in vjepa.model.parameters():
            p_.requires_grad_(False)
        Hg, Wg = vjepa.grid_shape

        frames_rgb, targets = [], []
        for ci, fi in tqdm(pool, desc="Teacher labels", unit="frame"):
            fr = loader.load_frame(clips[ci], fi)
            mask_path = loader.load_mask_path(clips[ci], fi)
            dets = teacher.extract(mask_path, fr.depth_m)
            targets.append(detections_to_target_maps(
                dets, Hg, Wg, image_size=image_size, classes=classes))
            frames_rgb.append(fr.rgb)

        # ── PHASE 1b: batched V-JEPA encoding (the GPU part) ────────────────
        grids = []
        bs = args.encode_batch
        pbar = tqdm(total=len(frames_rgb), desc="V-JEPA encode", unit="frame")
        for i in range(0, len(frames_rgb), bs):
            gs = vjepa.encode_frames_batch(frames_rgb[i:i+bs], batch_size=bs)
            grids.extend(gs)
            pbar.update(len(gs))
        pbar.close()

        G = torch.stack([torch.from_numpy(g).half() for g in grids])
        O = torch.stack([torch.from_numpy(t[0]) for t in targets])
        C = torch.stack([torch.from_numpy(t[1]) for t in targets])
        D = torch.stack([torch.from_numpy(t[2]) for t in targets])
        Vd = torch.stack([torch.from_numpy(t[3]) for t in targets])

        torch.save({"G": G, "O": O, "C": C, "D": D, "V": Vd,
                    "teacher_state": teacher.state()}, cache_path)
        print(f"Cache saved -> {cache_path} "
              f"(reused automatically on future runs)")
        del vjepa
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    N = G.shape[0]
    Hg, Wg, Dm = G.shape[1], G.shape[2], G.shape[3]
    n_obj = int(Vd.sum())
    n_cells = N * Hg * Wg
    print(f"Cache ready: {N} frames, {n_obj} object-cells "
          f"({n_obj / max(1, N):.1f}/frame, {100*n_obj/max(1,n_cells):.2f}% of cells).")
    if n_obj / max(1, N) < 2.0:
        print("  [WARN] under 2 object-cells/frame — check masks/teacher settings.")

    pos = max(1, n_obj)
    pos_weight = torch.tensor(min(float(n_cells - pos) / pos, 200.0),
                              device=device)
    print(f"  objectness pos_weight = {float(pos_weight):.1f}")

    # ── PHASE 2: train the probe over the cache ──────────────────────────────
    probe = LatentSpatialProbe(
        embed_dim=Dm, num_classes=len(classes),
        max_depth_m=float(cfg.get("mask_teacher", {}).get("max_range_m", 20.0)),
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)

    from tqdm.auto import tqdm as _tqdm
    print(f"Training probe for {args.steps} steps (batch {args.batch_size})")
    probe.train()
    t0 = time.time()
    running = 0.0
    pbar = _tqdm(range(1, args.steps + 1), desc="Training", unit="step")
    for step in pbar:
        idx = torch.randint(0, N, (min(args.batch_size, N),))
        grid = G[idx].to(device=device, dtype=torch.float32)
        obj_t = O[idx].to(device); cls_t = C[idx].to(device)
        depth_t = D[idx].to(device); valid = Vd[idx].to(device)

        import torch.nn.functional as F  # local for clarity
        pred = probe(grid)
        loss_obj = F.binary_cross_entropy_with_logits(
            pred["obj"], obj_t, pos_weight=pos_weight)
        if valid.any():
            loss_cls = F.cross_entropy(pred["cls"][valid], cls_t[valid].clamp(min=0))
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
            pbar.set_postfix(loss=f"{running/args.log_every:.4f}",
                             obj=f"{float(loss_obj):.3f}",
                             cls=f"{float(loss_cls):.3f}",
                             depth=f"{float(loss_depth):.3f}")
            running = 0.0
    print(f"Training done in {time.time() - t0:.1f}s.")

    # ── Self-check on train cache (must be >> 0 before eval is worth it) ────
    from models.latent_probe import probe_maps_to_detections
    from models.object_detector import Detection
    from eval.metrics import compute_object_hit_rate
    probe.eval()
    n_check = min(80, N)
    precs, recs = [], []
    with torch.no_grad():
        for j in torch.randint(0, N, (n_check,)).tolist():
            pr = probe(G[j:j+1].to(device=device, dtype=torch.float32))
            obj = torch.sigmoid(pr["obj"])[0].cpu().numpy()
            cls = torch.softmax(pr["cls"], -1)[0].cpu().numpy()
            dpt = pr["depth"][0].cpu().numpy()
            pred_d = probe_maps_to_detections(
                obj, cls, dpt, classes=classes,
                image_size=image_size, conf_threshold=0.3)
            gt_d = []
            ov, cv, dv = O[j].numpy(), C[j].numpy(), D[j].numpy()
            cell = image_size / Hg
            for r in range(Hg):
                for c in range(Wg):
                    if ov[r, c] > 0.5 and cv[r, c] >= 0:
                        bb = (int(c*cell), int(r*cell),
                              int((c+1)*cell), int((r+1)*cell))
                        gt_d.append(Detection(classes[int(cv[r, c])], 1.0, bb,
                                              depth_m=float(dv[r, c])))
            p_, r_, _ = compute_object_hit_rate(pred_d, gt_d)
            precs.append(p_); recs.append(r_)
    mp, mr = float(np.mean(precs)), float(np.mean(recs))
    f1 = 2 * mp * mr / (mp + mr + 1e-9)
    print(f"  TRAIN-set self-check (n={n_check}): P={mp:.3f} R={mr:.3f} F1={f1:.3f}")
    if f1 < 0.25:
        print("  [WARN] low train F1 — try more --steps / --cache_frames.")
    else:
        print("  probe learned the mask-teacher mapping ✓")

    # ── Save (teacher state travels with the probe) ─────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    probe.save(args.out, meta={
        "classes": classes, "grid": [Hg, Wg], "embed_dim": Dm,
        "cache_frames": N, "teacher_state": teacher.state(),
        "vjepa_checkpoint": cfg.model.vjepa.get("checkpoint"),
        "train_selfcheck_f1": f1,
    })
    print(f"\n✓ Probe saved -> {args.out}")
    print("  Next: python eval/evaluate_sanpo.py --config configs/vjepa.yaml "
          f"--sanpo_root {args.sanpo_root} --split eval --backend vjepa "
          "--output eval/results_vjepa.json")


if __name__ == "__main__":
    main()

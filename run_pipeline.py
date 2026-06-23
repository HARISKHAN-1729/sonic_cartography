"""
run_pipeline.py
────────────────
End-to-end demo: egocentric video → binaural predictive audio.

Reads a video file, extracts context frames, runs the full pipeline
at the specified horizon, and writes out:
  - A stereo .wav file of the predictive sonic scene
  - An annotated side-by-side visualization (context | predicted | GT)

Usage:
    python run_pipeline.py \
        --video data/sample.mp4 \
        --output_dir out/ \
        --horizon 1.5 \
        --config configs/default.yaml
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.world_model import build_world_model
from models.depth_estimator import DepthEstimator
from models.object_detector import ObjectDetector, Detection
from audio.scene_sonifier import SceneSonifier


def parse_args():
    p = argparse.ArgumentParser(
        description="Predictive Sonic Cartography — end-to-end demo"
    )
    p.add_argument("--video", type=str, required=True,
                   help="Path to input egocentric video (.mp4 / .avi / etc.)")
    p.add_argument("--output_dir", type=str, default="out/",
                   help="Directory for output files")
    p.add_argument("--horizon", type=float, default=1.5,
                   help="Prediction horizon in seconds (default: 1.5)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--all_horizons", action="store_true",
                   help="Run all configured horizons and save separate audio")
    p.add_argument("--no_vis", action="store_true",
                   help="Skip visualization output")
    return p.parse_args()


def extract_context_frames(
    video_path: str,
    n_frames: int = 8,
    target_fps: int = 8,
) -> list[np.ndarray]:
    """
    Extract n_frames from the END of the video (most recent context).

    Parameters
    ----------
    video_path : path to video file
    n_frames   : number of context frames to extract
    target_fps : subsample to this fps if video is faster

    Returns
    -------
    list of np.ndarray (H, W, 3) uint8, oldest first
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame step for target_fps subsampling
    step = max(1, int(round(video_fps / target_fps)))

    # Collect the last n_frames * step frames
    start_frame = max(0, total_frames - n_frames * step)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_count = 0
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            # BGR → RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1

    cap.release()

    if not frames:
        raise ValueError(f"Could not extract frames from {video_path}")

    # Pad to n_frames by repeating first frame if needed
    while len(frames) < n_frames:
        frames.insert(0, frames[0].copy())

    return frames[-n_frames:]


def save_visualization(
    context_frames: list[np.ndarray],
    predicted_frame: np.ndarray,
    detections: list[Detection],
    depth_m: np.ndarray,
    horizon_sec: float,
    output_path: str | Path,
) -> None:
    """
    Save a side-by-side visualization:
    Left: last context frame | Middle: predicted frame with detections |
    Right: depth map heatmap
    """
    # Use last context frame
    ctx = cv2.cvtColor(context_frames[-1], cv2.COLOR_RGB2BGR)
    pred = cv2.cvtColor(predicted_frame, cv2.COLOR_RGB2BGR)

    # Resize all to same height
    H = 256
    def resize_h(img, h):
        scale = h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, h))

    ctx_r  = resize_h(ctx,  H)
    pred_r = resize_h(pred, H)

    # Depth heatmap
    depth_vis = cv2.normalize(
        (1.0 / np.clip(depth_m, 0.1, 10.0)).astype(np.float32),
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    depth_r = resize_h(depth_color, H)

    # Draw detections on predicted frame
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        # Scale bbox to resized pred
        sx = pred_r.shape[1] / predicted_frame.shape[1]
        sy = pred_r.shape[0] / predicted_frame.shape[0]
        rx1, ry1 = int(x1*sx), int(y1*sy)
        rx2, ry2 = int(x2*sx), int(y2*sy)

        color = (0, 200, 100)  # green boxes
        cv2.rectangle(pred_r, (rx1, ry1), (rx2, ry2), color, 1)
        label = f"{det.label} {det.depth_m:.1f}m"
        cv2.putText(pred_r, label, (rx1, max(ry1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Add text labels
    for img, txt in [
        (ctx_r,   "Context (now)"),
        (pred_r,  f"Predicted (+{horizon_sec:.1f}s)"),
        (depth_r, "Depth map"),
    ]:
        cv2.putText(img, txt, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Horizontal concatenation
    vis = np.concatenate([ctx_r, pred_r, depth_r], axis=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
    print(f"  Visualization → {output_path}")


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = (
        list(cfg.prediction.horizons_sec)
        if args.all_horizons
        else [args.horizon]
    )

    print(f"\nPredictive Sonic Cartography")
    print(f"  Video   : {args.video}")
    print(f"  Horizon : {horizons}s")
    print(f"  Output  : {out_dir}\n")

    # ── Load models ────────────────────────────────────────────────────────
    print("Loading models (first run downloads checkpoints)...")
    t0 = time.time()

    world_model = build_world_model(cfg)   # backend chosen by cfg.world_model.backend
    depth_est = DepthEstimator.from_config(cfg.model.depth)
    detector  = ObjectDetector.from_config(cfg.model.detector)
    sonifier  = SceneSonifier.from_config(cfg.audio)

    print(f"  Models loaded in {time.time()-t0:.1f}s "
          f"(world model: {cfg.world_model.backend})\n")

    # ── Extract context frames ─────────────────────────────────────────────
    print("Extracting context frames...")
    context_frames = extract_context_frames(
        args.video,
        n_frames=cfg.world_model.get("context_frames", 8),
        target_fps=cfg.world_model.prediction_fps,
    )
    print(f"  Extracted {len(context_frames)} context frames\n")

    # ── Predict future frames ──────────────────────────────────────────────
    print(f"Predicting future at horizons {horizons}s...")
    t0 = time.time()
    predicted = world_model.predict_at_horizons(
        context_frames=context_frames,
        horizon_times_sec=horizons,
        action=cfg.world_model.get("action", "walk_forward"),
    )
    print(f"  Prediction done in {time.time()-t0:.1f}s\n")

    # ── Parse each predicted frame and sonify ─────────────────────────────
    video_stem = Path(args.video).stem

    for h_sec, pred_frame in sorted(predicted.items()):
        print(f"Horizon {h_sec:.1f}s:")

        # Depth
        depth_m, _ = depth_est.estimate(pred_frame)
        print(f"  Depth range: {depth_m.min():.1f}–{depth_m.max():.1f}m")

        # Detect
        dets = detector.detect(pred_frame, depth_m=depth_m)
        print(f"  Detected {len(dets)} objects:")
        for d in dets[:6]:
            print(f"    {d.label:<16} {d.depth_m:5.1f}m  az={d.azimuth_deg:+.0f}°  "
                  f"conf={d.confidence:.2f}")
        if len(dets) > 6:
            print(f"    ... and {len(dets)-6} more")

        # Sonify
        audio = sonifier.sonify(dets)
        audio_path = out_dir / f"{video_stem}_h{h_sec:.1f}s.wav"
        sonifier.save(audio, audio_path)
        print(f"  Audio → {audio_path}")

        # Visualize
        if not args.no_vis:
            vis_path = out_dir / f"{video_stem}_h{h_sec:.1f}s_vis.jpg"
            save_visualization(
                context_frames=context_frames,
                predicted_frame=pred_frame,
                detections=dets,
                depth_m=depth_m,
                horizon_sec=h_sec,
                output_path=vis_path,
            )

        print()

    print("Done.")


if __name__ == "__main__":
    main()

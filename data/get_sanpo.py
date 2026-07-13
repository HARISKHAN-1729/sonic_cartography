"""
data/get_sanpo.py
──────────────────
Complete SANPO dataset acquisition script.

THREE PATHS — pick whichever fits your situation:

  PATH A (easiest, ~50 GB):   Download SANPO-Synthetic test split
                               Free, no sign-up, pixel-perfect depth.
                               1961 sessions, great for eval.

  PATH B (~200 GB subset):    Download SANPO-Real test split (RGB only)
                               Free, real-world pedestrian footage.
                               Requires gcloud CLI.

  PATH C (0 download):        Build a SANPO-compatible dataset yourself
                               using your own phone + an iPhone/Android
                               depth-capable camera. We provide a
                               conversion script.

Usage:
    # Path A — synthetic (recommended for LBR paper)
    python data/get_sanpo.py --path a

    # Path B — real subset (RGB + depth, test split only)
    python data/get_sanpo.py --path b --max_sessions 100

    # Path C — convert your own footage
    python data/get_sanpo.py --path c --input_dir /path/to/your/videos/
"""

from __future__ import annotations
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sanpo"

# ─────────────────────────────────────────────────────────────────────────────
# GCS bucket paths (publicly accessible, no auth needed for download)
# ─────────────────────────────────────────────────────────────────────────────
GCS_BASE        = "gs://gresearch/sanpo_dataset/v0"
GCS_REAL        = f"{GCS_BASE}/sanpo-real"
GCS_SYNTHETIC   = f"{GCS_BASE}/sanpo-synthetic"
GCS_SPLITS_REAL = f"{GCS_REAL}/splits"
GCS_SPLITS_SYN  = f"{GCS_SYNTHETIC}/splits"


# ══════════════════════════════════════════════════════════════════════════════
# PATH A — Download SANPO-Synthetic test split
# ══════════════════════════════════════════════════════════════════════════════

def path_a_synthetic(max_sessions: int = -1, want_panoptic: bool = False):
    """
    Download SANPO-Synthetic test sessions in ONE command.

    Real bucket layout (per session):
        {base}/{session_id}/description.json
        {base}/{session_id}/{camera_head|camera_chest}/left/video_frames/*.png
        {base}/{session_id}/{sensor}/left/depth_maps/*.float16.gz   (metres)
        {base}/{session_id}/{sensor}/left/segmentation_masks/*.png   (optional)
        {base}/{session_id}/{sensor}/camera_poses.csv                (optional)

    Local layout produced (what SANPOLoader reads):
        data/sanpo/sanpo-synthetic/splits/test_session_ids.txt
        data/sanpo/sanpo-synthetic/sessions/{session_id}/rgb/*.png
        data/sanpo/sanpo-synthetic/sessions/{session_id}/depth/*.npz  (float32 m)
        data/sanpo/sanpo-synthetic/sessions/{session_id}/metadata.json
        (panoptic/ and camera_poses.csv only if requested / available)

    Depth files are *.float16.gz: the first two float16 values are H and W,
    the rest are metric depth; we convert each to a compressed .npz.

    Sizes: full test split ~10 GB; --max_sessions caps it (try 1 first).
    """
    print("\n" + "═" * 60)
    print("PATH A — SANPO-Synthetic download (single command)")
    print("═" * 60)
    _check_gcloud()

    out_dir = DATA_DIR / "sanpo-synthetic"
    sessions_dir = out_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: resolve the list of test session IDs ───────────────────────
    print("\n[1/3] Resolving test session IDs...")
    session_ids = _ensure_split_ids(GCS_SYNTHETIC, out_dir, split="test")
    if not session_ids:
        raise RuntimeError(
            "Could not resolve any session IDs. Check bucket access with:\n"
            f"  gcloud storage ls {GCS_SYNTHETIC}/"
        )
    if max_sessions > 0:
        session_ids = session_ids[:max_sessions]
    print(f"  Using {len(session_ids)} session(s).")

    # ── Step 2 & 3: per-session download + depth conversion ────────────────
    print(f"\n[2/3] Downloading + [3/3] converting {len(session_ids)} session(s)...\n")
    for i, sid in enumerate(session_ids, start=1):
        print(f"[{i}/{len(session_ids)}] {sid}")
        local_session = sessions_dir / sid
        needs_masks = want_panoptic and not any(
            (local_session / "panoptic").glob("*.png")
            if (local_session / "panoptic").exists() else [])
        if local_session.exists() and _session_complete(local_session) \
                and not needs_masks:
            print("  already complete — skipping")
            continue
        if local_session.exists() and _session_complete(local_session) \
                and needs_masks:
            # Backfill only the segmentation masks (rgb/depth kept as-is).
            print("  backfilling panoptic masks only...")
            sensor = _detect_sensor(f"{GCS_SYNTHETIC}/{sid}")
            _download_pattern(
                f"{GCS_SYNTHETIC}/{sid}/{sensor}/left/segmentation_masks/*.png",
                local_session / "panoptic")
            n_masks = len(list((local_session / "panoptic").glob("*.png")))
            print(f"  masks: {n_masks}  ✓")
            continue
        _download_synthetic_session(
            remote_session=f"{GCS_SYNTHETIC}/{sid}",
            local_session=local_session,
            want_panoptic=want_panoptic,
        )

    print(f"\n✓ SANPO-Synthetic ready at: {out_dir}")
    print(f"  Sessions on disk: {len(list(sessions_dir.iterdir()))}")
    print("\nNext:")
    print(f"  python train_probe.py --config configs/vjepa.yaml "
          f"--sanpo_root {DATA_DIR} --out checkpoints/probe.pt --steps 4000")
    print(f"  python eval/evaluate_unified.py --config configs/vjepa.yaml "
          f"--sanpo_root {DATA_DIR} --output eval/results_vjepa.json")


# ── Path A helpers ──────────────────────────────────────────────────────────

def _ensure_split_ids(base: str, local_base: Path, split: str = "test") -> list:
    """
    Return the list of session IDs for `split`. Strategy:
      1. use a local splits file if already present;
      2. else download a remote splits file (trying common names);
      3. else list the bucket top level and derive session IDs directly.
    """
    splits_dir = local_base / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    local_file = splits_dir / f"{split}_session_ids.txt"

    # 1) local file already there
    if local_file.exists():
        ids = [l.strip() for l in local_file.read_text().splitlines() if l.strip()]
        if ids:
            print(f"  Using existing split file: {local_file}")
            return ids

    # 2) try downloading a remote splits file (ensure dest dir exists; copy file)
    for remote_name in (f"{split}_session_ids.txt", f"{split}.txt",
                         f"{split}_clips.txt"):
        remote = f"{base}/splits/{remote_name}"
        if _remote_exists(remote):
            run([
                "gcloud", "storage", "cp", remote, str(local_file),
            ])
            ids = [l.strip() for l in local_file.read_text().splitlines() if l.strip()]
            if ids:
                print(f"  Downloaded split file: {remote_name} ({len(ids)} ids)")
                return ids

    # 3) derive from a bucket listing (filter out non-session entries)
    print("  No splits file found; listing bucket to derive session IDs...")
    ids = _list_bucket_sessions(base + "/")
    # write them out so subsequent runs are reproducible
    if ids:
        local_file.write_text("\n".join(ids) + "\n")
        print(f"  Derived {len(ids)} session IDs and wrote {local_file}")
    return ids


def _download_synthetic_session(remote_session: str, local_session: Path,
                                want_panoptic: bool = False) -> None:
    """Download one SANPO-Synthetic session and convert its depth to .npz."""
    rgb_dir = local_session / "rgb"
    raw_depth_dir = local_session / "depth_raw"
    depth_dir = local_session / "depth"
    panoptic_dir = local_session / "panoptic"

    sensor = _detect_sensor(remote_session)
    print(f"  sensor: {sensor}")
    remote_left = f"{remote_session}/{sensor}/left"

    _download_pattern(f"{remote_left}/video_frames/*.png", rgb_dir)
    _download_pattern(f"{remote_left}/depth_maps/*.float16.gz", raw_depth_dir)
    if want_panoptic:
        _download_pattern(f"{remote_left}/segmentation_masks/*.png", panoptic_dir)

    # metadata (description.json -> metadata.json)
    desc = f"{remote_session}/description.json"
    if _remote_exists(desc):
        run(["gcloud", "storage", "cp", desc, str(local_session / "metadata.json")])

    # optional camera poses
    poses = f"{remote_session}/{sensor}/camera_poses.csv"
    if _remote_exists(poses):
        run(["gcloud", "storage", "cp", poses, str(local_session / "camera_poses.csv")])

    print("  converting depth (.float16.gz -> .npz, metres)...")
    n_depth = _convert_depth_files(raw_depth_dir, depth_dir)

    rgb_n = len(list(rgb_dir.glob("*.png")))
    print(f"  rgb frames: {rgb_n}   depth maps: {n_depth}")
    if rgb_n < 10 or n_depth < 10:
        raise RuntimeError(f"Too few frames downloaded for {local_session.name}; "
                           "delete the session folder and re-run.")

    # remove the raw .gz depth once conversion succeeded
    if n_depth == len(list(raw_depth_dir.glob('*.float16.gz'))):
        shutil.rmtree(raw_depth_dir, ignore_errors=True)
    print("  ✓ session complete")


def _detect_sensor(remote_session: str) -> str:
    """Return 'camera_head' or 'camera_chest' depending on which exists."""
    for sensor in ("camera_head", "camera_chest"):
        probe = f"{remote_session}/{sensor}/left/video_frames/000000.png"
        if _remote_exists(probe):
            return sensor
    raise RuntimeError(f"No camera_head/camera_chest under {remote_session}")


def _download_pattern(source_glob: str, destination: Path) -> None:
    """Download all objects matching a GCS wildcard into an existing dir."""
    destination.mkdir(parents=True, exist_ok=True)
    run(["gcloud", "storage", "cp", "--no-clobber",
         source_glob, str(destination) + "/"])


def _convert_depth_files(raw_dir: Path, output_dir: Path) -> int:
    """
    Convert SANPO *.float16.gz depth maps to compressed .npz (float32 metres).
    First two float16 values are H and W; the rest are metric depth.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    raw_files = sorted(raw_dir.glob("*.float16.gz"))
    for idx, raw_path in enumerate(raw_files, start=1):
        out_path = output_dir / (raw_path.name.removesuffix(".float16.gz") + ".npz")
        if out_path.exists():
            converted += 1
            continue
        with gzip.open(raw_path, "rb") as fh:
            values = np.frombuffer(fh.read(), dtype="<f2")
        if values.size < 3:
            raise ValueError(f"Invalid depth file: {raw_path}")
        h, w = int(values[0]), int(values[1])
        pixels = values[2:]
        if h <= 0 or w <= 0 or pixels.size != h * w:
            raise ValueError(f"Bad depth dims {h}x{w} in {raw_path}")
        depth = pixels.reshape(h, w).astype(np.float32)
        np.savez_compressed(out_path, depth=depth)
        converted += 1
        if idx % 50 == 0:
            print(f"    {idx}/{len(raw_files)} depth maps")
    return converted


# ══════════════════════════════════════════════════════════════════════════════
# PATH B — Download SANPO-Real test split (RGB + depth)
# ══════════════════════════════════════════════════════════════════════════════

def path_b_real(max_sessions: int = 100):
    """
    Download SANPO-Real test sessions (RGB frames + CREStereo depth maps).

    Real data from 701 outdoor sessions recorded by volunteer runners
    in urban, park, and suburban settings across 4 US cities.
    Cameras: ZED Mini (head-mounted) + ZED 2i (chest-mounted).
    Depth: CREStereo dense depth maps in metres (npz).

    Approximate sizes:
        Full real (701 sessions): ~2 TB
        Test split (~140 sessions): ~400 GB
        100 sessions: ~280 GB
        10 sessions:  ~28 GB  ← good for a quick test run
    """
    print("\n" + "═"*60)
    print("PATH B — SANPO-Real download")
    print("═"*60)

    _check_gcloud()

    out_dir = DATA_DIR / "sanpo-real"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download splits
    print("\n[1/3] Downloading split lists...")
    _gcloud_cp(f"{GCS_SPLITS_REAL}/", str(out_dir / "splits"), recursive=True)

    splits_path = out_dir / "splits" / "test.txt"
    if splits_path.exists():
        with open(splits_path) as f:
            sessions = [line.strip() for line in f if line.strip()]
        print(f"  Found {len(sessions)} sessions in test split")
    else:
        sessions = _list_bucket_sessions(GCS_REAL + "/sessions/")

    if max_sessions > 0:
        sessions = sessions[:max_sessions]
        print(f"  Limiting to {max_sessions} sessions (~{max_sessions*2.8:.0f} GB)")

    print(f"\n[2/3] Downloading {len(sessions)} real sessions...")
    print("  Downloading: RGB left frames + CREStereo depth only")
    print("  Skipping:    right stereo frames, raw ZED depth, panoptic (saves ~60%)\n")

    sessions_dir = out_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    for i, session_id in enumerate(sessions):
        dest = sessions_dir / session_id
        if dest.exists() and _session_complete(dest):
            print(f"  [{i+1}/{len(sessions)}] {session_id} — skip (complete)")
            continue

        print(f"  [{i+1}/{len(sessions)}] Downloading {session_id}...")
        src = f"{GCS_REAL}/sessions/{session_id}"

        # Only download what we need: left RGB + CREStereo depth
        _gcloud_cp(src + "/left/",        str(dest / "rgb"),   recursive=True)
        _gcloud_cp(src + "/depth_cre/",   str(dest / "depth"), recursive=True)
        _gcloud_cp(src + "/metadata.json", str(dest / "metadata.json"))

    # Convert
    print(f"\n[3/3] Converting to sonic_cartography format...")
    _convert_real_to_loader_format(out_dir)

    print(f"\n✓ SANPO-Real ready at: {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# PATH C — Build your own SANPO-compatible dataset
# ══════════════════════════════════════════════════════════════════════════════

def path_c_custom(input_dir: str, fps_target: int = 10):
    """
    Convert your own egocentric walking video into SANPO-compatible format.

    WHAT YOU NEED TO RECORD:
    ─────────────────────────
    Option 1 (best): iPhone 12 Pro+ or iPad Pro with LiDAR
        → Use "Record3D" app (free) — exports RGB + metric depth per frame
        → Walk around for 30s+ per clip, outdoors

    Option 2 (good): Any Android phone with ToF sensor
        (Samsung S20 Ultra, Pixel 9 Pro, etc.)
        → Use "DepthLab" or "Depth Camera" apps
        → Exports RGB + depth as synchronized video

    Option 3 (works): Any smartphone
        → Record RGB video only
        → We use Depth Anything V2 to generate pseudo depth (good enough for eval)

    HOW TO RECORD WELL:
    ────────────────────
    - Mount phone at chest or head height (not hand-held; too shaky)
    - Walk at a normal pace — not running
    - Walk around: sidewalks, intersections, indoor corridors, cafeteria
    - 30-60 seconds per clip is ideal
    - 10+ clips gives a solid small test set (~300 eval samples)
    - Record in good light; avoid backlit scenes
    - Keep the lens clean

    WHAT THIS SCRIPT DOES:
    ───────────────────────
    1. Reads video files from input_dir (mp4/mov/avi)
    2. Extracts frames at fps_target
    3. If depth files are present (Record3D .r3d or side-by-side depth video):
       converts them to metric depth NPZ files
    4. Otherwise: runs Depth Anything V2 to generate pseudo depth
    5. Writes everything to data/sanpo/custom/ in our loader format
    6. Writes a test.txt split file
    """
    print("\n" + "═"*60)
    print("PATH C — Convert custom egocentric footage")
    print("═"*60)

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    out_dir = DATA_DIR / "custom"
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = out_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    # Find all video files
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    videos = [
        f for f in sorted(input_path.iterdir())
        if f.suffix.lower() in video_exts
    ]
    print(f"\nFound {len(videos)} video files in {input_dir}")

    if not videos:
        print("\nNo video files found! Supported formats: .mp4 .mov .avi .mkv .webm")
        print(f"Place your walking videos in: {input_dir}")
        return

    session_ids = []

    for video_idx, video_path in enumerate(videos):
        session_id = f"custom_{video_idx:04d}_{video_path.stem}"
        session_dir = sessions_dir / session_id
        rgb_dir     = session_dir / "rgb"
        depth_dir   = session_dir / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{video_idx+1}/{len(videos)}] Processing: {video_path.name}")

        # Step 1: Extract frames
        n_frames = _extract_frames(video_path, rgb_dir, fps_target)
        print(f"  Extracted {n_frames} frames at {fps_target} fps")

        if n_frames < 20:
            print(f"  Too few frames ({n_frames}), skipping")
            shutil.rmtree(session_dir)
            continue

        # Step 2: Get depth
        # Check for paired depth video (same name + "_depth" suffix)
        depth_video = video_path.parent / (video_path.stem + "_depth" + video_path.suffix)
        record3d_dir = video_path.parent / video_path.stem  # Record3D unzipped folder

        if record3d_dir.is_dir() and (record3d_dir / "rgbd").exists():
            print(f"  Found Record3D export at {record3d_dir}")
            _convert_record3d(record3d_dir, rgb_dir, depth_dir, n_frames)
        elif depth_video.exists():
            print(f"  Found paired depth video: {depth_video.name}")
            _extract_depth_video(depth_video, depth_dir, fps_target, n_frames)
        else:
            print(f"  No depth found — generating pseudo-depth with Depth Anything V2...")
            _generate_pseudo_depth(rgb_dir, depth_dir)

        # Step 3: Write metadata
        meta = {
            "session_id": session_id,
            "source_video": str(video_path),
            "fps": fps_target,
            "n_frames": n_frames,
            "scene_type": "outdoor",  # update manually if indoor
            "depth_type": "pseudo" if not (depth_video.exists() or record3d_dir.is_dir())
                          else "metric",
        }
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        session_ids.append(session_id)
        print(f"  ✓ Session saved: {session_dir}")

    # Write split file (all custom sessions go to test)
    split_dir = out_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    with open(split_dir / "test.txt", "w") as f:
        f.write("\n".join(session_ids))

    print(f"\n{'═'*60}")
    print(f"✓ Custom dataset ready: {len(session_ids)} sessions")
    print(f"  Location: {out_dir}")
    print(f"\nTest with:")
    print(f"  python eval/evaluate.py --sanpo_root {DATA_DIR} --max_clips {len(session_ids)}")
    print(f"  (change sanpo_root to {out_dir} if running only custom data)")


# ══════════════════════════════════════════════════════════════════════════════
# Conversion helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_frames(
    video_path: Path,
    out_dir: Path,
    fps: int,
) -> int:
    """Extract frames from video at target fps using ffmpeg or opencv."""
    try:
        # Try ffmpeg first (best quality, handles all formats)
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",              # high quality JPEG
            str(out_dir / "frame_%06d.jpg"),
            "-loglevel", "error",
            "-y",
        ]
        subprocess.run(cmd, check=True)
        frames = list(out_dir.glob("*.jpg"))
        return len(frames)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: opencv
        return _extract_frames_opencv(video_path, out_dir, fps)


def _extract_frames_opencv(video_path: Path, out_dir: Path, fps: int) -> int:
    try:
        import cv2
    except ImportError:
        raise ImportError("pip install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(video_fps / fps)))

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            cv2.imwrite(str(out_dir / f"frame_{saved:06d}.jpg"), frame)
            saved += 1
        count += 1

    cap.release()
    return saved


def _generate_pseudo_depth(rgb_dir: Path, depth_dir: Path) -> None:
    """
    Run Depth Anything V2 on all frames to generate metric-scaled depth.
    Requires GPU for reasonable speed.
    """
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        from PIL import Image
    except ImportError:
        print("    [WARN] transformers/torch not installed; skipping depth generation")
        print("    Install: pip install transformers torch")
        return

    device = 0 if (torch.cuda.is_available()) else -1
    dtype  = torch.float16 if device == 0 else torch.float32

    print("    Loading Depth Anything V2 ViT-L...")
    pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device,
        torch_dtype=dtype,
    )

    frame_paths = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    print(f"    Estimating depth for {len(frame_paths)} frames...")

    # Empirical outdoor scale factor
    SCALE_M = 10.0

    for i, fp in enumerate(frame_paths):
        out_path = depth_dir / (fp.stem + ".npz")
        if out_path.exists():
            continue

        img = Image.open(fp).convert("RGB")
        result = pipe(img)
        depth_raw = np.array(result["depth"], dtype=np.float32)

        # Normalise inverse depth → metric
        d_min, d_max = depth_raw.min(), depth_raw.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_raw - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_raw)

        depth_m = np.clip(
            (1.0 - depth_norm) * SCALE_M, 0.1, SCALE_M
        ).astype(np.float32)

        np.savez_compressed(str(out_path), depth=depth_m)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(frame_paths)} frames done")


def _convert_record3d(
    r3d_dir: Path,
    rgb_dir: Path,
    depth_dir: Path,
    n_frames: int,
) -> None:
    """
    Convert Record3D export to our format.
    Record3D exports: rgbd/ folder with .jpg + .depth (16-bit PNG, mm) pairs.
    """
    rgbd_dir = r3d_dir / "rgbd"
    for i, depth_file in enumerate(sorted(rgbd_dir.glob("*.depth"))):
        # Depth
        depth_raw = np.frombuffer(depth_file.read_bytes(), dtype=np.float32)
        # Record3D depth is in metres (float32 raw binary)
        # Try to infer H×W from n_frames and file size
        n_pixels = len(depth_raw)
        # Common resolutions: 256×192=49152, 480×640=307200
        depth_m = depth_raw.reshape(-1)  # will reshape below per resolution

        # Try to reshape to nearest common aspect ratio
        for h, w in [(192, 256), (480, 640), (720, 960), (1080, 1440)]:
            if h * w == n_pixels:
                depth_m = depth_raw.reshape(h, w)
                break
        else:
            side = int(n_pixels ** 0.5)
            depth_m = depth_raw.reshape(side, -1)

        np.savez_compressed(str(depth_dir / f"frame_{i:06d}.npz"), depth=depth_m)

        # RGB companion
        rgb_file = depth_file.with_suffix(".jpg")
        if rgb_file.exists():
            dest = rgb_dir / f"frame_{i:06d}.jpg"
            if not dest.exists():
                shutil.copy2(rgb_file, dest)

    print(f"    Converted {i+1} Record3D frames")


def _extract_depth_video(
    depth_video: Path,
    depth_dir: Path,
    fps: int,
    n_frames: int,
) -> None:
    """
    Extract depth from a paired depth video (greyscale, 16-bit values ≈ mm).
    Some apps output a side-by-side RGB+depth video; others a separate file.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("pip install opencv-python")

    cap = cv2.VideoCapture(str(depth_video))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(video_fps / fps)))

    count  = 0
    saved  = 0
    while saved < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            # Convert 8-bit greyscale → approximate metres
            # (many apps encode 0–255 as 0–10m)
            if len(frame.shape) == 3:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                grey = frame
            depth_m = grey.astype(np.float32) / 255.0 * 10.0
            np.savez_compressed(
                str(depth_dir / f"frame_{saved:06d}.npz"), depth=depth_m
            )
            saved += 1
        count += 1

    cap.release()


def _convert_synthetic_to_loader_format(out_dir: Path) -> None:
    """
    SANPO-Synthetic stores depth as .npz with key 'depth' — already correct.
    We just ensure the directory structure matches our SANPOLoader expectations.
    """
    sessions_dir = out_dir / "sessions"
    if not sessions_dir.exists():
        return

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        # Rename 'rgb_left' → 'rgb' if needed
        rgb_left = session_dir / "rgb_left"
        rgb      = session_dir / "rgb"
        if rgb_left.exists() and not rgb.exists():
            rgb_left.rename(rgb)
        # Rename 'depth_cre' → 'depth' if needed
        depth_cre = session_dir / "depth_cre"
        depth     = session_dir / "depth"
        if depth_cre.exists() and not depth.exists():
            depth_cre.rename(depth)


def _convert_real_to_loader_format(out_dir: Path) -> None:
    """Same renaming as synthetic."""
    _convert_synthetic_to_loader_format(out_dir)


def _session_complete(session_dir: Path) -> bool:
    """Check if a session has been fully downloaded."""
    return (
        (session_dir / "rgb").exists() and
        (session_dir / "depth").exists() and
        len(list((session_dir / "rgb").glob("*.jpg")) +
            list((session_dir / "rgb").glob("*.png"))) >= 10
    )


def _gcloud_cp(src: str, dest: str, recursive: bool = False,
               exclude: list = None) -> None:
    cmd = ["gcloud", "storage", "cp"]
    if recursive:
        cmd.append("-r")
    if exclude:
        for pat in exclude:
            cmd += ["--exclude", pat]
    cmd += [src, dest]
    subprocess.run(cmd, check=True)


def run(command: list) -> None:
    """Run a gcloud (or other) command, echoing it, raising on failure."""
    print("    $", " ".join(map(str, command)))
    subprocess.run(command, check=True)


def _remote_exists(uri: str) -> bool:
    """True if a GCS object/prefix exists (quietly)."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", uri],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


# Entries at the synthetic top level that are NOT sessions.
_NON_SESSION_NAMES = {"splits", "LICENSE", "README", "metadata", ""}


def _list_bucket_sessions(prefix: str) -> list[str]:
    """List session IDs at a GCS prefix, excluding non-session entries/files."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", prefix],
        capture_output=True, text=True, check=True
    )
    sessions = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        is_dir = line.endswith("/")                 # sessions are folders
        name = line.rstrip("/").split("/")[-1]
        if not is_dir:                              # skip top-level files
            continue
        if name in _NON_SESSION_NAMES or "." in name:
            continue
        sessions.append(name)
    return sessions


def _check_gcloud() -> None:
    """Ensure gcloud CLI is installed."""
    if shutil.which("gcloud") is None:
        print("\n[ERROR] gcloud CLI not found.")
        print("Install from: https://cloud.google.com/sdk/docs/install")
        print("\nQuick install (Linux/Mac):")
        print("  curl https://sdk.cloud.google.com | bash")
        print("  gcloud init  # (no Google account needed for public buckets)")
        sys.exit(1)
    print("  gcloud CLI found ✓")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Download or build the SANPO dataset for Predictive Sonic Cartography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/get_sanpo.py --path a                        # Synthetic, all test sessions
  python data/get_sanpo.py --path a --max_sessions 50      # Synthetic, 50 sessions (~2.5 GB)
  python data/get_sanpo.py --path b --max_sessions 10      # Real, 10 sessions (~28 GB)
  python data/get_sanpo.py --path c --input_dir ~/videos/  # Your own footage
        """,
    )
    p.add_argument("--path", choices=["a", "b", "c"], required=True,
                   help="A=synthetic, B=real, C=custom footage")
    p.add_argument("--max_sessions", type=int, default=-1,
                   help="Max sessions to download (A/B only; -1 = all)")
    p.add_argument("--input_dir", type=str, default=None,
                   help="Directory with your own video files (Path C only)")
    p.add_argument("--fps", type=int, default=10,
                   help="Target fps for frame extraction (Path C only)")
    p.add_argument("--panoptic", action="store_true",
                   help="Also download segmentation masks (Path A; not needed by the pipeline)")
    args = p.parse_args()

    if args.path == "a":
        path_a_synthetic(max_sessions=args.max_sessions, want_panoptic=args.panoptic)
    elif args.path == "b":
        path_b_real(max_sessions=args.max_sessions)
    elif args.path == "c":
        if not args.input_dir:
            p.error("--input_dir required for Path C")
        path_c_custom(input_dir=args.input_dir, fps_target=args.fps)


if __name__ == "__main__":
    main()

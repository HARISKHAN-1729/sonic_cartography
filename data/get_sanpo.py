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

def path_a_synthetic(max_sessions: int = -1):
    """
    Download SANPO-Synthetic test sessions.

    SANPO-Synthetic has pixel-perfect depth maps and panoptic masks,
    making it ideal for offline evaluation — you know exactly where
    every object is, so ground-truth detections can be derived directly
    from the panoptic labels (no need to run YOLO-World on GT frames).

    Structure after download:
        data/sanpo/
        └── sanpo-synthetic/
            ├── splits/
            │   ├── test.txt       ← list of test session IDs
            │   └── train.txt
            └── sessions/
                ├── session_00001/
                │   ├── rgb/            ← PNG frames
                │   ├── depth/          ← .npz float32 depth (metres)
                │   ├── panoptic/       ← panoptic mask PNGs
                │   └── metadata.json
                └── ...

    Approximate sizes:
        Full synthetic (1961 sessions): ~50 GB
        Test split only (~392 sessions): ~10 GB
        RGB only (no panoptic): ~6 GB
    """
    print("\n" + "═"*60)
    print("PATH A — SANPO-Synthetic download")
    print("═"*60)

    _check_gcloud()

    out_dir = DATA_DIR / "sanpo-synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download splits file to know which sessions are in test
    print("\n[1/3] Downloading split lists...")
    _gcloud_cp(f"{GCS_SPLITS_SYN}/", str(out_dir / "splits"), recursive=True)

    splits_test_path = out_dir / "splits" / "test.txt"
    if not splits_test_path.exists():
        # Fallback: list and filter by name
        print("  Splits file not found at expected path, listing bucket...")
        sessions = _list_bucket_sessions(GCS_SYNTHETIC + "/sessions/")
    else:
        with open(splits_test_path) as f:
            sessions = [line.strip() for line in f if line.strip()]
        print(f"  Found {len(sessions)} sessions in test split")

    if max_sessions > 0:
        sessions = sessions[:max_sessions]
        print(f"  Limiting to first {max_sessions} sessions")

    # Step 2: Download sessions
    sessions_dir = out_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    print(f"\n[2/3] Downloading {len(sessions)} sessions...")
    print("  (This includes RGB frames + depth NPZ files)")
    print("  Tip: interrupt at any time; gcloud resumes partial downloads\n")

    for i, session_id in enumerate(sessions):
        dest = sessions_dir / session_id
        if dest.exists() and _session_complete(dest):
            print(f"  [{i+1}/{len(sessions)}] {session_id} — already complete, skipping")
            continue

        print(f"  [{i+1}/{len(sessions)}] {session_id}...")
        src = f"{GCS_SYNTHETIC}/sessions/{session_id}"
        _gcloud_cp(src, str(dest), recursive=True,
                   exclude=["*_right.png", "*.mp4"])  # skip right-eye frames + video

    # Step 3: Convert to our loader format
    print(f"\n[3/3] Converting to sonic_cartography format...")
    _convert_synthetic_to_loader_format(out_dir)

    print(f"\n✓ SANPO-Synthetic ready at: {out_dir}")
    print(f"  Sessions: {len(list((sessions_dir).iterdir()))}")
    print(f"\nTest with:")
    print(f"  python eval/evaluate.py --sanpo_root {DATA_DIR} --max_clips 20")


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


def _list_bucket_sessions(prefix: str) -> list[str]:
    """List session IDs in a GCS bucket prefix."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", prefix],
        capture_output=True, text=True, check=True
    )
    sessions = []
    for line in result.stdout.strip().split("\n"):
        name = line.strip().rstrip("/").split("/")[-1]
        if name:
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
    args = p.parse_args()

    if args.path == "a":
        path_a_synthetic(max_sessions=args.max_sessions)
    elif args.path == "b":
        path_b_real(max_sessions=args.max_sessions)
    elif args.path == "c":
        if not args.input_dir:
            p.error("--input_dir required for Path C")
        path_c_custom(input_dir=args.input_dir, fps_target=args.fps)


if __name__ == "__main__":
    main()

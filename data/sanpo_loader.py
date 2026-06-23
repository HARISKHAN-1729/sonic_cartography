"""
data/sanpo_loader.py
─────────────────────
Loader for the SANPO dataset (Scene Understanding, Accessibility,
and human Navigation POpulated).

SANPO is a large-scale egocentric video dataset specifically designed
for pedestrian navigation, with RGB video, depth maps, and semantic
segmentation for both indoor and outdoor scenes.

Dataset: https://github.com/google-research-datasets/sanpo-dataset
Paper: "SANPO: A Scene Understanding, Accessibility and Human
        Navigation Dataset", WACV 2025.

Expected directory structure:
    sanpo_root/
    ├── outdoor/
    │   ├── clip_0001/
    │   │   ├── rgb/
    │   │   │   ├── frame_000000.jpg
    │   │   │   └── ...
    │   │   ├── depth/
    │   │   │   ├── frame_000000.png   (uint16, millimetres)
    │   │   │   └── ...
    │   │   └── metadata.json
    │   └── ...
    └── indoor/
        └── ...
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, List
from PIL import Image
from dataclasses import dataclass


@dataclass
class SANPOClip:
    """A single video clip from the SANPO dataset."""
    clip_id: str
    scene_type: str           # "outdoor" | "indoor"
    rgb_frames: List[Path]    # sorted frame paths
    depth_frames: List[Path]  # matching depth frame paths
    fps: float
    metadata: dict


@dataclass
class SANPOFrame:
    """A single annotated frame with optional future ground-truth."""
    clip_id: str
    frame_idx: int
    rgb: np.ndarray            # (H, W, 3) uint8
    depth_m: np.ndarray        # (H, W) float32, metres
    timestamp_sec: float


class SANPOLoader:
    """
    Loads SANPO clips for offline evaluation.

    Usage
    -----
    loader = SANPOLoader(sanpo_root="data/sanpo/", split="test")
    for clip in loader.iter_clips():
        context_frames, future_frames = loader.get_context_and_future(
            clip, context_len=8, horizon_frames=[4, 8, 12, 16, 20]
        )
    """

    DEFAULT_FPS = 10.0
    DEPTH_SCALE_MM_TO_M = 0.001  # uint16 PNG stores depth in mm

    def __init__(
        self,
        sanpo_root: str | Path,
        split: str = "test",
        scene_types: Optional[List[str]] = None,
        max_clips: int = -1,
    ):
        self.root = Path(sanpo_root)
        self.split = split
        self.scene_types = scene_types or ["outdoor", "indoor"]
        self.max_clips = max_clips

        self._clips: List[SANPOClip] = []
        self._scan_dataset()

    def _scan_dataset(self) -> None:
        """Discover all clips in the dataset directory."""
        split_file = self.root / f"{self.split}_clips.txt"

        if split_file.exists():
            # Use official split file if available
            with open(split_file) as f:
                clip_ids = [line.strip() for line in f if line.strip()]
        else:
            # Fall back to scanning all directories
            clip_ids = None

        count = 0
        for scene_type in self.scene_types:
            scene_dir = self.root / scene_type
            if not scene_dir.exists():
                continue

            for clip_dir in sorted(scene_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue

                # Filter by official split if available
                if clip_ids is not None and clip_dir.name not in clip_ids:
                    continue

                rgb_dir   = clip_dir / "rgb"
                depth_dir = clip_dir / "depth"

                if not rgb_dir.exists():
                    continue

                rgb_frames = sorted(rgb_dir.glob("*.jpg")) + \
                             sorted(rgb_dir.glob("*.png"))
                rgb_frames.sort(key=lambda p: p.stem)

                depth_frames = []
                if depth_dir.exists():
                    depth_frames = sorted(depth_dir.glob("*.png"))
                    depth_frames.sort(key=lambda p: p.stem)

                if len(rgb_frames) < 20:  # skip very short clips
                    continue

                # Load metadata
                meta_path = clip_dir / "metadata.json"
                metadata = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        metadata = json.load(f)

                fps = float(metadata.get("fps", self.DEFAULT_FPS))

                self._clips.append(SANPOClip(
                    clip_id=clip_dir.name,
                    scene_type=scene_type,
                    rgb_frames=rgb_frames,
                    depth_frames=depth_frames,
                    fps=fps,
                    metadata=metadata,
                ))

                count += 1
                if self.max_clips > 0 and count >= self.max_clips:
                    return

    def __len__(self) -> int:
        return len(self._clips)

    def iter_clips(self) -> Iterator[SANPOClip]:
        for clip in self._clips:
            yield clip

    def load_frame(
        self, clip: SANPOClip, frame_idx: int
    ) -> SANPOFrame:
        """Load a single RGB + depth frame pair."""
        rgb_path = clip.rgb_frames[frame_idx]
        rgb = np.array(Image.open(rgb_path).convert("RGB"))

        # Load depth if available
        depth_m = np.zeros(rgb.shape[:2], dtype=np.float32)
        if frame_idx < len(clip.depth_frames):
            depth_raw = np.array(
                Image.open(clip.depth_frames[frame_idx]), dtype=np.uint16
            )
            depth_m = depth_raw.astype(np.float32) * self.DEPTH_SCALE_MM_TO_M
            # Resize to match RGB if necessary
            if depth_m.shape != rgb.shape[:2]:
                depth_img = Image.fromarray(depth_raw).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.NEAREST
                )
                depth_m = np.array(depth_img, dtype=np.float32) * \
                          self.DEPTH_SCALE_MM_TO_M

        timestamp_sec = frame_idx / clip.fps

        return SANPOFrame(
            clip_id=clip.clip_id,
            frame_idx=frame_idx,
            rgb=rgb,
            depth_m=depth_m,
            timestamp_sec=timestamp_sec,
        )

    def get_context_and_future(
        self,
        clip: SANPOClip,
        context_len: int = 8,
        horizon_frames: List[int] = (4, 8, 12, 16, 20),
        start_idx: Optional[int] = None,
    ) -> tuple[List[SANPOFrame], dict[int, SANPOFrame]]:
        """
        Sample a context window and future ground-truth frames from a clip.

        Parameters
        ----------
        clip            : SANPOClip
        context_len     : number of context frames
        horizon_frames  : list of future frame offsets to use as GT
        start_idx       : explicit start; if None, sampled randomly

        Returns
        -------
        context_frames  : list of SANPOFrame (length context_len)
        future_gt       : dict mapping horizon_offset → SANPOFrame
        """
        max_horizon = max(horizon_frames)
        max_start = len(clip.rgb_frames) - context_len - max_horizon - 1

        if max_start <= 0:
            # Clip too short: start from 0
            start_idx = 0
        elif start_idx is None:
            start_idx = np.random.randint(0, max_start)

        context_frames = [
            self.load_frame(clip, start_idx + i)
            for i in range(context_len)
        ]

        future_gt = {}
        for h in horizon_frames:
            future_idx = start_idx + context_len + h - 1
            if future_idx < len(clip.rgb_frames):
                future_gt[h] = self.load_frame(clip, future_idx)

        return context_frames, future_gt

    def horizon_offset_to_sec(
        self, clip: SANPOClip, horizon_frames: int
    ) -> float:
        return horizon_frames / clip.fps

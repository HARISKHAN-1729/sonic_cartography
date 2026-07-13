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
    mask_frames: List[Path] = None   # panoptic segmentation masks (optional)


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
    DEPTH_MAX_M = 100.0          # clamp sky/inf; downstream caps at ~8-10 m anyway

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
        """
        Discover clips by finding any directory that contains an ``rgb/``
        subfolder with frames. This handles both the legacy
        ``<root>/{outdoor,indoor}/<clip>/`` layout and the real SANPO layout
        produced by the downloader: ``<root>/sanpo-synthetic/sessions/<id>/``.
        """
        # Resolve a split id-list if present (SANPO uses *_session_ids.txt).
        clip_ids = None
        split_candidates = []
        for name in (f"{self.split}_session_ids.txt", f"{self.split}_clips.txt"):
            split_candidates += [
                self.root / "splits" / name,
                self.root / "sanpo-synthetic" / "splits" / name,
                self.root / name,
            ]
        for sp in split_candidates:
            if sp.exists():
                clip_ids = [l.strip() for l in sp.read_text().splitlines() if l.strip()]
                break

        count = 0
        seen = set()
        # Any folder with an rgb/ child is a clip, at any depth under root.
        for rgb_dir in sorted(self.root.glob("**/rgb")):
            if not rgb_dir.is_dir():
                continue
            clip_dir = rgb_dir.parent
            if clip_dir in seen:
                continue
            seen.add(clip_dir)

            if clip_ids is not None and clip_dir.name not in clip_ids:
                continue

            rgb_frames = sorted(
                list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")),
                key=lambda p: p.stem,
            )
            if len(rgb_frames) < 10:        # skip very short clips
                continue

            depth_dir = clip_dir / "depth"
            depth_frames = []
            if depth_dir.exists():
                # .npz (SANPO-Synthetic, metres) preferred; .png (uint16 mm) fallback.
                depth_frames = sorted(
                    list(depth_dir.glob("*.npz")) + list(depth_dir.glob("*.png")),
                    key=lambda p: p.stem,
                )

            mask_dir = clip_dir / "panoptic"
            mask_frames = []
            if mask_dir.exists():
                mask_frames = sorted(mask_dir.glob("*.png"), key=lambda p: p.stem)

            meta_path = clip_dir / "metadata.json"
            metadata = {}
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text())
                except Exception:
                    metadata = {}

            fps = float(metadata.get("fps", self.DEFAULT_FPS))
            low = str(clip_dir).lower()
            scene_type = ("indoor" if "indoor" in low else
                          "outdoor" if "outdoor" in low else "synthetic")

            self._clips.append(SANPOClip(
                clip_id=clip_dir.name,
                scene_type=scene_type,
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                fps=fps,
                metadata=metadata,
                mask_frames=mask_frames,
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
        H, W = rgb.shape[:2]

        depth_m = np.zeros((H, W), dtype=np.float32)
        if frame_idx < len(clip.depth_frames):
            dp = clip.depth_frames[frame_idx]
            if dp.suffix == ".npz":
                # SANPO-Synthetic: float32 metres under key 'depth'.
                with np.load(dp) as f:
                    key = "depth" if "depth" in f.files else f.files[0]
                    depth_m = f[key].astype(np.float32)
            else:
                # Legacy PNG: uint16 millimetres.
                depth_raw = np.array(Image.open(dp), dtype=np.uint16)
                depth_m = depth_raw.astype(np.float32) * self.DEPTH_SCALE_MM_TO_M

            # Sanitize: sky / non-finite -> far (ignored downstream), clamp absurd values.
            depth_m = np.nan_to_num(
                depth_m, nan=self.DEPTH_MAX_M,
                posinf=self.DEPTH_MAX_M, neginf=0.0,
            )
            depth_m = np.clip(depth_m, 0.0, self.DEPTH_MAX_M)

            # Resize to match RGB if the depth map differs in size.
            if depth_m.shape != (H, W):
                depth_img = Image.fromarray(depth_m).resize(
                    (W, H), Image.NEAREST
                )
                depth_m = np.array(depth_img, dtype=np.float32)

        timestamp_sec = frame_idx / clip.fps

        return SANPOFrame(
            clip_id=clip.clip_id,
            frame_idx=frame_idx,
            rgb=rgb,
            depth_m=depth_m,
            timestamp_sec=timestamp_sec,
        )

    def load_mask_path(self, clip, frame_idx):
        """Return the panoptic mask Path for a frame, or None if unavailable."""
        mf = getattr(clip, "mask_frames", None) or []
        if frame_idx < len(mf):
            return mf[frame_idx]
        return None

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

"""
models/depth_estimator.py
─────────────────────────
Wrapper around Depth Anything V2 for monocular depth estimation.

Given a single RGB frame (predicted or real), returns a metric-scale
depth map in metres (approximate), plus a normalised [0,1] map for
downstream spatial binning.

Pretrained checkpoints (HuggingFace Hub):
  depth-anything/Depth-Anything-V2-Small-hf   (~25M params, fastest)
  depth-anything/Depth-Anything-V2-Base-hf    (~98M params)
  depth-anything/Depth-Anything-V2-Large-hf   (~335M params, best)

Install:
  pip install transformers>=4.40.0
"""

from __future__ import annotations
import torch
import numpy as np
from PIL import Image
from typing import Union


class DepthEstimator:
    """
    Monocular depth estimator using Depth Anything V2.

    Usage
    -----
    depth_est = DepthEstimator.from_config(cfg.model.depth)
    depth_m, depth_norm = depth_est.estimate(frame_rgb)
    # depth_m:    np.ndarray (H, W) float32, approximate metres
    # depth_norm: np.ndarray (H, W) float32, in [0, 1], 0=far, 1=close
    """

    # Empirical scale factor to convert relative depth → approximate metres.
    # Tuned for outdoor pedestrian scenes. Adjust if using indoor data.
    RELATIVE_TO_METRIC_SCALE = 10.0  # relative=1.0 → ~10m in outdoor scenes

    def __init__(
        self,
        pipeline,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_config(cls, cfg) -> "DepthEstimator":
        from transformers import pipeline as hf_pipeline  # type: ignore

        checkpoint = cfg.get(
            "checkpoint", "depth-anything/Depth-Anything-V2-Large-hf"
        )
        device = cfg.get("device", "cuda")
        dtype_str = cfg.get("dtype", "float16")
        dtype = torch.float16 if dtype_str == "float16" else torch.float32

        torch_dtype = dtype if device == "cuda" else torch.float32

        pipe = hf_pipeline(
            task="depth-estimation",
            model=checkpoint,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch_dtype,
        )

        return cls(pipeline=pipe, device=device, dtype=dtype)

    @torch.no_grad()
    def estimate(
        self, frame: Union[np.ndarray, Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth for a single RGB frame.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3) uint8 or PIL.Image

        Returns
        -------
        depth_m    : np.ndarray (H, W) float32 — approx metric depth in metres
        depth_norm : np.ndarray (H, W) float32 — normalised [0,1], 1=closest
        """
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        result = self.pipeline(frame)
        depth_raw = np.array(result["depth"], dtype=np.float32)  # (H, W)

        # Depth Anything outputs inverse depth (higher = closer).
        # Normalise to [0, 1] where 1 = closest point in scene.
        d_min, d_max = depth_raw.min(), depth_raw.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_raw - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_raw)

        # Approximate metric depth: invert normalised map and scale
        # (0=far becomes large distance; 1=close becomes small distance)
        depth_m = (1.0 - depth_norm) * self.RELATIVE_TO_METRIC_SCALE
        depth_m = np.clip(depth_m, 0.1, self.RELATIVE_TO_METRIC_SCALE)

        return depth_m, depth_norm

    @torch.no_grad()
    def estimate_batch(
        self, frames: list
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Estimate depth for a list of frames."""
        return [self.estimate(f) for f in frames]

    @staticmethod
    def depth_at_bbox(
        depth_m: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> float:
        """
        Return median depth inside a bounding box.
        bbox: (x1, y1, x2, y2) in pixel coords.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_m.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        roi = depth_m[y1:y2, x1:x2]
        if roi.size == 0:
            return float("nan")
        return float(np.median(roi))

    @staticmethod
    def azimuth_from_bbox(
        bbox: tuple[int, int, int, int], frame_width: int
    ) -> float:
        """
        Compute approximate horizontal azimuth angle (degrees) of object
        centre relative to image centre.
        Positive = right, negative = left. Range ≈ [-60, +60] for typical FOV.
        Assumes ~90° horizontal FOV (common for smartphone cameras).
        """
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2.0
        relative_x = (cx / frame_width) - 0.5   # in [-0.5, +0.5]
        azimuth_deg = relative_x * 90.0          # map to ±45° (half 90° FOV)
        return float(azimuth_deg)

    @staticmethod
    def elevation_from_bbox(
        bbox: tuple[int, int, int, int], frame_height: int
    ) -> float:
        """
        Compute approximate vertical elevation angle (degrees).
        Positive = above horizon, negative = below. Range ≈ [-45, +45].
        """
        _, y1, _, y2 = bbox
        cy = (y1 + y2) / 2.0
        relative_y = 0.5 - (cy / frame_height)   # flip: image y is top-down
        elevation_deg = relative_y * 90.0
        return float(elevation_deg)

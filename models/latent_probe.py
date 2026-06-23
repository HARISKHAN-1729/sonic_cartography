"""
models/latent_probe.py
───────────────────────
A small, trainable probe that reads the navigation layout (object presence,
class, distance) directly off a V-JEPA 2 predicted token grid — so we never
need to decode pixels.

Design
──────
Input  : predicted future token grid  (B, Hg, Wg, D)
Heads  : per spatial cell ->
           objectness  (1)            : is a navigation-relevant object here?
           class       (num_classes)  : which category?
           depth       (1, metres)    : how far?
Azimuth / elevation are NOT learned — they come from the cell's (col, row)
position via the same FOV geometry used by the pixel detector, so the latent
and pixel paths are directly comparable.

The backbone (V-JEPA 2) stays frozen; only this probe is trained. That keeps
training to a single GPU and fast (see train_probe.py).

The conversion `probe_maps_to_detections(...)` is pure NumPy so it can be
unit-tested without torch and reused by the evaluator and the sonifier.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from models.object_detector import (
    Detection,
    _azimuth_from_bbox,
    _elevation_from_bbox,
)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
    _Base = nn.Module
except ImportError:
    _HAS_TORCH = False
    _Base = object


# Default navigation categories (must match configs/*.yaml detector.categories).
DEFAULT_CLASSES = [
    "person", "bicycle", "car", "motorbike", "bus", "truck",
    "traffic light", "stop sign", "bench", "door", "stairs",
    "curb", "pole", "fire hydrant", "bollard", "wall", "tree",
    "trash can", "table", "chair",
]


# ─────────────────────────────────────────────────────────────────────────────
# Trainable probe (torch)
# ─────────────────────────────────────────────────────────────────────────────

class LatentSpatialProbe(_Base):
    """
    Lightweight dense head over the (Hg, Wg, D) token grid.

    A 1x1-conv stack keeps each spatial cell independent (per-cell readout),
    which preserves the spatial-to-azimuth correspondence.

    forward(grid)  grid: (B, Hg, Wg, D)
      returns dict:
        "obj":   (B, Hg, Wg)             objectness logits
        "cls":   (B, Hg, Wg, num_cls)    class logits
        "depth": (B, Hg, Wg)             depth in metres (>=0)
    """

    def __init__(self, embed_dim: int, num_classes: int = len(DEFAULT_CLASSES),
                 hidden: int = 256, max_depth_m: float = 10.0):
        if not _HAS_TORCH:
            raise ImportError("LatentSpatialProbe needs torch.")
        super().__init__()
        self.num_classes = num_classes
        self.max_depth_m = max_depth_m
        self.trunk = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),  # mild spatial context
            nn.GELU(),
        )
        self.obj_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.cls_head = nn.Conv2d(hidden, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, grid):
        # (B, Hg, Wg, D) -> (B, D, Hg, Wg)
        x = grid.permute(0, 3, 1, 2).contiguous()
        h = self.trunk(x)
        obj = self.obj_head(h).squeeze(1)                  # (B, Hg, Wg)
        cls = self.cls_head(h).permute(0, 2, 3, 1)         # (B, Hg, Wg, C)
        depth = torch.sigmoid(self.depth_head(h).squeeze(1)) * self.max_depth_m
        return {"obj": obj, "cls": cls, "depth": depth}

    # --- persistence ---
    def save(self, path: str, meta: Optional[dict] = None):
        torch.save({"state_dict": self.state_dict(),
                    "num_classes": self.num_classes,
                    "max_depth_m": self.max_depth_m,
                    "meta": meta or {}}, path)

    @classmethod
    def load(cls, path: str, embed_dim: int, map_location="cpu") -> "LatentSpatialProbe":
        ckpt = torch.load(path, map_location=map_location)
        probe = cls(embed_dim=embed_dim,
                    num_classes=ckpt.get("num_classes", len(DEFAULT_CLASSES)),
                    max_depth_m=ckpt.get("max_depth_m", 10.0))
        probe.load_state_dict(ckpt["state_dict"])
        return probe


# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy conversion: probe maps -> Detection list  (torch-free, testable)
# ─────────────────────────────────────────────────────────────────────────────

def probe_maps_to_detections(
    obj_map: np.ndarray,            # (Hg, Wg) objectness PROBABILITIES [0,1]
    cls_map: np.ndarray,            # (Hg, Wg, C) class probabilities or logits
    depth_map: np.ndarray,          # (Hg, Wg) depth in metres
    classes: List[str] = None,
    image_size: int = 256,
    conf_threshold: float = 0.5,
    max_objects: int = 12,
) -> List[Detection]:
    """
    Turn per-cell probe outputs into a list of Detections compatible with the
    existing SceneSonifier and eval/metrics.py.

    Azimuth/elevation are derived from each firing cell's pixel-space bbox via
    the same helpers the pixel detector uses, so metrics are apples-to-apples.
    """
    classes = classes or DEFAULT_CLASSES
    Hg, Wg = obj_map.shape
    cell_h = image_size / Hg
    cell_w = image_size / Wg

    dets: List[Detection] = []
    for r in range(Hg):
        for c in range(Wg):
            score = float(obj_map[r, c])
            if score < conf_threshold:
                continue
            k = int(np.argmax(cls_map[r, c]))
            label = classes[k] if k < len(classes) else "object"

            x1 = int(c * cell_w); x2 = int((c + 1) * cell_w)
            y1 = int(r * cell_h); y2 = int((r + 1) * cell_h)
            bbox = (x1, y1, x2, y2)

            dets.append(Detection(
                label=label,
                confidence=score,
                bbox=bbox,
                depth_m=float(depth_map[r, c]),
                azimuth_deg=_azimuth_from_bbox(bbox, image_size),
                elevation_deg=_elevation_from_bbox(bbox, image_size),
            ))

    dets.sort(key=lambda d: d.depth_m)      # closest first
    return dets[:max_objects]


# ─────────────────────────────────────────────────────────────────────────────
# Teacher-label construction: pixel detections -> per-cell target maps
# Used by train_probe.py to distill the pixel pipeline into the latent probe.
# ─────────────────────────────────────────────────────────────────────────────

def detections_to_target_maps(
    detections: List[Detection],
    grid_h: int,
    grid_w: int,
    image_size: int = 256,
    classes: List[str] = None,
):
    """
    Rasterize teacher detections (from Depth Anything + YOLO-World on the REAL
    frame) into per-cell training targets.

    Returns (obj, cls_idx, depth, valid) as NumPy arrays:
        obj    : (Hg, Wg) float {0,1}
        cls_idx: (Hg, Wg) int64 class index (-1 where no object)
        depth  : (Hg, Wg) float metres (0 where no object)
        valid  : (Hg, Wg) bool — cells that carry an object (for masked loss)
    """
    classes = classes or DEFAULT_CLASSES
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    obj = np.zeros((grid_h, grid_w), dtype=np.float32)
    cls_idx = np.full((grid_h, grid_w), -1, dtype=np.int64)
    depth = np.zeros((grid_h, grid_w), dtype=np.float32)

    cell_h = image_size / grid_h
    cell_w = image_size / grid_w

    # For overlapping objects in a cell, keep the closest (smallest depth).
    best_depth = np.full((grid_h, grid_w), np.inf, dtype=np.float32)

    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        c = int(min(grid_w - 1, max(0, cx // cell_w)))
        r = int(min(grid_h - 1, max(0, cy // cell_h)))
        if d.depth_m < best_depth[r, c]:
            best_depth[r, c] = d.depth_m
            obj[r, c] = 1.0
            cls_idx[r, c] = cls_to_idx.get(d.label, -1)
            depth[r, c] = d.depth_m

    valid = obj > 0.5
    return obj, cls_idx, depth, valid

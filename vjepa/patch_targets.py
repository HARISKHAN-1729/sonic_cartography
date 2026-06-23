"""
vjepa/patch_targets.py
───────────────────────
Build per-patch supervision targets for the SceneProbe from full-resolution
SANPO annotations.

V-JEPA 2 encodes a frame into a grid of patch tokens (e.g. 16×16 for a 256px
input with patch size 16). To train a probe that reads navigation attributes
off those tokens, we need a label per patch:

    class_grid : (Gh, Gw) int   — dominant NAV class id in that patch
    depth_grid : (Gh, Gw) float — median metric depth (m) of that patch
    valid_grid : (Gh, Gw) bool  — whether the patch has usable depth

These are produced by downsampling the full-res panoptic id map and depth map
onto the patch grid. Pure NumPy; no torch, fully CPU-testable.
"""

from __future__ import annotations
import numpy as np

from vjepa.class_map import BACKGROUND_ID, NUM_CLASSES


def panoptic_to_nav_id_map(
    panoptic_ids: np.ndarray,
    sanpo_id_to_nav_id: dict[int, int],
) -> np.ndarray:
    """
    Convert a full-res SANPO panoptic id map (H, W) into a NAV-class id map.

    panoptic_ids may encode instances as (semantic_id * 1000 + instance);
    we recover the semantic id by integer-dividing when values exceed the
    label table range. Unknown ids map to background.
    """
    out = np.zeros_like(panoptic_ids, dtype=np.int64)
    max_known = max(sanpo_id_to_nav_id) if sanpo_id_to_nav_id else 0
    for pid in np.unique(panoptic_ids):
        sem = int(pid)
        if sem not in sanpo_id_to_nav_id and sem > max_known:
            sem = sem // 1000  # strip instance encoding
        out[panoptic_ids == pid] = sanpo_id_to_nav_id.get(sem, BACKGROUND_ID)
    return out


def build_patch_targets(
    nav_id_map: np.ndarray,      # (H, W) int  NAV class ids
    depth_m: np.ndarray,         # (H, W) float metres
    grid_h: int,
    grid_w: int,
    min_valid_depth: float = 0.1,
    max_valid_depth: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample full-res annotations onto a (grid_h, grid_w) patch grid.

    For each patch cell:
      • class = most frequent NON-background NAV id in the cell if any
                non-background pixels exist, else background.
      • depth = median depth over valid pixels in the cell.
      • valid = whether any valid-depth pixel exists.

    Returns (class_grid, depth_grid, valid_grid).
    """
    H, W = nav_id_map.shape
    ys = np.linspace(0, H, grid_h + 1).astype(int)
    xs = np.linspace(0, W, grid_w + 1).astype(int)

    class_grid = np.zeros((grid_h, grid_w), dtype=np.int64)
    depth_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    valid_grid = np.zeros((grid_h, grid_w), dtype=bool)

    for gy in range(grid_h):
        for gx in range(grid_w):
            cls_cell = nav_id_map[ys[gy]:ys[gy + 1], xs[gx]:xs[gx + 1]]
            dep_cell = depth_m[ys[gy]:ys[gy + 1], xs[gx]:xs[gx + 1]]

            # Class: most frequent non-background id in the cell.
            fg = cls_cell[cls_cell != BACKGROUND_ID]
            if fg.size > 0:
                counts = np.bincount(fg, minlength=NUM_CLASSES)
                class_grid[gy, gx] = int(np.argmax(counts))
            else:
                class_grid[gy, gx] = BACKGROUND_ID

            # Depth: median over valid pixels.
            valid = (dep_cell >= min_valid_depth) & (dep_cell <= max_valid_depth)
            if valid.any():
                depth_grid[gy, gx] = float(np.median(dep_cell[valid]))
                valid_grid[gy, gx] = True

    return class_grid, depth_grid, valid_grid

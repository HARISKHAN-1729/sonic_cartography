"""
vjepa/detections_from_grid.py
──────────────────────────────
Convert a per-patch (class, depth, confidence) grid — produced by the probe on
a predicted V-JEPA latent — into a list of Detection objects compatible with
the existing eval/metrics.py and audio/scene_sonifier.py.

This is the bridge between V-JEPA's latent-space prediction and the rest of
the PSC pipeline: instead of a decoded RGB frame + off-the-shelf detector, the
probe yields a labelled patch grid, and we group contiguous same-class patches
into detections, reading distance from patch depth and azimuth/elevation from
patch position.

Pure NumPy + scipy.ndimage; fully CPU-testable.
"""

from __future__ import annotations
import numpy as np
from typing import List

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from models.object_detector import Detection
from vjepa.class_map import ID_TO_CLASS, BACKGROUND_ID


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connectivity connected components. Falls back to a tiny flood fill."""
    if _HAS_SCIPY:
        lab, n = _ndi.label(mask)
        return lab, int(n)
    # Minimal BFS fallback (rarely used; scipy is in requirements).
    lab = np.zeros_like(mask, dtype=np.int32)
    n = 0
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    if 0 <= y < h and 0 <= x < w and mask[y, x] and lab[y, x] == 0:
                        lab[y, x] = n
                        stack += [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]
    return lab, n


def detections_from_grid(
    class_grid: np.ndarray,      # (Gh, Gw) int   NAV class ids
    depth_grid: np.ndarray,      # (Gh, Gw) float metres
    conf_grid: np.ndarray | None,  # (Gh, Gw) float [0,1] or None
    frame_width: int,
    frame_height: int,
    patch_size: int = 16,
    fov_horizontal_deg: float = 90.0,
    fov_vertical_deg: float = 60.0,
    min_component_patches: int = 1,
    conf_threshold: float = 0.3,
) -> List[Detection]:
    """
    Group contiguous same-class patches into Detection objects.

    Parameters
    ----------
    class_grid    : per-patch predicted NAV class id
    depth_grid    : per-patch predicted depth (metres)
    conf_grid     : per-patch confidence (max class prob); None -> all 1.0
    frame_width/height : pixel size the grid maps onto (for bbox + angles)
    patch_size    : pixels per patch (grid_w * patch_size ≈ frame_width)
    fov_*         : camera field of view used to convert position -> angle
    min_component_patches : drop components smaller than this
    conf_threshold : ignore patches below this confidence

    Returns
    -------
    list[Detection], sorted by distance (closest first).
    """
    Gh, Gw = class_grid.shape
    if conf_grid is None:
        conf_grid = np.ones((Gh, Gw), dtype=np.float32)

    dets: List[Detection] = []

    # Process each foreground class independently so adjacent different-class
    # regions don't merge.
    for cls_id in np.unique(class_grid):
        if cls_id == BACKGROUND_ID:
            continue
        mask = (class_grid == cls_id) & (conf_grid >= conf_threshold)
        if not mask.any():
            continue

        lab, n = _label_components(mask)
        for comp in range(1, n + 1):
            ys, xs = np.where(lab == comp)
            if ys.size < min_component_patches:
                continue

            # bbox in pixel coords
            x1 = int(xs.min() * patch_size)
            y1 = int(ys.min() * patch_size)
            x2 = int((xs.max() + 1) * patch_size)
            y2 = int((ys.max() + 1) * patch_size)

            # depth: median over valid component patches
            comp_depths = depth_grid[ys, xs]
            comp_depths = comp_depths[comp_depths > 0]
            depth_m = float(np.median(comp_depths)) if comp_depths.size else 5.0

            # angles from component centroid
            cx = float(xs.mean()) + 0.5
            cy = float(ys.mean()) + 0.5
            azimuth = ((cx / Gw) - 0.5) * fov_horizontal_deg
            elevation = (0.5 - (cy / Gh)) * fov_vertical_deg

            confidence = float(conf_grid[ys, xs].mean())

            dets.append(Detection(
                label=ID_TO_CLASS[int(cls_id)],
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                depth_m=depth_m,
                azimuth_deg=float(azimuth),
                elevation_deg=float(elevation),
            ))

    dets.sort(key=lambda d: d.depth_m)
    return dets

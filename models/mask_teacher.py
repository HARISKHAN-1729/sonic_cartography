"""
models/mask_teacher.py
───────────────────────
Ground-truth scene-layout extractor for SANPO: panoptic segmentation mask +
metric depth  ->  list[Detection].

WHY THIS EXISTS (replaces YOLO-World + Depth Anything as teacher/GT)
─────────────────────────────────────────────────────────────────────
YOLO-World is trained on real photos and fires on ~1.3 objects/frame on
SANPO-Synthetic's rendered imagery — starving the probe of labels and
corrupting the eval ground truth. SANPO-Synthetic ships PIXEL-PERFECT
panoptic masks and metric depth, so we extract the layout directly from
ground truth instead. Following the SANPO paper's own object definition
(Waghmare et al., WACV 2025, Fig. 7): an "object" is a connected component
of each unique panoptic label covering at least 100 px.

ROBUST BY CONSTRUCTION
──────────────────────
- Mask PNGs are auto-decoded (8-bit / 16-bit single channel, or RGB packed
  R + 256·G + 65536·B), and the semantic/instance split (e.g. value//1000)
  is auto-calibrated from a sample of masks.
- Class names come from a taxonomy json/csv if present in the dataset;
  otherwise stable placeholder names "c<semantic_id>" are used. Because the
  SAME extractor produces probe-training labels AND eval ground truth,
  label matching in the hit-rate metric is consistent either way.
- Stuff (sky/road/sidewalk/…) is filtered geometrically: components larger
  than `max_area_frac` of the frame, or farther than `max_range_m`, are
  excluded — no hand-written class blacklist required.
"""

from __future__ import annotations

import json
import csv as _csv
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from PIL import Image
from scipy import ndimage

from models.object_detector import (
    Detection, _azimuth_from_bbox, _elevation_from_bbox,
)


class MaskTeacher:
    """
    Usage
    -----
        teacher = MaskTeacher(min_px=100, max_area_frac=0.25, max_range_m=20.0)
        teacher.calibrate(sample_mask_paths)         # once, auto-detects encoding
        dets = teacher.extract(mask_path_or_array, depth_m, frame_hw=(H, W))
        teacher.class_names                           # stable vocabulary
    """

    def __init__(
        self,
        min_px: int = 100,             # SANPO paper's object definition
        max_area_frac: float = 0.25,   # bigger than this => stuff, skip
        max_range_m: float = 20.0,     # beyond this: not navigation-relevant
        max_objects: int = 24,
        taxonomy_path: Optional[str] = None,
    ):
        self.min_px = min_px
        self.max_area_frac = max_area_frac
        self.max_range_m = max_range_m
        self.max_objects = max_objects

        self.divisor: int = 1          # panoptic value // divisor = semantic id
        self._calibrated = False
        self._id_to_name: Dict[int, str] = {}
        self.class_names: List[str] = []
        self._name_of: Dict[int, str] = {}

        if taxonomy_path:
            self._load_taxonomy(taxonomy_path)

    # ── Calibration ─────────────────────────────────────────────────────────

    def calibrate(self, sample_masks: List) -> None:
        """
        Inspect a handful of masks to auto-detect the panoptic encoding and
        build the semantic class vocabulary. `sample_masks` is a list of
        paths or arrays (10-30 masks from different sessions is plenty).
        """
        all_vals = set()
        for m in sample_masks:
            lab = self._decode(m)
            vals, counts = np.unique(lab, return_counts=True)
            # ignore ultra-rare speckle values in calibration
            all_vals.update(int(v) for v, c in zip(vals, counts) if c >= 20)

        vals = sorted(all_vals)
        if not vals:
            raise ValueError("MaskTeacher.calibrate: no label values found.")

        # Detect an instance divisor: panoptic values like k*1000 + i.
        # Accept a divisor d when it yields a plausibly small semantic set and
        # either compresses many instances into few classes, or every nonzero
        # value is >= d (pure k*d+i encoding with possibly 1 instance/class).
        self.divisor = 1
        nz = [v for v in vals if v > 0]
        for d in (1000, 256):
            if not nz or max(nz) < d:
                continue
            sems = {v // d for v in vals}
            compresses = len(sems) < 0.6 * len(vals)
            pure = all(v >= d for v in nz)
            if 2 <= len(sems) <= 64 and (compresses or pure):
                self.divisor = d
                break

        sem_ids = sorted({v // self.divisor for v in vals})
        self._name_of = {
            s: self._id_to_name.get(s, f"c{s}") for s in sem_ids
        }
        self.class_names = [self._name_of[s] for s in sem_ids]
        self._calibrated = True
        print(f"  [MaskTeacher] calibrated: divisor={self.divisor}, "
              f"{len(self.class_names)} semantic classes "
              f"({'taxonomy names' if self._id_to_name else 'placeholder names'})")

    def state(self) -> dict:
        """Serializable calibration state (store in the probe checkpoint)."""
        return {"divisor": self.divisor,
                "class_names": list(self.class_names),
                "sem_name_map": {str(k): v for k, v in self._name_of.items()},
                "min_px": self.min_px, "max_area_frac": self.max_area_frac,
                "max_range_m": self.max_range_m}

    @classmethod
    def from_state(cls, st: dict) -> "MaskTeacher":
        t = cls(min_px=st.get("min_px", 100),
                max_area_frac=st.get("max_area_frac", 0.25),
                max_range_m=st.get("max_range_m", 20.0))
        t.divisor = int(st["divisor"])
        t.class_names = list(st["class_names"])
        t._name_of = {int(k): v for k, v in st.get("sem_name_map", {}).items()}
        t._calibrated = True
        return t

    # ── Extraction (the core call) ──────────────────────────────────────────

    def extract(
        self,
        mask,                       # path | np.ndarray label map | PIL.Image
        depth_m: np.ndarray,        # (H, W) metric depth (SANPO GT)
        frame_hw: Optional[tuple] = None,
    ) -> List[Detection]:
        """Panoptic mask + depth  ->  Detection list (closest first)."""
        if not self._calibrated:
            self.calibrate([mask])

        lab = self._decode(mask)
        H, W = lab.shape
        if depth_m.shape != (H, W):
            depth_m = np.array(
                Image.fromarray(depth_m).resize((W, H), Image.NEAREST),
                dtype=np.float32)

        area = H * W
        dets: List[Detection] = []

        for val in np.unique(lab):
            region = lab == val
            n_total = int(region.sum())
            if n_total < self.min_px:
                continue

            # SANPO object definition: connected components per unique label
            comps, n_comp = ndimage.label(region)
            for ci in range(1, n_comp + 1):
                comp = comps == ci
                npx = int(comp.sum())
                if npx < self.min_px:
                    continue
                if npx > self.max_area_frac * area:
                    continue                       # stuff (sky/road/...)

                d_vals = depth_m[comp]
                d_vals = d_vals[np.isfinite(d_vals) & (d_vals > 0)]
                if d_vals.size == 0:
                    continue
                med_d = float(np.median(d_vals))
                if med_d > self.max_range_m:
                    continue                       # too far to matter

                ys, xs = np.nonzero(comp)
                x1, x2 = int(xs.min()), int(xs.max()) + 1
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                bbox = (x1, y1, x2, y2)

                sem = int(val) // self.divisor
                label = self._name_of.get(sem, f"c{sem}")

                dets.append(Detection(
                    label=label,
                    confidence=1.0,                # ground truth
                    bbox=bbox,
                    depth_m=med_d,
                    azimuth_deg=_azimuth_from_bbox(bbox, W),
                    elevation_deg=_elevation_from_bbox(bbox, H),
                ))

        dets.sort(key=lambda d: d.depth_m)
        return dets[: self.max_objects]

    # ── Decoding & taxonomy ─────────────────────────────────────────────────

    @staticmethod
    def _decode(mask) -> np.ndarray:
        """Auto-decode a mask (path/PIL/array) into an int32 label map."""
        if isinstance(mask, (str, Path)):
            mask = Image.open(mask)
        if isinstance(mask, Image.Image):
            arr = np.array(mask)
        else:
            arr = np.asarray(mask)

        if arr.ndim == 3:                          # RGB(A) packed
            arr = arr[..., :3].astype(np.int32)
            if (arr[..., 0] == arr[..., 1]).all() and \
               (arr[..., 1] == arr[..., 2]).all():
                return arr[..., 0]                 # grayscale stored as RGB
            return arr[..., 0] + 256 * arr[..., 1] + 65536 * arr[..., 2]
        return arr.astype(np.int32)                # 8/16-bit single channel

    def _load_taxonomy(self, path: str) -> None:
        """Load id->name from a SANPO taxonomy json or csv, if available."""
        p = Path(path)
        if not p.exists():
            return
        try:
            if p.suffix == ".json":
                data = json.loads(p.read_text())
                items = data.items() if isinstance(data, dict) else \
                    ((d.get("id"), d.get("name")) for d in data)
                for k, v in items:
                    if k is not None and v:
                        self._id_to_name[int(k)] = str(v).strip().lower()
            else:                                   # csv: id,name columns
                with open(p) as fh:
                    for row in _csv.DictReader(fh):
                        ks = [c for c in row if "id" in c.lower()]
                        ns = [c for c in row if "name" in c.lower()
                              or "label" in c.lower()]
                        if ks and ns:
                            self._id_to_name[int(row[ks[0]])] = \
                                str(row[ns[0]]).strip().lower()
            if self._id_to_name:
                print(f"  [MaskTeacher] taxonomy loaded: "
                      f"{len(self._id_to_name)} named classes")
        except Exception as e:
            print(f"  [MaskTeacher] taxonomy load failed ({e}); "
                  f"using placeholder names")


# ── Layout-space baselines (share the mask extractor) ───────────────────────

def warp_layout_forward(
    mask_lab: np.ndarray,          # (H, W) decoded label map at time t
    depth_m: np.ndarray,           # (H, W) depth at time t
    flow: np.ndarray,              # (H, W, 2) optical flow t-1 -> t
    k: float,                      # how many inter-frame steps ahead
) -> tuple:
    """
    Constant-velocity 'flow' baseline in LAYOUT space: warp the ground-truth
    mask (nearest) and depth (linear) forward by k·flow, so the baseline is
    scored with the same GT-quality extraction as everything else — only the
    dynamics assumption differs.
    """
    import cv2
    H, W = mask_lab.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xs - k * flow[..., 0]).astype(np.float32)
    map_y = (ys - k * flow[..., 1]).astype(np.float32)
    warped_mask = cv2.remap(mask_lab.astype(np.float32), map_x, map_y,
                            interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REPLICATE).astype(np.int32)
    warped_depth = cv2.remap(depth_m.astype(np.float32), map_x, map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return warped_mask, warped_depth


def compute_flow(prev_rgb: np.ndarray, last_rgb: np.ndarray) -> np.ndarray:
    """Dense Farneback flow between the last two context frames."""
    import cv2
    g0 = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

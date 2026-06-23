"""
models/object_detector.py
──────────────────────────
Wrapper around YOLO-World (ultralytics) for open-vocabulary
object detection on predicted egocentric frames.

YOLO-World supports arbitrary text-defined categories at inference
time — we pass the navigation-relevant category list from config.

Install:
  pip install ultralytics>=8.2.0
"""

from __future__ import annotations
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Union


@dataclass
class Detection:
    """Single detected object."""
    label: str               # object class name
    confidence: float        # detection confidence [0, 1]
    bbox: tuple              # (x1, y1, x2, y2) in pixels
    depth_m: float = 0.0     # filled in by pipeline using depth map
    azimuth_deg: float = 0.0 # horizontal angle from centre
    elevation_deg: float = 0.0


class ObjectDetector:
    """
    Open-vocabulary object detector using YOLO-World.

    Usage
    -----
    detector = ObjectDetector.from_config(cfg.model.detector)
    detections = detector.detect(frame_rgb)
    # returns list[Detection]
    """

    def __init__(
        self,
        model,
        categories: List[str],
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ):
        self.model = model
        self.categories = categories
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Set the open-vocabulary categories
        self.model.set_classes(categories)

    @classmethod
    def from_config(cls, cfg) -> "ObjectDetector":
        from ultralytics import YOLOWorld  # type: ignore

        checkpoint = cfg.get("checkpoint", "yolov8x-worldv2.pt")
        device = cfg.get("device", "cuda")

        model = YOLOWorld(checkpoint)
        model.to(device)

        categories = list(cfg.get(
            "categories",
            ["person", "car", "bicycle", "stairs", "door", "wall", "curb"],
        ))

        return cls(
            model=model,
            categories=categories,
            conf_threshold=cfg.get("conf_threshold", 0.35),
            iou_threshold=cfg.get("iou_threshold", 0.45),
            device=device,
        )

    def detect(
        self,
        frame: Union[np.ndarray, Image.Image],
        depth_m: np.ndarray = None,
    ) -> List[Detection]:
        """
        Detect objects in a single frame.

        Parameters
        ----------
        frame   : RGB image (np.ndarray HxWx3 uint8 or PIL.Image)
        depth_m : optional depth map (H, W) float32 in metres

        Returns
        -------
        list of Detection (sorted by distance, closest first)
        """
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)
        else:
            frame_np = frame

        h, w = frame_np.shape[:2]

        results = self.model.predict(
            source=frame_np,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_idx = int(boxes.cls[i].cpu().numpy())

                if cls_idx >= len(self.categories):
                    continue

                label = self.categories[cls_idx]
                x1, y1, x2, y2 = (
                    int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                )
                bbox = (x1, y1, x2, y2)

                # Compute spatial angles
                azimuth = _azimuth_from_bbox(bbox, w)
                elevation = _elevation_from_bbox(bbox, h)

                # Compute depth if map provided
                depth = 5.0  # default fallback
                if depth_m is not None:
                    depth = _depth_at_bbox(depth_m, bbox)
                    if np.isnan(depth):
                        depth = 5.0

                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=bbox,
                        depth_m=depth,
                        azimuth_deg=azimuth,
                        elevation_deg=elevation,
                    )
                )

        # Sort by depth: closest first (most urgent for navigation)
        detections.sort(key=lambda d: d.depth_m)
        return detections

    def detect_batch(
        self,
        frames: List,
        depth_maps: List[np.ndarray] = None,
    ) -> List[List[Detection]]:
        """Detect objects in a list of frames."""
        results = []
        for i, frame in enumerate(frames):
            dm = depth_maps[i] if depth_maps is not None else None
            results.append(self.detect(frame, depth_m=dm))
        return results


# ── Standalone geometry helpers ────────────────────────────────────────────

def _depth_at_bbox(
    depth_m: np.ndarray, bbox: tuple[int, int, int, int]
) -> float:
    x1, y1, x2, y2 = bbox
    h, w = depth_m.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    roi = depth_m[y1:y2, x1:x2]
    return float(np.nanmedian(roi)) if roi.size > 0 else float("nan")


def _azimuth_from_bbox(
    bbox: tuple[int, int, int, int], frame_width: int
) -> float:
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0
    return ((cx / frame_width) - 0.5) * 90.0


def _elevation_from_bbox(
    bbox: tuple[int, int, int, int], frame_height: int
) -> float:
    _, y1, _, y2 = bbox
    cy = (y1 + y2) / 2.0
    return (0.5 - (cy / frame_height)) * 90.0

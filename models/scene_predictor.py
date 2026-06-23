"""
models/scene_predictor.py
──────────────────────────
Unifies every backend behind ONE contract:

    predict_scene_at_horizons(context_frames, horizons_sec)
        -> dict[horizon_sec -> list[Detection]]

This is the key to a clean ablation: the evaluator calls this and never cares
whether the predicted scene came from pixels or latents.

Two implementations
────────────────────
  PixelScenePredictor   context -> world_model (flow|nwm|last_frame)
                        -> predicted RGB frame -> Depth Anything + YOLO-World
                        -> Detections

  LatentScenePredictor  context -> V-JEPA 2 -> predicted token grid
                        -> trained probe -> Detections   (no pixels)

Factory `build_scene_predictor(cfg)` reads `scene_predictor.type` ("pixel" or
"latent") and wires the right pieces. Pixel reuses models/world_model.py,
models/depth_estimator.py, models/object_detector.py. Latent reuses
models/vjepa_world_model.py + models/latent_probe.py.
"""

from __future__ import annotations

from typing import List, Dict, Protocol, runtime_checkable
import numpy as np

from models.object_detector import Detection


@runtime_checkable
class ScenePredictor(Protocol):
    def predict_scene_at_horizons(
        self,
        context_frames: List[np.ndarray],
        horizon_times_sec: List[float],
    ) -> Dict[float, List[Detection]]:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Pixel path: world model -> frame -> depth + detector -> detections
# ─────────────────────────────────────────────────────────────────────────────

class PixelScenePredictor:
    def __init__(self, world_model, depth_est, detector, action: str = "walk_forward"):
        self.world_model = world_model
        self.depth_est = depth_est
        self.detector = detector
        self.action = action

    @classmethod
    def from_config(cls, cfg):
        from models.world_model import build_world_model
        from models.depth_estimator import DepthEstimator
        from models.object_detector import ObjectDetector
        return cls(
            world_model=build_world_model(cfg),
            depth_est=DepthEstimator.from_config(cfg["model"]["depth"]),
            detector=ObjectDetector.from_config(cfg["model"]["detector"]),
            action=cfg.get("world_model", {}).get("action", "walk_forward"),
        )

    def predict_scene_at_horizons(self, context_frames, horizon_times_sec):
        frames = self.world_model.predict_at_horizons(
            context_frames, horizon_times_sec, action=self.action
        )
        out: Dict[float, List[Detection]] = {}
        for h, frame in frames.items():
            depth_m, _ = self.depth_est.estimate(frame)
            out[h] = self.detector.detect(frame, depth_m=depth_m)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Latent path: V-JEPA 2 -> predicted token grid -> probe -> detections
# ─────────────────────────────────────────────────────────────────────────────

class LatentScenePredictor:
    def __init__(self, vjepa, probe, classes=None, image_size: int = 256,
                 conf_threshold: float = 0.5, max_objects: int = 12):
        self.vjepa = vjepa
        self.probe = probe
        self.classes = classes
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.max_objects = max_objects

    @classmethod
    def from_config(cls, cfg):
        import torch
        from models.vjepa_world_model import VJEPA2WorldModel
        from models.latent_probe import LatentSpatialProbe, DEFAULT_CLASSES

        vcfg = cfg["model"]["vjepa"]
        vjepa = VJEPA2WorldModel.from_config(vcfg)

        probe_path = cfg.get("scene_predictor", {}).get("probe_path",
                                                         "checkpoints/probe.pt")
        probe = LatentSpatialProbe.load(probe_path, embed_dim=vjepa.embed_dim)
        probe.eval().to(vjepa.device)

        sp = cfg.get("scene_predictor", {})
        return cls(
            vjepa=vjepa,
            probe=probe,
            classes=list(cfg["model"]["detector"].get("categories", DEFAULT_CLASSES)),
            image_size=int(vcfg.get("image_size", 256)),
            conf_threshold=float(sp.get("conf_threshold", 0.5)),
            max_objects=int(sp.get("max_objects", 12)),
        )

    def predict_scene_at_horizons(self, context_frames, horizon_times_sec):
        import torch
        from models.latent_probe import probe_maps_to_detections

        grids = self.vjepa.predict_latent_at_horizons(
            context_frames, horizon_times_sec
        )  # dict[h -> (Hg, Wg, D)] numpy

        out: Dict[float, List[Detection]] = {}
        for h, grid in grids.items():
            with torch.no_grad():
                g = torch.from_numpy(grid).unsqueeze(0).to(
                    next(self.probe.parameters()).device
                ).float()
                pred = self.probe(g)
                obj = torch.sigmoid(pred["obj"])[0].cpu().numpy()
                cls = torch.softmax(pred["cls"], dim=-1)[0].cpu().numpy()
                depth = pred["depth"][0].cpu().numpy()
            out[h] = probe_maps_to_detections(
                obj_map=obj, cls_map=cls, depth_map=depth,
                classes=self.classes, image_size=self.image_size,
                conf_threshold=self.conf_threshold, max_objects=self.max_objects,
            )
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_scene_predictor(cfg) -> ScenePredictor:
    """
    Reads cfg.scene_predictor.type:
        "pixel"  -> PixelScenePredictor  (flow|nwm|last_frame via world_model)
        "latent" -> LatentScenePredictor (V-JEPA 2 + trained probe)
    """
    sp = cfg.get("scene_predictor", {}) if hasattr(cfg, "get") else {}
    kind = sp.get("type", "pixel")
    if kind == "pixel":
        return PixelScenePredictor.from_config(cfg)
    if kind == "latent":
        return LatentScenePredictor.from_config(cfg)
    raise ValueError(f"Unknown scene_predictor.type '{kind}' (use 'pixel' or 'latent').")

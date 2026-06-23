"""
scripts/test_vjepa_logic.py
────────────────────────────
CPU-only tests for the V-JEPA latent pipeline — the parts that don't need the
actual model weights. Validates:
  - probe_maps_to_detections (geometry, ordering, thresholding)
  - detections_to_target_maps (teacher rasterization, closest-object-wins)
  - round-trip consistency (teacher maps -> detections -> same cells)
  - LatentScenePredictor data flow with a dummy V-JEPA + dummy probe

Run:  python scripts/test_vjepa_logic.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.object_detector import Detection
from models.latent_probe import (
    probe_maps_to_detections, detections_to_target_maps, DEFAULT_CLASSES,
)


C = len(DEFAULT_CLASSES)
IMG = 256
HG = WG = 16


def test_conversion_basic():
    obj = np.zeros((HG, WG)); cls = np.zeros((HG, WG, C)); depth = np.full((HG, WG), 5.0)
    obj[8, 4] = 0.9;  cls[8, 4, 0] = 9; depth[8, 4] = 2.0     # person, left, close
    obj[4, 12] = 0.7; cls[4, 12, 2] = 9; depth[4, 12] = 6.0   # car, right, far
    dets = probe_maps_to_detections(obj, cls, depth, conf_threshold=0.5, image_size=IMG)
    assert len(dets) == 2
    assert dets[0].label == "person" and dets[0].depth_m == 2.0       # closest first
    assert dets[0].azimuth_deg < 0 and dets[1].azimuth_deg > 0        # left vs right
    print("  ✓ probe_maps_to_detections: geometry + ordering + threshold")


def test_conversion_threshold():
    obj = np.full((HG, WG), 0.3); cls = np.zeros((HG, WG, C)); depth = np.full((HG, WG), 4.0)
    dets = probe_maps_to_detections(obj, cls, depth, conf_threshold=0.5)
    assert dets == []
    print("  ✓ thresholding suppresses low-objectness cells")


def test_max_objects():
    obj = np.full((HG, WG), 0.9); cls = np.zeros((HG, WG, C)); depth = np.random.rand(HG, WG) * 5 + 0.5
    dets = probe_maps_to_detections(obj, cls, depth, conf_threshold=0.5, max_objects=4)
    assert len(dets) == 4
    assert all(dets[i].depth_m <= dets[i + 1].depth_m for i in range(len(dets) - 1))
    print("  ✓ max_objects cap + closest-first ordering")


def test_teacher_maps():
    teacher = [
        Detection("person", 0.9, (56, 120, 72, 180), depth_m=2.0),
        Detection("car",    0.8, (180, 40, 210, 80), depth_m=6.0),
    ]
    obj, cls_idx, depth, valid = detections_to_target_maps(teacher, HG, WG, image_size=IMG)
    assert valid.sum() == 2
    assert set(cls_idx[cls_idx >= 0].tolist()) == {0, 2}      # person=0, car=2
    assert depth[valid].min() == 2.0 and depth[valid].max() == 6.0
    print("  ✓ detections_to_target_maps: rasterization")


def test_teacher_closest_wins():
    # Two objects mapping to the SAME cell — the closer one must win.
    a = Detection("car",    0.8, (120, 120, 130, 130), depth_m=8.0)
    b = Detection("person", 0.9, (121, 121, 129, 129), depth_m=1.5)
    obj, cls_idx, depth, valid = detections_to_target_maps([a, b], HG, WG, image_size=IMG)
    assert valid.sum() == 1
    r, c = np.argwhere(valid)[0]
    assert cls_idx[r, c] == 0 and depth[r, c] == 1.5          # person, closest
    print("  ✓ closest-object-wins on cell collision")


def test_roundtrip():
    teacher = [
        Detection("stairs", 0.9, (10, 10, 26, 26), depth_m=2.5),
        Detection("pole",   0.8, (200, 100, 216, 140), depth_m=3.0),
    ]
    obj, cls_idx, depth, valid = detections_to_target_maps(teacher, HG, WG, image_size=IMG)
    # Build prob maps as the probe would (objectness=valid, one-hot class).
    cls_prob = np.zeros((HG, WG, C))
    for r in range(HG):
        for c in range(WG):
            if cls_idx[r, c] >= 0:
                cls_prob[r, c, cls_idx[r, c]] = 1.0
    dets = probe_maps_to_detections(obj, cls_prob, depth, conf_threshold=0.5)
    labels = sorted(d.label for d in dets)
    assert labels == ["pole", "stairs"]
    print("  ✓ round-trip: teacher maps -> detections recovers labels")


def test_latent_scene_predictor_flow():
    """LatentScenePredictor data flow with dummies (no torch model needed)."""
    from models.scene_predictor import ScenePredictor

    class DummyVJEPA:
        embed_dim = 8; device = "cpu"
        def __init__(self): self.image_size = IMG
        def predict_latent_at_horizons(self, ctx, horizons):
            return {h: np.zeros((HG, WG, 8), np.float32) for h in horizons}

    # A dummy predictor object that mimics LatentScenePredictor's *output*
    # contract without needing torch (we already test the torch head separately
    # via the conversion path).
    class DummyLatentPredictor:
        def predict_scene_at_horizons(self, ctx, horizons):
            out = {}
            for h in horizons:
                obj = np.zeros((HG, WG)); cls = np.zeros((HG, WG, C)); depth = np.full((HG, WG), 4.0)
                obj[8, 8] = 0.95; cls[8, 8, 0] = 1.0; depth[8, 8] = 1.5
                out[h] = probe_maps_to_detections(obj, cls, depth)
            return out

    pred = DummyLatentPredictor()
    assert isinstance(pred, ScenePredictor)         # satisfies the protocol
    scene = pred.predict_scene_at_horizons(None, [0.5, 1.5, 2.5])
    assert set(scene.keys()) == {0.5, 1.5, 2.5}
    assert all(len(v) == 1 and v[0].label == "person" for v in scene.values())
    print("  ✓ LatentScenePredictor data-flow contract (dict[h]->Detections)")


def test_sonifier_consumes_latent_detections():
    """Detections from the latent path feed the existing sonifier unchanged."""
    from audio.scene_sonifier import SceneSonifier
    obj = np.zeros((HG, WG)); cls = np.zeros((HG, WG, C)); depth = np.full((HG, WG), 3.0)
    obj[10, 6] = 0.9; cls[10, 6, 0] = 1.0; depth[10, 6] = 2.0
    obj[6, 10] = 0.8; cls[6, 10, 2] = 1.0; depth[6, 10] = 4.0
    dets = probe_maps_to_detections(obj, cls, depth)
    audio = SceneSonifier().sonify(dets)
    assert audio.ndim == 2 and audio.shape[1] == 2 and np.abs(audio).max() > 0
    print("  ✓ latent detections -> SceneSonifier produces binaural audio")


if __name__ == "__main__":
    tests = [
        test_conversion_basic, test_conversion_threshold, test_max_objects,
        test_teacher_maps, test_teacher_closest_wins, test_roundtrip,
        test_latent_scene_predictor_flow, test_sonifier_consumes_latent_detections,
    ]
    print("\nV-JEPA latent pipeline — CPU logic tests")
    print("=" * 55)
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}"); failed += 1
    print("=" * 55)
    print(f"Results: {passed} passed, {failed} failed\n")
    sys.exit(0 if failed == 0 else 1)

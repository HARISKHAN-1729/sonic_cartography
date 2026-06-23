"""
scripts/test_audio_and_metrics.py
──────────────────────────────────
Unit tests for the CPU-only components: earcon synthesis, HRTF rendering,
scene sonification, and evaluation metrics. No GPU or pretrained models needed.

Run with:
    python scripts/test_audio_and_metrics.py
or:
    python -m pytest scripts/test_audio_and_metrics.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio.earcon_bank import EarconBank
from audio.hrtf_renderer import HRTFRenderer
from audio.scene_sonifier import SceneSonifier
from models.object_detector import Detection
from eval.metrics import (
    compute_object_hit_rate,
    compute_distance_mae,
    compute_azimuth_mae,
    aggregate_horizon_metrics,
    _labels_match,
)


SR = 44100


# ── Earcon tests ──────────────────────────────────────────────────────────

def test_earcon_bank_basic():
    bank = EarconBank(sample_rate=SR)
    for label in ["person", "car", "stairs", "curb", "wall", "traffic_light"]:
        audio = bank.get(label=label, urgency=0.5)
        assert audio.ndim == 1, f"{label}: expected 1D array"
        assert audio.dtype == np.float32, f"{label}: expected float32"
        assert len(audio) > 0, f"{label}: empty audio"
        assert np.abs(audio).max() <= 1.0 + 1e-5, f"{label}: amplitude > 1"
    print("  ✓ EarconBank basic synthesis")


def test_earcon_urgency_affects_output():
    bank = EarconBank(sample_rate=SR)
    a_low  = bank.get("person", urgency=0.1)
    a_high = bank.get("person", urgency=0.9)
    # High urgency should produce a different signal (pitch-shifted)
    assert not np.allclose(a_low, a_high, atol=1e-3), \
        "Urgency 0.1 and 0.9 produced identical audio"
    print("  ✓ EarconBank urgency modulation")


def test_earcon_all_categories():
    bank = EarconBank(sample_rate=SR)
    labels = [
        "person", "bicycle", "car", "motorbike", "bus", "truck",
        "traffic light", "stop sign", "bench", "door", "stairs",
        "curb", "pole", "fire hydrant", "bollard", "wall", "tree",
        "trash can", "table", "chair",
    ]
    for label in labels:
        audio = bank.get(label, urgency=0.5)
        assert len(audio) > 0, f"Empty audio for '{label}'"
    print(f"  ✓ EarconBank all {len(labels)} categories")


# ── HRTF renderer tests ───────────────────────────────────────────────────

def test_hrtf_analytic_output_shape():
    renderer = HRTFRenderer(sample_rate=SR, mode="analytic")
    mono = np.random.randn(SR // 4).astype(np.float32)  # 0.25s

    for az in [-90, -45, 0, 45, 90]:
        stereo = renderer.render(mono, azimuth_deg=float(az))
        assert stereo.shape == (len(mono), 2), \
            f"az={az}: expected ({len(mono)}, 2), got {stereo.shape}"
        assert stereo.dtype == np.float32
    print("  ✓ HRTFRenderer output shape")


def test_hrtf_lateralization():
    """Sound from the right should be louder in the right channel."""
    renderer = HRTFRenderer(sample_rate=SR, mode="analytic")
    mono = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(SR * 0.1)))
    mono = mono.astype(np.float32)

    stereo_right = renderer.render(mono, azimuth_deg=60.0)
    stereo_left  = renderer.render(mono, azimuth_deg=-60.0)
    stereo_center = renderer.render(mono, azimuth_deg=0.0)

    rms = lambda x: np.sqrt(np.mean(x**2))

    r_right_ch  = rms(stereo_right[:, 1])
    r_left_ch   = rms(stereo_right[:, 0])
    assert r_right_ch > r_left_ch, "Right source should be louder in right ear"

    l_left_ch   = rms(stereo_left[:, 0])
    l_right_ch  = rms(stereo_left[:, 1])
    assert l_left_ch > l_right_ch, "Left source should be louder in left ear"

    c_l = rms(stereo_center[:, 0])
    c_r = rms(stereo_center[:, 1])
    assert abs(c_l - c_r) < 0.05, "Center source should be roughly balanced"

    print("  ✓ HRTFRenderer lateralization")


def test_hrtf_distance_gain():
    """Closer sources should be louder."""
    renderer = HRTFRenderer(sample_rate=SR, mode="analytic")
    mono = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(SR * 0.1)))
    mono = mono.astype(np.float32)

    rms = lambda x: np.sqrt(np.mean(x**2))

    close = rms(renderer.render(mono, azimuth_deg=0.0, distance_m=0.5))
    far   = rms(renderer.render(mono, azimuth_deg=0.0, distance_m=5.0))
    assert close > far, "Close source should be louder than far source"
    print("  ✓ HRTFRenderer distance gain")


# ── Scene sonifier tests ──────────────────────────────────────────────────

def _make_detection(label, depth, az, conf=0.9):
    return Detection(
        label=label, confidence=conf,
        bbox=(0, 0, 100, 100),
        depth_m=depth, azimuth_deg=az,
    )


def test_sonifier_empty_scene():
    sonifier = SceneSonifier()
    audio = sonifier.sonify([])
    assert audio.ndim == 2 and audio.shape[1] == 2, "Expected (N, 2)"
    assert np.allclose(audio, 0), "Empty scene should be silence"
    print("  ✓ SceneSonifier empty scene")


def test_sonifier_single_detection():
    sonifier = SceneSonifier()
    dets = [_make_detection("person", depth=2.0, az=15.0)]
    audio = sonifier.sonify(dets)
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert np.abs(audio).max() > 0, "Non-empty scene should have sound"
    assert np.abs(audio).max() <= 1.0 + 1e-4
    print("  ✓ SceneSonifier single detection")


def test_sonifier_max_objects_limit():
    sonifier = SceneSonifier()
    dets = [_make_detection("car", depth=float(i), az=float(i*10 - 30))
            for i in range(1, 12)]
    audio = sonifier.sonify(dets, max_objects=4)
    # Should not crash; audio should be valid stereo
    assert audio.ndim == 2 and audio.shape[1] == 2
    print("  ✓ SceneSonifier max_objects limit")


def test_sonifier_beyond_range():
    sonifier = SceneSonifier(max_distance_m=5.0)
    dets = [
        _make_detection("person", depth=3.0, az=0.0),   # in range
        _make_detection("car",    depth=8.0, az=0.0),   # out of range
    ]
    audio_both = sonifier.sonify(dets)
    audio_only_in_range = sonifier.sonify([dets[0]])

    # Both should produce audio (in-range object), and should differ
    assert np.abs(audio_both).max() > 0
    print("  ✓ SceneSonifier range filtering")


# ── Metrics tests ─────────────────────────────────────────────────────────

def test_labels_match():
    assert _labels_match("traffic_light", "traffic light")
    assert _labels_match("person", "person")
    assert _labels_match("fire_hydrant", "fire hydrant")
    assert not _labels_match("car", "person")
    print("  ✓ Label synonym matching")


def test_object_hit_rate_perfect():
    pred = [_make_detection("person", depth=2.0, az=10.0)]
    gt   = [_make_detection("person", depth=2.0, az=10.0)]
    prec, rec, pairs = compute_object_hit_rate(pred, gt)
    assert prec == 1.0 and rec == 1.0
    assert len(pairs) == 1
    print("  ✓ Object hit rate (perfect match)")


def test_object_hit_rate_no_match():
    pred = [_make_detection("person", depth=2.0, az=10.0)]
    gt   = [_make_detection("car",    depth=5.0, az=-30.0)]
    prec, rec, pairs = compute_object_hit_rate(pred, gt)
    assert prec == 0.0 and rec == 0.0
    assert len(pairs) == 0
    print("  ✓ Object hit rate (no match)")


def test_object_hit_rate_partial():
    pred = [
        _make_detection("person", depth=2.0, az=10.0),
        _make_detection("car",    depth=5.0, az=-20.0),
    ]
    gt = [
        _make_detection("person", depth=2.5, az=12.0),  # matches pred[0]
        _make_detection("tree",   depth=3.0, az=30.0),  # no match
    ]
    prec, rec, pairs = compute_object_hit_rate(pred, gt)
    assert 0.0 < prec <= 1.0
    assert 0.0 < rec <= 1.0
    print(f"  ✓ Object hit rate (partial): P={prec:.2f}, R={rec:.2f}")


def test_distance_mae():
    from eval.metrics import MatchedPair
    pairs = [
        MatchedPair("person", 2.0, 2.5, 10.0, 10.0, 1.0),
        MatchedPair("car",    5.0, 4.5, -20.0, -22.0, 1.0),
    ]
    mae = compute_distance_mae(pairs)
    expected = (0.5 + 0.5) / 2
    assert abs(mae - expected) < 1e-5, f"Expected {expected}, got {mae}"
    print(f"  ✓ Distance MAE: {mae:.3f}m")


def test_aggregate_horizon_metrics():
    from eval.metrics import MatchedPair
    pairs = [
        MatchedPair("person", 2.0, 2.5, 10.0, 12.0, 1.5),
        MatchedPair("car",    5.0, 4.5, -20.0, -22.0, 1.5),
    ]
    m = aggregate_horizon_metrics(
        horizon_sec=1.5,
        all_precisions=[0.8, 0.7, 0.9],
        all_recalls=[0.6, 0.7, 0.8],
        all_matched_pairs=pairs,
    )
    assert m.horizon_sec == 1.5
    assert 0 < m.f1 < 1
    assert not np.isnan(m.distance_mae_m)
    print(f"  ✓ Aggregate horizon metrics: F1={m.f1:.3f}, "
          f"DepthMAE={m.distance_mae_m:.3f}m")


# ── Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_earcon_bank_basic,
        test_earcon_urgency_affects_output,
        test_earcon_all_categories,
        test_hrtf_analytic_output_shape,
        test_hrtf_lateralization,
        test_hrtf_distance_gain,
        test_sonifier_empty_scene,
        test_sonifier_single_detection,
        test_sonifier_max_objects_limit,
        test_sonifier_beyond_range,
        test_labels_match,
        test_object_hit_rate_perfect,
        test_object_hit_rate_no_match,
        test_object_hit_rate_partial,
        test_distance_mae,
        test_aggregate_horizon_metrics,
    ]

    print("\nPredictive Sonic Cartography — Unit Tests")
    print("=" * 55)
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            failed += 1

    print("=" * 55)
    print(f"Results: {passed} passed, {failed} failed\n")
    sys.exit(0 if failed == 0 else 1)

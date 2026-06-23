"""
audio/earcon_bank.py
─────────────────────
Earcon (audio icon) synthesizer for navigation-relevant object classes.

Each object class maps to a distinct synthesized sound with:
  - A characteristic frequency / timbre
  - An urgency-modulated envelope (faster attack = more urgent = closer)
  - No language dependency (non-speech, universally interpretable)

Design principles:
  - Low frequencies → large / structural objects (walls, vehicles)
  - High frequencies → small / dynamic objects (people, traffic lights)
  - Periodic pulse → static objects (bench, pole)
  - Modulated / AM → dynamic objects (person, bicycle)
  - Simple sine / triangle → landmark objects (stairs, door)

All earcons are synthesized in real-time; no audio files needed.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION_MS = 120
DEFAULT_FADE_MS = 20


class EarconBank:
    """
    Synthesizes earcons (non-speech audio icons) for detected object classes.

    Usage
    -----
    bank = EarconBank(sample_rate=44100)
    audio = bank.get(label="person", urgency=0.8, duration_ms=120)
    # audio: np.ndarray (N,) float32 in [-1, 1]
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        freq_map: Optional[dict] = None,
        duration_ms: int = DEFAULT_DURATION_MS,
        fade_ms: int = DEFAULT_FADE_MS,
    ):
        self.sr = sample_rate
        self.duration_ms = duration_ms
        self.fade_ms = fade_ms
        self.freq_map = freq_map or DEFAULT_FREQ_MAP

    def get(
        self,
        label: str,
        urgency: float = 0.5,
        duration_ms: Optional[int] = None,
    ) -> np.ndarray:
        """
        Synthesize an earcon for the given object label.

        Parameters
        ----------
        label      : object class (e.g. "person", "stairs")
        urgency    : float [0, 1] — 0 = far/low urgency, 1 = close/high
                     Controls attack time and modulation rate.
        duration_ms: override default earcon duration

        Returns
        -------
        np.ndarray (N,) float32 in [-1, 1]
        """
        dur_ms = duration_ms or self.duration_ms
        freq = self.freq_map.get(label, 440)  # default: A4

        # Choose synthesis strategy per object category
        category = _get_category(label)

        if category == "vehicle":
            audio = self._synth_vehicle(freq, urgency, dur_ms)
        elif category == "person":
            audio = self._synth_person(freq, urgency, dur_ms)
        elif category == "landmark":
            audio = self._synth_landmark(freq, urgency, dur_ms)
        elif category == "hazard":
            audio = self._synth_hazard(freq, urgency, dur_ms)
        elif category == "structural":
            audio = self._synth_structural(freq, urgency, dur_ms)
        else:
            audio = self._synth_generic(freq, urgency, dur_ms)

        # Apply fade-in / fade-out envelope
        audio = self._apply_fade(audio, self.fade_ms)

        # Normalise
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio /= peak

        return audio.astype(np.float32)

    # ── Synthesis methods ──────────────────────────────────────────────────

    def _synth_vehicle(
        self, freq: float, urgency: float, dur_ms: int
    ) -> np.ndarray:
        """Low rumble + AM modulation. Higher urgency → faster modulation."""
        t = self._time_array(dur_ms)
        mod_rate = 4.0 + urgency * 12.0  # 4–16 Hz AM
        carrier = np.sin(2 * np.pi * freq * t)
        modulator = 0.5 + 0.5 * np.sin(2 * np.pi * mod_rate * t)
        # Add a low harmonic for richer texture
        harmonic = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        return ((carrier + harmonic) * modulator).astype(np.float32)

    def _synth_person(
        self, freq: float, urgency: float, dur_ms: int
    ) -> np.ndarray:
        """
        AM-modulated tone with slight inharmonicity — feels 'alive' / dynamic.
        Higher urgency → higher pitch shift (Doppler-like).
        """
        t = self._time_array(dur_ms)
        pitch_shift = 1.0 + urgency * 0.15  # up to +15% pitch
        f = freq * pitch_shift
        mod_rate = 8.0 + urgency * 8.0  # 8–16 Hz
        carrier = np.sin(2 * np.pi * f * t)
        mod = 0.4 + 0.6 * np.sin(2 * np.pi * mod_rate * t)
        noise_component = 0.05 * np.random.randn(len(t))
        return ((carrier * mod) + noise_component).astype(np.float32)

    def _synth_landmark(
        self, freq: float, urgency: float, dur_ms: int
    ) -> np.ndarray:
        """
        Clean sine tone — landmarks are structural, predictable.
        Stairs: two-tone ascending. Door: single ping.
        """
        t = self._time_array(dur_ms)
        # Two-note motif: base + minor third
        f2 = freq * (2 ** (3 / 12))  # minor third
        split = len(t) // 2
        audio = np.zeros(len(t), dtype=np.float32)
        audio[:split] = np.sin(2 * np.pi * freq * t[:split])
        audio[split:] = np.sin(2 * np.pi * f2 * t[split:])
        return audio

    def _synth_hazard(
        self, freq: float, urgency: float, dur_ms: int
    ) -> np.ndarray:
        """
        Rapid pulse for ground-level hazards (curb, step).
        Higher urgency → more pulses.
        """
        t = self._time_array(dur_ms)
        n_pulses = 2 + int(urgency * 3)  # 2–5 pulses
        pulse_freq = n_pulses / (dur_ms / 1000.0)
        envelope = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * pulse_freq * t))
        carrier = np.sin(2 * np.pi * freq * t)
        return (carrier * envelope).astype(np.float32)

    def _synth_structural(
        self, freq: float, urgency: float, dur_ms: int
    ) -> np.ndarray:
        """
        Sawtooth-like (sum of harmonics) for structural objects.
        Lower urgency bias: structural objects rarely urgent.
        """
        t = self._time_array(dur_ms)
        audio = np.zeros(len(t), dtype=np.float32)
        # Sawtooth approximation: sum of harmonics
        for k in range(1, 5):
            audio += (1.0 / k) * np.sin(2 * np.pi * freq * k * t)
        return audio

    def _synth_generic(
        self, freq: float, urgency: float, dur_ms: int
    ) -> np.ndarray:
        """Fallback: simple sine."""
        t = self._time_array(dur_ms)
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _time_array(self, dur_ms: int) -> np.ndarray:
        n_samples = int(self.sr * dur_ms / 1000)
        return np.linspace(0, dur_ms / 1000.0, n_samples, endpoint=False)

    def _apply_fade(self, audio: np.ndarray, fade_ms: int) -> np.ndarray:
        fade_samples = min(int(self.sr * fade_ms / 1000), len(audio) // 4)
        if fade_samples == 0:
            return audio
        fade_in  = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        audio = audio.copy()
        audio[:fade_samples]  *= fade_in
        audio[-fade_samples:] *= fade_out
        return audio


# ── Category classification ────────────────────────────────────────────────

VEHICLE_LABELS   = {"car", "bus", "truck", "motorbike", "bicycle"}
PERSON_LABELS    = {"person"}
LANDMARK_LABELS  = {"stairs", "door", "traffic_light", "stop_sign"}
HAZARD_LABELS    = {"curb", "bollard", "fire_hydrant", "pole"}
STRUCTURAL_LABELS = {"wall", "tree", "bench", "table", "chair", "trash_can"}


def _get_category(label: str) -> str:
    label = label.replace(" ", "_")
    if label in VEHICLE_LABELS:    return "vehicle"
    if label in PERSON_LABELS:     return "person"
    if label in LANDMARK_LABELS:   return "landmark"
    if label in HAZARD_LABELS:     return "hazard"
    if label in STRUCTURAL_LABELS: return "structural"
    return "generic"


# Default frequency map (Hz) — must match configs/default.yaml
DEFAULT_FREQ_MAP = {
    "person":        440,
    "bicycle":       550,
    "car":           220,
    "motorbike":     330,
    "bus":           110,
    "truck":         130,
    "traffic_light": 880,
    "traffic light": 880,
    "stop_sign":     660,
    "stop sign":     660,
    "bench":         392,
    "door":          494,
    "stairs":        523,
    "curb":          415,
    "pole":          370,
    "fire_hydrant":  349,
    "fire hydrant":  349,
    "bollard":       330,
    "wall":          196,
    "tree":          262,
    "trash_can":     311,
    "trash can":     311,
    "table":         294,
    "chair":         277,
}

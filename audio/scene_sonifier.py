"""
audio/scene_sonifier.py
────────────────────────
Converts a parsed scene (list of Detections) into a binaural audio scene.

The output is a stereo wav array where each detected object is rendered
as a spatialized earcon at its azimuth/elevation/distance, with urgency
proportional to proximity.

Objects are layered — closer objects are more prominent. Objects beyond
max_distance_m are silent (below hearing threshold assumption).
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
import soundfile as sf
from pathlib import Path

from audio.earcon_bank import EarconBank
from audio.hrtf_renderer import HRTFRenderer
from models.object_detector import Detection


class SceneSonifier:
    """
    Converts a list of Detections into a single binaural stereo audio array.

    Usage
    -----
    sonifier = SceneSonifier.from_config(cfg.audio)
    stereo = sonifier.sonify(detections)
    # stereo: np.ndarray (N, 2) float32
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        earcon_bank: Optional[EarconBank] = None,
        hrtf_renderer: Optional[HRTFRenderer] = None,
        max_distance_m: float = 8.0,
        max_gain_db: float = 0.0,
        min_gain_db: float = -30.0,
        earcon_duration_ms: int = 120,
        pulse_interval_ms: int = 400,
    ):
        self.sr = sample_rate
        self.earcon_bank = earcon_bank or EarconBank(sample_rate=sample_rate)
        self.hrtf = hrtf_renderer or HRTFRenderer(
            sample_rate=sample_rate, mode="analytic"
        )
        self.max_distance_m = max_distance_m
        self.max_gain_db = max_gain_db
        self.min_gain_db = min_gain_db
        self.earcon_duration_ms = earcon_duration_ms
        self.pulse_interval_ms = pulse_interval_ms

    @classmethod
    def from_config(cls, cfg) -> "SceneSonifier":
        sr = cfg.get("sample_rate", 44100)
        freq_map = dict(cfg.get("earcon_frequencies", {}))

        bank = EarconBank(
            sample_rate=sr,
            freq_map=freq_map if freq_map else None,
            duration_ms=cfg.get("earcon_duration_ms", 120),
            fade_ms=cfg.get("fade_ms", 20),
        )

        hrtf = HRTFRenderer(
            sample_rate=sr,
            mode="analytic",
        )

        return cls(
            sample_rate=sr,
            earcon_bank=bank,
            hrtf_renderer=hrtf,
            max_distance_m=cfg.get("max_distance_m", 8.0),
            max_gain_db=cfg.get("max_gain_db_db", 0.0),
            min_gain_db=cfg.get("min_gain_db", -30.0),
            earcon_duration_ms=cfg.get("earcon_duration_ms", 120),
            pulse_interval_ms=cfg.get("pulse_interval_ms", 400),
        )

    def sonify(
        self,
        detections: List[Detection],
        max_objects: int = 6,
    ) -> np.ndarray:
        """
        Render a list of detections into a binaural audio array.

        Objects are sorted by proximity (closest first) and capped at
        max_objects to keep the soundscape interpretable.

        Parameters
        ----------
        detections  : list of Detection (from object_detector.py)
        max_objects : max simultaneous earcons (cognitive load limit)

        Returns
        -------
        np.ndarray (N, 2) float32 — stereo binaural at self.sr
        """
        # Filter out objects beyond audible range
        in_range = [
            d for d in detections
            if d.depth_m <= self.max_distance_m
        ]

        # Sort by proximity (closest = most urgent)
        in_range.sort(key=lambda d: d.depth_m)

        # Limit to max_objects to avoid cognitive overload
        in_range = in_range[:max_objects]

        if not in_range:
            # Return short silence
            n_samples = int(self.sr * self.earcon_duration_ms / 1000)
            return np.zeros((n_samples, 2), dtype=np.float32)

        # Determine output length: pulse interval * n_objects or earcon dur
        # Strategy: objects are rendered at staggered time offsets so the
        # soundscape doesn't overwhelm at once. Closest object is first.
        interval_samples = int(self.sr * self.pulse_interval_ms / 1000)
        earcon_samples   = int(self.sr * self.earcon_duration_ms / 1000)
        total_samples    = (
            (len(in_range) - 1) * interval_samples + earcon_samples
        )
        output = np.zeros((total_samples, 2), dtype=np.float32)

        for i, det in enumerate(in_range):
            start = i * interval_samples
            end   = start + earcon_samples

            # Compute urgency [0, 1] — 0 = far, 1 = touching
            urgency = self._distance_to_urgency(det.depth_m)

            # Synthesize earcon (mono)
            mono_earcon = self.earcon_bank.get(
                label=det.label,
                urgency=urgency,
                duration_ms=self.earcon_duration_ms,
            )

            # Compute gain from distance
            gain = self._distance_to_gain(det.depth_m)
            mono_earcon *= gain

            # Spatialize with HRTF
            binaural = self.hrtf.render(
                mono_signal=mono_earcon,
                azimuth_deg=det.azimuth_deg,
                elevation_deg=det.elevation_deg,
                distance_m=det.depth_m,
            )  # (earcon_samples, 2)

            # Mix into output buffer
            actual_len = min(binaural.shape[0], total_samples - start)
            output[start : start + actual_len] += binaural[:actual_len]

        # Final limiter: hard clip to [-1, 1]
        output = np.clip(output, -1.0, 1.0)
        return output.astype(np.float32)

    def sonify_sequence(
        self,
        detection_sequence: List[List[Detection]],
        horizon_times_sec: List[float],
    ) -> dict[float, np.ndarray]:
        """
        Sonify predictions at multiple time horizons.

        Parameters
        ----------
        detection_sequence  : list of detection lists, one per horizon
        horizon_times_sec   : matching list of horizon times

        Returns
        -------
        dict mapping horizon_sec → stereo np.ndarray
        """
        result = {}
        for horizon, detections in zip(horizon_times_sec, detection_sequence):
            result[horizon] = self.sonify(detections)
        return result

    def save(
        self,
        audio: np.ndarray,
        path: str | Path,
    ) -> None:
        """Write binaural stereo to a wav file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, self.sr, subtype="PCM_16")

    # ── Private helpers ────────────────────────────────────────────────────

    def _distance_to_urgency(self, dist_m: float) -> float:
        """Map distance to urgency [0, 1]. 0m=1.0, max_dist=0.0."""
        return float(
            np.clip(1.0 - (dist_m / self.max_distance_m), 0.0, 1.0)
        )

    def _distance_to_gain(self, dist_m: float) -> float:
        """
        Map distance to linear gain.
        Interpolates between max_gain_db (close) and min_gain_db (far).
        """
        dist_m = max(0.1, dist_m)
        t = np.clip(dist_m / self.max_distance_m, 0.0, 1.0)
        gain_db = self.max_gain_db + t * (self.min_gain_db - self.max_gain_db)
        return float(10 ** (gain_db / 20.0))

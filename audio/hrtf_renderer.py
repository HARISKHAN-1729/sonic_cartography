"""
audio/hrtf_renderer.py
───────────────────────
Binaural HRTF renderer for spatialized audio.

Uses a minimal analytic HRTF model (Woodworth-Schlosberg ITD + ILD)
that requires no external data files — good for prototyping.

For higher fidelity, drop in the MIT KEMAR HRTF dataset:
  http://sound.media.mit.edu/resources/KEMAR.html
  (set KEMAR_SOFA_PATH env var to the .sofa file)

The pyfar/python-sofa libraries can load SOFA format HRTFs; this file
includes a loader for both paths.
"""

from __future__ import annotations
import numpy as np
from scipy import signal
from typing import Optional
import warnings


# ── Constants ─────────────────────────────────────────────────────────────
HEAD_RADIUS_M = 0.0875        # average human head radius
SPEED_OF_SOUND_M_S = 343.0
DEFAULT_SAMPLE_RATE = 44100


class HRTFRenderer:
    """
    Renders a mono audio signal into a binaural stereo pair
    given azimuth and elevation angles.

    Supports two modes:
      'analytic' — fast Woodworth ITD + frequency-shaped ILD (no data needed)
      'sofa'     — loads a real HRTF dataset (.sofa format)

    Parameters
    ----------
    sample_rate : int
    mode        : 'analytic' | 'sofa'
    sofa_path   : path to .sofa file (required if mode='sofa')
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mode: str = "analytic",
        sofa_path: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.mode = mode
        self._sofa = None

        if mode == "sofa":
            if sofa_path is None:
                import os
                sofa_path = os.environ.get("KEMAR_SOFA_PATH")
            if sofa_path:
                self._load_sofa(sofa_path)
            else:
                warnings.warn(
                    "SOFA path not provided; falling back to analytic HRTF.",
                    UserWarning,
                )
                self.mode = "analytic"

    def _load_sofa(self, path: str):
        """Load SOFA HRTF file (requires python-sofa or pysofaconventions)."""
        try:
            import sofa  # type: ignore
            self._sofa = sofa.Database.open(path)
        except ImportError:
            warnings.warn(
                "python-sofa not installed (pip install python-sofa). "
                "Falling back to analytic HRTF.",
                UserWarning,
            )
            self.mode = "analytic"

    def render(
        self,
        mono_signal: np.ndarray,
        azimuth_deg: float,
        elevation_deg: float = 0.0,
        distance_m: float = 1.0,
    ) -> np.ndarray:
        """
        Spatialize a mono signal to binaural stereo.

        Parameters
        ----------
        mono_signal   : np.ndarray (N,) float32 in [-1, 1]
        azimuth_deg   : horizontal angle, +right / -left, range [-180, 180]
        elevation_deg : vertical angle, +up / -down, range [-90, 90]
        distance_m    : source distance in metres (for gain scaling)

        Returns
        -------
        np.ndarray (N, 2) float32 — [left, right] stereo
        """
        if self.mode == "sofa" and self._sofa is not None:
            return self._render_sofa(
                mono_signal, azimuth_deg, elevation_deg, distance_m
            )
        return self._render_analytic(
            mono_signal, azimuth_deg, elevation_deg, distance_m
        )

    # ── Analytic HRTF ─────────────────────────────────────────────────────

    def _render_analytic(
        self,
        mono: np.ndarray,
        az: float,
        el: float,
        dist: float,
    ) -> np.ndarray:
        """
        Analytic binaural rendering.

        ITD: Woodworth & Schlosberg spherical head model
        ILD: Frequency-dependent interaural level difference
             approximated as a first-order shelving filter
        """
        az_rad = np.radians(az)

        # ── ITD (Interaural Time Difference) ────────────────────────────
        # Woodworth model: ITD = (R/c) * (sin(θ) + θ) for |θ| ≤ π/2
        # Simplified: ITD = (R/c) * sin(θ) for small angles
        itd_s = (HEAD_RADIUS_M / SPEED_OF_SOUND_M_S) * np.sin(az_rad)
        itd_samples = int(abs(itd_s) * self.sample_rate)
        itd_samples = min(itd_samples, len(mono) - 1)

        n = len(mono)
        left = np.zeros(n, dtype=np.float32)
        right = np.zeros(n, dtype=np.float32)

        if az >= 0:  # source on the right: right ear leads
            right[:] = mono
            if itd_samples > 0:
                left[itd_samples:] = mono[:-itd_samples]
            else:
                left[:] = mono
        else:        # source on the left: left ear leads
            left[:] = mono
            if itd_samples > 0:
                right[itd_samples:] = mono[:-itd_samples]
            else:
                right[:] = mono

        # ── ILD (Interaural Level Difference) ──────────────────────────
        # Ipsilateral ear louder; contralateral ear attenuated & low-pass
        ild_db = 6.0 * abs(np.sin(az_rad))  # up to ~6 dB
        ild_linear = 10 ** (ild_db / 20.0)

        if az >= 0:  # right louder
            right *= ild_linear
            left = self._apply_shadow_filter(left, cutoff_hz=1200.0)
        else:        # left louder
            left *= ild_linear
            right = self._apply_shadow_filter(right, cutoff_hz=1200.0)

        # ── Elevation cue: HRTF pinna notch (simplified) ───────────────
        # Elevations above horizon add a spectral notch around 8 kHz
        # (pinna diffraction effect)
        if abs(el) > 20:
            notch_gain = -4.0 * (abs(el) / 90.0)
            left  = self._apply_notch(left,  freq=8000.0, gain_db=notch_gain)
            right = self._apply_notch(right, freq=8000.0, gain_db=notch_gain)

        # ── Distance gain ───────────────────────────────────────────────
        # 1/r distance law, clamped
        dist = max(0.5, dist)
        dist_gain = np.clip(1.0 / dist, 0.01, 2.0)
        left  *= dist_gain
        right *= dist_gain

        stereo = np.stack([left, right], axis=-1)  # (N, 2)
        return stereo.astype(np.float32)

    def _apply_shadow_filter(
        self, signal_: np.ndarray, cutoff_hz: float = 1200.0
    ) -> np.ndarray:
        """Low-pass Butterworth filter simulating head shadow on far ear."""
        nyq = self.sample_rate / 2.0
        cutoff_norm = min(cutoff_hz / nyq, 0.99)
        b, a = signal.butter(2, cutoff_norm, btype="low")
        return signal.lfilter(b, a, signal_).astype(np.float32)

    def _apply_notch(
        self,
        sig: np.ndarray,
        freq: float = 8000.0,
        gain_db: float = -4.0,
        q: float = 8.0,
    ) -> np.ndarray:
        """Peaking EQ notch (pinna elevation cue)."""
        nyq = self.sample_rate / 2.0
        freq_norm = min(freq / nyq, 0.99)
        b, a = signal.iirpeak(freq_norm, q)
        # Scale b to achieve approximate gain_db at peak
        gain_linear = 10 ** (gain_db / 40.0)
        b = b * gain_linear
        return signal.lfilter(b, a, sig).astype(np.float32)

    # ── SOFA-based rendering ───────────────────────────────────────────────

    def _render_sofa(
        self,
        mono: np.ndarray,
        az: float,
        el: float,
        dist: float,
    ) -> np.ndarray:
        """Convolve with nearest SOFA HRTF measurement."""
        # Find nearest measurement position in SOFA dataset
        positions = self._sofa.Source.Position.get_values(
            system="spherical"
        )  # (N, 3): [az, el, dist]

        az_arr = positions[:, 0]
        el_arr = positions[:, 1]

        # Nearest-neighbour lookup (L2 in angle space)
        diffs = (az_arr - az) ** 2 + (el_arr - el) ** 2
        idx = int(np.argmin(diffs))

        # Get left/right HRIRs
        hrir_l = self._sofa.Data.IR.get_values()[idx, 0, :]  # left ear
        hrir_r = self._sofa.Data.IR.get_values()[idx, 1, :]  # right ear

        left  = signal.fftconvolve(mono, hrir_l, mode="full")[: len(mono)]
        right = signal.fftconvolve(mono, hrir_r, mode="full")[: len(mono)]

        # Distance gain
        dist = max(0.5, dist)
        gain = np.clip(1.0 / dist, 0.01, 2.0)
        left  = (left  * gain).astype(np.float32)
        right = (right * gain).astype(np.float32)

        return np.stack([left, right], axis=-1)

"""
models/world_model.py
──────────────────────
Swappable "world model" abstraction for predicting future egocentric frames.

WHY THIS EXISTS
───────────────
The Navigation World Model (NWM, Bar et al. CVPR 2025) is powerful but
comes with three practical problems for *this* project:

  1. Its HuggingFace repo (facebook/nwm) ships only GATED .pth.tar
     state-dicts — there is no `from_pretrained()` / `.rollout()` API.
     You must build the CDiT model from the repo's models.py and load
     weights manually (see models/nwm_wrapper.py for the faithful path).

  2. It is a diffusion model conditioned on CONTINUOUS robot pose-deltas
     (Δx, Δy, Δyaw), trained on ground-robot datasets (RECON, SCAND,
     TartanDrive, HuRoN). Pedestrian SANPO frames have no matching action
     labels, so "walk forward" conditioning is only approximate.

  3. Each prediction needs a multi-step diffusion sampling loop + VAE
     decode — heavy, and a hard dependency on the (CC-BY-NC) repo.

This module decouples the rest of the PSC pipeline from NWM by defining a
single interface, `WorldModel.predict_at_horizons(...)`, with three
interchangeable backends:

  • "nwm"        — faithful NWM CDiT integration (best quality, needs GPU +
                   gated checkpoint + the cloned repo). See nwm_wrapper.py.
  • "flow"       — dense optical-flow constant-velocity scene extrapolation.
                   Runs on CPU, no checkpoint, works on ANY egocentric video
                   including pedestrian SANPO. Great default + paper baseline.
  • "last_frame" — naive "predict = most recent frame". The trivial lower
                   bound for the ablation.

For the IROS abstract this is a feature, not a workaround: reporting
F1-vs-lead-time for {nwm, flow, last_frame} is a clean ablation that shows
how much the world model actually buys you over cheap baselines.

All backends share the same I/O contract as the original NWMWrapper:
    predict(context_frames, action, n_frames)        -> list[np.ndarray]
    predict_at_horizons(context_frames, horizons_sec) -> dict[float, np.ndarray]
so they are drop-in replacements anywhere NWMWrapper was used.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Protocol, runtime_checkable

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ─────────────────────────────────────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class WorldModel(Protocol):
    """Common interface for every future-frame predictor."""

    prediction_fps: int

    def predict(
        self,
        context_frames: List[np.ndarray],
        action: str = "walk_forward",
        n_frames: int = 12,
    ) -> List[np.ndarray]:
        ...

    def predict_at_horizons(
        self,
        context_frames: List[np.ndarray],
        horizon_times_sec: List[float],
        action: str = "walk_forward",
    ) -> Dict[float, np.ndarray]:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Shared mixin: turn a per-frame predict() into predict_at_horizons()
# ─────────────────────────────────────────────────────────────────────────────

class _HorizonMixin:
    """Provides predict_at_horizons() given a working predict() + fps."""

    prediction_fps: int

    def predict_at_horizons(
        self,
        context_frames: List[np.ndarray],
        horizon_times_sec: List[float],
        action: str = "walk_forward",
    ) -> Dict[float, np.ndarray]:
        max_frames = int(round(max(horizon_times_sec) * self.prediction_fps)) + 1
        all_frames = self.predict(context_frames, action=action, n_frames=max_frames)
        out: Dict[float, np.ndarray] = {}
        for t in horizon_times_sec:
            idx = min(int(round(t * self.prediction_fps)), len(all_frames) - 1)
            out[t] = all_frames[idx]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1 — Optical-flow constant-velocity extrapolation  (CPU, runs today)
# ─────────────────────────────────────────────────────────────────────────────

class FlowBackend(_HorizonMixin):
    """
    Predicts future frames by estimating dense optical flow between the last
    two context frames and forward-warping the most recent frame along that
    motion field, scaled by the prediction horizon.

    Intuition: as the pedestrian keeps walking, the apparent motion of the
    scene continues at roughly constant velocity over short horizons (0.5–2.5s).
    Forward-warping the last frame by k·flow gives a physically-plausible
    "what I'm about to see" estimate — exactly the signal PSC needs to sonify.

    This is a legitimate constant-velocity *world model* baseline. It needs no
    training, no checkpoint, no GPU, and — unlike NWM — works directly on
    pedestrian egocentric video with no action labels.

    Parameters
    ----------
    prediction_fps     : frames/sec the horizon indexing assumes
    flow_amplify       : global multiplier on extrapolated motion (tune 0.8–1.5)
    expand_on_approach : if True, slightly scales up motion magnitude over the
                         horizon to emulate objects looming as you near them
    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma:
                         Farneback optical-flow parameters
    """

    def __init__(
        self,
        prediction_fps: int = 8,
        flow_amplify: float = 1.0,
        expand_on_approach: bool = True,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ):
        if not _HAS_CV2:
            raise ImportError(
                "FlowBackend needs OpenCV. Install with: pip install opencv-python"
            )
        self.prediction_fps = prediction_fps
        self.flow_amplify = flow_amplify
        self.expand_on_approach = expand_on_approach
        self._fb = dict(
            pyr_scale=pyr_scale, levels=levels, winsize=winsize,
            iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=0,
        )

    @classmethod
    def from_config(cls, cfg) -> "FlowBackend":
        return cls(
            prediction_fps=cfg.get("prediction_fps", 8),
            flow_amplify=cfg.get("flow_amplify", 1.0),
            expand_on_approach=cfg.get("expand_on_approach", True),
        )

    def predict(
        self,
        context_frames: List[np.ndarray],
        action: str = "walk_forward",
        n_frames: int = 12,
    ) -> List[np.ndarray]:
        if len(context_frames) < 2:
            # Not enough motion context: fall back to repeating last frame
            last = _as_uint8_rgb(context_frames[-1])
            return [last.copy() for _ in range(n_frames)]

        prev = _as_uint8_rgb(context_frames[-2])
        last = _as_uint8_rgb(context_frames[-1])

        g_prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        g_last = cv2.cvtColor(last, cv2.COLOR_RGB2GRAY)

        # Dense flow describing motion from prev -> last (one inter-frame step)
        flow = cv2.calcOpticalFlowFarneback(g_prev, g_last, None, **self._fb)

        h, w = flow.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

        out: List[np.ndarray] = []
        for step in range(1, n_frames + 1):
            # k = how many inter-frame motions ahead this future frame is
            k = float(step) * self.flow_amplify
            if self.expand_on_approach:
                # Mild super-linear growth to emulate looming (objects get
                # closer faster than linear as you approach them)
                k *= (1.0 + 0.04 * step)

            # Backward map for cv2.remap: sample source at (x - k·u, y - k·v)
            map_x = (xs - k * flow[..., 0]).astype(np.float32)
            map_y = (ys - k * flow[..., 1]).astype(np.float32)
            pred = cv2.remap(
                last, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            out.append(pred)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2 — Last-frame baseline (trivial lower bound for the ablation)
# ─────────────────────────────────────────────────────────────────────────────

class LastFrameBackend(_HorizonMixin):
    """Predicts every future frame as a copy of the most recent context frame."""

    def __init__(self, prediction_fps: int = 8):
        self.prediction_fps = prediction_fps

    @classmethod
    def from_config(cls, cfg) -> "LastFrameBackend":
        return cls(prediction_fps=cfg.get("prediction_fps", 8))

    def predict(
        self,
        context_frames: List[np.ndarray],
        action: str = "walk_forward",
        n_frames: int = 12,
    ) -> List[np.ndarray]:
        last = _as_uint8_rgb(context_frames[-1])
        return [last.copy() for _ in range(n_frames)]


# ─────────────────────────────────────────────────────────────────────────────
# Backend 3 — NWM (faithful). Thin adapter around models/nwm_wrapper.py
# ─────────────────────────────────────────────────────────────────────────────

class NWMBackend(_HorizonMixin):
    """
    Adapter that wraps the faithful NWMWrapper so it satisfies the WorldModel
    interface. Kept thin on purpose: all the heavy, repo-specific logic
    (VAE encode/decode, CDiT construction, diffusion sampling, continuous
    action conditioning, .pth.tar loading) lives in nwm_wrapper.py.
    """

    def __init__(self, wrapper):
        self._w = wrapper
        self.prediction_fps = getattr(wrapper, "prediction_fps", 8)

    @classmethod
    def from_config(cls, cfg) -> "NWMBackend":
        from models.nwm_wrapper import NWMWrapper  # local import: heavy deps
        return cls(NWMWrapper.from_config(cfg))

    def predict(self, context_frames, action="walk_forward", n_frames=12):
        return self._w.predict(context_frames, action=action, n_frames=n_frames)

    def predict_at_horizons(self, context_frames, horizon_times_sec,
                            action="walk_forward"):
        # NWMWrapper already implements an optimized horizon predictor
        return self._w.predict_at_horizons(
            context_frames, horizon_times_sec, action=action
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_world_model(cfg) -> WorldModel:
    """
    Construct a world model from a config node.

    Expected config shape (configs/default.yaml):

        world_model:
          backend: "flow"          # "nwm" | "flow" | "last_frame"
          prediction_fps: 8
          flow:
            flow_amplify: 1.0
            expand_on_approach: true

        model:
          nwm:                      # only read when backend == "nwm"
            ...

    The factory keeps NWM's heavy imports lazy so that "flow"/"last_frame"
    work in environments without torch / diffusers / the NWM repo.
    """
    wm_cfg = cfg.get("world_model", {}) if hasattr(cfg, "get") else {}
    backend = wm_cfg.get("backend", "flow")
    fps = wm_cfg.get("prediction_fps", 8)

    if backend == "nwm":
        nwm_cfg = cfg["model"]["nwm"]
        # Make sure the wrapper's fps matches the world_model fps
        try:
            nwm_cfg["prediction_fps"] = fps
        except Exception:
            pass
        return NWMBackend.from_config(nwm_cfg)

    if backend == "flow":
        flow_cfg = wm_cfg.get("flow", {})
        return FlowBackend(
            prediction_fps=fps,
            flow_amplify=flow_cfg.get("flow_amplify", 1.0),
            expand_on_approach=flow_cfg.get("expand_on_approach", True),
        )

    if backend == "last_frame":
        return LastFrameBackend(prediction_fps=fps)

    raise ValueError(
        f"Unknown world_model.backend '{backend}'. "
        f"Choose one of: 'nwm', 'flow', 'last_frame'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _as_uint8_rgb(frame) -> np.ndarray:
    """Coerce a frame (PIL.Image or ndarray, any dtype) to HxWx3 uint8 RGB."""
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:  # grayscale -> RGB
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:  # RGBA -> RGB
        frame = frame[..., :3]
    return frame

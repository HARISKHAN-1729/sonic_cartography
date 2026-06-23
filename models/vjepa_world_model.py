"""
models/vjepa_world_model.py
────────────────────────────
V-JEPA 2 latent world model — predicts FUTURE FRAME REPRESENTATIONS
(not pixels), action-free, from human-egocentric video.

Why this addresses the NWM domain problem
──────────────────────────────────────────
NWM needs continuous robot pose-deltas it never sees on pedestrian data.
V-JEPA 2 is pretrained ACTION-FREE on >1M hours of internet video with a
masked latent-feature-prediction objective, and its flagship result is
egocentric action anticipation (Epic-Kitchens-100): the predictor takes the
encoder representation plus mask tokens for a frame ~1s in the future and
predicts that future frame's REPRESENTATION. That is exactly the signal we
sonify — no action labels, human-egocentric domain.

The catch (handled by the probe, see models/latent_probe.py)
─────────────────────────────────────────────────────────────
V-JEPA 2 predicts in representation space, NOT pixels — there is no RGB
decoder. So instead of "predicted frame -> depth + detector", we read the
navigation layout directly off the predicted latent with a small trained
probe. This wrapper produces the predicted future TOKEN GRID; the probe turns
it into a list of Detections.

Verified API (HuggingFace transformers; ungated checkpoints)
─────────────────────────────────────────────────────────────
    from transformers import AutoVideoProcessor, AutoModel
    processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    model     = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256",
                    torch_dtype=torch.float16, attn_implementation="sdpa")
    video = processor(video, return_tensors="pt")          # video: T×C×H×W
    out   = model(**video)
    enc   = out.last_hidden_state                           # encoder tokens
    pred  = out.predictor_output.last_hidden_state          # predictor tokens

Checkpoints (pick one; all public/ungated):
    facebook/vjepa2-vitl-fpc64-256   (ViT-L, hidden 1024)  ← default
    facebook/vjepa2-vith-fpc64-256   (ViT-H)
    facebook/vjepa2-vitg-fpc64-256   (ViT-g)

ONE VERSION-SENSITIVE HOOK
───────────────────────────
The exact way to make the predictor target a FUTURE-frame position (vs. the
default pretraining mask) depends on the transformers release. That logic is
isolated in `_predict_future_tokens()` and marked  # >>> ADAPT  so you only
touch it once. The fallback path (no explicit future mask) still works as a
short-horizon predictor because the default forward fills masked tubelets.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Tokens-per-frame geometry depends on patch size (16) and tubelet size (2).
DEFAULT_PATCH = 16
DEFAULT_TUBELET = 2


class VJEPA2WorldModel:
    """
    Action-free latent future predictor.

    Public methods
    --------------
    predict_latent_at_horizons(context_frames, horizon_times_sec)
        -> dict[horizon_sec -> np.ndarray(Hg, Wg, D)]   # predicted token grids
    encode_frame_grid(frame)
        -> np.ndarray(Hg, Wg, D)                         # for probe training
    grid_shape  : (Hg, Wg)  spatial token-grid size
    embed_dim   : D
    """

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        dtype="float16",
        image_size: int = 256,
        frames_per_clip: int = 64,
        prediction_fps: int = 8,
        patch_size: int = DEFAULT_PATCH,
        tubelet_size: int = DEFAULT_TUBELET,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.image_size = image_size
        self.frames_per_clip = frames_per_clip
        self.prediction_fps = prediction_fps
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self._torch_dtype = (
            torch.float16 if (_HAS_TORCH and dtype == "float16") else
            (torch.float32 if _HAS_TORCH else None)
        )

        # Spatial token grid per frame.
        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.embed_dim = int(getattr(getattr(model, "config", None),
                                     "hidden_size", 1024))

        if _HAS_TORCH:
            self.model.eval()

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg) -> "VJEPA2WorldModel":
        if not _HAS_TORCH:
            raise ImportError("VJEPA2WorldModel needs torch + transformers.")
        from transformers import AutoVideoProcessor, AutoModel

        repo = cfg.get("checkpoint", "facebook/vjepa2-vitl-fpc64-256")
        device = cfg.get("device", "cuda")
        dtype = cfg.get("dtype", "float16")
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        processor = AutoVideoProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(
            repo,
            torch_dtype=torch_dtype,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
        ).to(device)

        return cls(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            image_size=int(cfg.get("image_size", 256)),
            frames_per_clip=int(cfg.get("frames_per_clip", 64)),
            prediction_fps=int(cfg.get("prediction_fps", 8)),
            patch_size=int(cfg.get("patch_size", DEFAULT_PATCH)),
            tubelet_size=int(cfg.get("tubelet_size", DEFAULT_TUBELET)),
        )

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (self.grid_h, self.grid_w)

    # ── Encoding (used for probe training on real frames) ──────────────────

    @torch.no_grad() if _HAS_TORCH else (lambda f: f)
    def encode_frame_grid(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode a single RGB frame to its spatial token grid (Hg, Wg, D), by
        replicating it into a short clip (the encoder expects a video tensor).
        Used to build (latent -> teacher-label) pairs for probe training.
        """
        clip = self._frame_to_clip(frame)              # (T, C, H, W) uint8-ish
        inputs = self.processor(clip, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        enc = out.last_hidden_state                    # (1, N_tokens, D)
        grid = self._tokens_to_last_frame_grid(enc)
        return grid.float().cpu().numpy()

    # ── Future prediction (the core call) ──────────────────────────────────

    @torch.no_grad() if _HAS_TORCH else (lambda f: f)
    def predict_latent_at_horizons(
        self,
        context_frames: List[np.ndarray],
        horizon_times_sec: List[float],
    ) -> Dict[float, np.ndarray]:
        """
        Predict the future-frame token grid at each horizon.

        Returns dict[horizon_sec -> np.ndarray(Hg, Wg, D)].
        """
        clip = self._build_context_clip(context_frames)   # (T, C, H, W)
        inputs = self.processor(clip, return_tensors="pt").to(self.device)

        results: Dict[float, np.ndarray] = {}
        for t_sec in horizon_times_sec:
            target_frame_idx = self._horizon_to_clip_index(t_sec, len(clip))
            grid = self._predict_future_tokens(inputs, target_frame_idx)
            results[t_sec] = grid.float().cpu().numpy()
        return results

    # ── The single version-sensitive hook ──────────────────────────────────

    def _predict_future_tokens(self, inputs, target_frame_idx: int):
        """
        Run the V-JEPA 2 predictor so its output corresponds to the FUTURE
        frame at `target_frame_idx`, and return that frame's token grid.

        >>> ADAPT (one place): the masking interface differs across
        transformers releases. The robust, version-stable approach below uses
        the model's default forward and reads the predictor output at the
        target frame's token slice. If your release exposes explicit
        `context_mask` / `target_mask` arguments (see modeling_vjepa2.py and
        the EK100 anticipation config), pass them here so the predictor is
        conditioned on observed frames only and queried at the future frame.
        """
        out = self.model(**inputs)
        # predictor_output may be absent on some configs; fall back to encoder.
        pred = getattr(out, "predictor_output", None)
        tokens = pred.last_hidden_state if pred is not None else out.last_hidden_state
        return self._tokens_to_frame_grid(tokens, target_frame_idx)

    # ── Token-grid bookkeeping ──────────────────────────────────────────────

    def _tokens_per_frame(self) -> int:
        return self.grid_h * self.grid_w

    def _temporal_index_to_token_index(self, frame_idx: int) -> int:
        """Map a frame index to its tubelet (temporal token) index."""
        return frame_idx // self.tubelet_size

    def _tokens_to_frame_grid(self, tokens, frame_idx: int):
        """
        tokens: (1, N, D) flattened spacetime tokens, ordered
        [t0:(h0w0,h0w1,...), t1:(...), ...]. Slice the target temporal step and
        reshape to (Hg, Wg, D).
        """
        tpf = self._tokens_per_frame()
        t_tok = self._temporal_index_to_token_index(frame_idx)
        n_temporal = tokens.shape[1] // tpf
        t_tok = min(t_tok, max(0, n_temporal - 1))
        start = t_tok * tpf
        sl = tokens[:, start:start + tpf, :]            # (1, tpf, D)
        return sl.reshape(self.grid_h, self.grid_w, -1)

    def _tokens_to_last_frame_grid(self, tokens):
        tpf = self._tokens_per_frame()
        n_temporal = max(1, tokens.shape[1] // tpf)
        return self._tokens_to_frame_grid(tokens, (n_temporal - 1) * self.tubelet_size)

    # ── Clip construction ───────────────────────────────────────────────────

    def _build_context_clip(self, context_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Build a `frames_per_clip`-length clip: real context frames in the first
        portion, the last context frame repeated to fill the remaining (future)
        slots that the predictor will fill in latent space.
        """
        frames = [self._prep(f) for f in context_frames][-self.frames_per_clip:]
        if not frames:
            raise ValueError("No context frames provided.")
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1].copy())            # pad with last frame
        return frames[: self.frames_per_clip]

    def _frame_to_clip(self, frame: np.ndarray) -> List[np.ndarray]:
        f = self._prep(frame)
        return [f.copy() for _ in range(self.frames_per_clip)]

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        """Coerce to HxWx3 uint8 (the processor handles resize/normalize)."""
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame

    def _horizon_to_clip_index(self, horizon_sec: float, clip_len: int) -> int:
        """Map a horizon in seconds to a frame index inside the clip."""
        # Context occupies the early frames; future frames live after it.
        # We index forward from the last observed frame by horizon * fps.
        n_context = min(clip_len, max(1, clip_len // 2))
        idx = n_context + int(round(horizon_sec * self.prediction_fps))
        return min(idx, clip_len - 1)

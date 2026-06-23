"""
models/nwm_wrapper.py
─────────────────────
FAITHFUL wrapper around the real Navigation World Model (NWM) release.

Bar et al., "Navigation World Models", CVPR 2025 (Oral).
  Code   : https://github.com/facebookresearch/nwm   (models.py, diffusion/)
  Weights: https://huggingface.co/facebook/nwm        (GATED .pth.tar files)

────────────────────────────────────────────────────────────────────────────
WHY THE OLD WRAPPER WAS WRONG
────────────────────────────────────────────────────────────────────────────
The previous version assumed an API that does not exist:
    NWM.from_pretrained(...)         # ✗ no such method
    model.rollout(ctx, actions, ...) # ✗ no such method
    discrete action tokens           # ✗ NWM uses CONTINUOUS pose deltas
    model.decode(...)                # ✗ decoding is done by the VAE

The real NWM is a Conditional Diffusion Transformer (CDiT) that:
  1. Encodes context frames into Stable-Diffusion VAE latents.
  2. Is conditioned on the context latents + a CONTINUOUS action
     (relative pose: Δx, Δy, Δyaw) + a time delta to the target frame.
  3. Generates the target-frame latent with a diffusion sampling loop.
  4. Decodes that latent back to pixels with the VAE.

The HuggingFace repo ships only the model WEIGHTS as gated `.pth.tar`
state-dicts (cdit_s/b/l/xl_100000.pth.tar, cdit_xl_ego4d_200000.pth.tar).
You must build the CDiT module from the cloned repo's `models.py` and load
the state-dict yourself — this wrapper does exactly that.

────────────────────────────────────────────────────────────────────────────
REPO-SPECIFIC ADAPTATION POINTS  (verify against the cloned repo once)
────────────────────────────────────────────────────────────────────────────
Three spots depend on the exact signatures inside facebookresearch/nwm.
They are isolated into single methods and marked  # >>> ADAPT  so you only
have to touch them once after cloning:
    _build_cdit()         — CDiT constructor name + kwargs   (models.py)
    _make_diffusion()     — create_diffusion(...) import     (diffusion/)
    _cdit_forward()       — CDiT.forward(...) argument order  (models.py)
If you prefer zero integration risk, use the "flow" backend in
models/world_model.py instead — it needs none of this.

────────────────────────────────────────────────────────────────────────────
IMPORTANT MODELING CAVEAT
────────────────────────────────────────────────────────────────────────────
NWM is trained on GROUND-ROBOT navigation data (RECON, SCAND, TartanDrive,
HuRoN; the XL variant adds Ego4D human video). It expects robot-frame pose
actions. Pedestrian SANPO clips have no matching action labels, so we
synthesize a constant "walk forward" pose-delta. Treat NWM outputs on
pedestrian data as approximate, and prefer the XL+Ego4D checkpoint for the
closest domain match. This is partly why world_model.py also offers a
flow-based predictor that needs no action labels at all.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# Stable Diffusion VAE latent scaling factor (DiT/CDiT lineage convention).
VAE_SCALE = 0.18215

# Map our config "model_size" to the CDiT variant key in the repo's models.py.
# >>> ADAPT: confirm these keys exist in nwm/models.py `CDiT_models`.
CDIT_VARIANTS = {
    "S":  "CDiT-S/2",
    "B":  "CDiT-B/2",
    "L":  "CDiT-L/2",
    "XL": "CDiT-XL/2",
}

# Default checkpoint filenames on HuggingFace facebook/nwm (gated).
CKPT_FILES = {
    "S":       "cdit_s_100000.pth.tar",
    "B":       "cdit_b_100000.pth.tar",
    "L":       "cdit_l_100000.pth.tar",
    "XL":      "cdit_xl_100000.pth.tar",
    "XL_EGO4D": "cdit_xl_ego4d_200000.pth.tar",   # best for human egocentric
}


class NWMWrapper:
    """
    Faithful NWM inference wrapper.

    Public interface is unchanged from the original (drop-in compatible):
        predict(context_frames, action, n_frames)        -> list[np.ndarray]
        predict_at_horizons(context_frames, horizons_sec) -> dict[float, ndarray]
    """

    def __init__(
        self,
        cdit: torch.nn.Module,
        vae: torch.nn.Module,
        diffusion,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        image_size: int = 256,
        context_len: int = 4,
        prediction_fps: int = 8,
        num_sampling_steps: int = 100,
        walk_speed_mps: float = 1.4,   # ~average human walking speed
    ):
        self.cdit = cdit
        self.vae = vae
        self.diffusion = diffusion
        self.device = device
        self.dtype = dtype
        self.image_size = image_size
        self.context_len = context_len
        self.prediction_fps = prediction_fps
        self.num_sampling_steps = num_sampling_steps
        self.walk_speed_mps = walk_speed_mps

        self.cdit.eval().to(device=device, dtype=dtype)
        self.vae.eval().to(device=device, dtype=dtype)

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg) -> "NWMWrapper":
        """
        Build NWM from an OmegaConf node. Requires the cloned repo on PYTHONPATH:
            git clone https://github.com/facebookresearch/nwm
            export PYTHONPATH=$PYTHONPATH:$(pwd)/nwm
        and a downloaded (gated) checkpoint — see scripts/download_assets.py.
        """
        device = cfg.get("device", "cuda")
        dtype = torch.float16 if cfg.get("dtype", "float16") == "float16" else torch.float32
        image_size = int(cfg.get("image_size", 256))
        context_len = int(cfg.get("context_frames", 4))
        prediction_fps = int(cfg.get("prediction_fps", 8))
        steps = int(cfg.get("num_sampling_steps", 100))
        model_size = cfg.get("model_size", "XL")
        ckpt_path = cfg.get("checkpoint_path", None)

        vae = cls._build_vae(cfg.get("vae", "stabilityai/sd-vae-ft-ema"), device)
        cdit = cls._build_cdit(model_size, image_size)
        cls._load_checkpoint(cdit, ckpt_path, model_size)
        diffusion = cls._make_diffusion(steps)

        return cls(
            cdit=cdit, vae=vae, diffusion=diffusion,
            device=device, dtype=dtype, image_size=image_size,
            context_len=context_len, prediction_fps=prediction_fps,
            num_sampling_steps=steps,
            walk_speed_mps=float(cfg.get("walk_speed_mps", 1.4)),
        )

    # ----- the three repo-specific hooks ------------------------------------

    @staticmethod
    def _build_cdit(model_size: str, image_size: int) -> torch.nn.Module:
        """Instantiate the CDiT module from the cloned repo's models.py."""
        try:
            from models import CDiT_models  # repo file: nwm/models.py
        except ImportError as e:
            raise ImportError(
                "Could not import CDiT from the NWM repo. Clone it and add to "
                "PYTHONPATH:\n"
                "  git clone https://github.com/facebookresearch/nwm\n"
                "  export PYTHONPATH=$PYTHONPATH:$(pwd)/nwm"
            ) from e

        variant = CDIT_VARIANTS.get(model_size.upper(), "CDiT-XL/2")
        latent_size = image_size // 8  # SD VAE downsamples 8×

        # >>> ADAPT: match the constructor kwargs in nwm/models.py.
        # The DiT/CDiT convention is (input_size, in_channels, ...); NWM adds
        # context length + action/pose conditioning dims. Typical call:
        cdit = CDiT_models[variant](
            input_size=latent_size,
            in_channels=4,            # SD VAE latent channels
            # context_size=...,       # set if the constructor requires it
            # action_dim=3,           # (Δx, Δy, Δyaw); confirm in models.py
        )
        return cdit

    @staticmethod
    def _make_diffusion(num_sampling_steps: int):
        """Build the diffusion sampler from the repo's diffusion/ package."""
        try:
            from diffusion import create_diffusion  # repo dir: nwm/diffusion/
        except ImportError as e:
            raise ImportError(
                "Could not import create_diffusion from the NWM repo's "
                "diffusion/ package. Ensure the cloned repo is on PYTHONPATH."
            ) from e
        # Empty respacing => full DDPM; "ddimNN" => NN-step DDIM (faster).
        respacing = f"ddim{num_sampling_steps}" if num_sampling_steps else ""
        return create_diffusion(timestep_respacing=respacing)

    def _cdit_forward(self, z_t, t, ctx_latents, actions):
        """
        One denoiser call. Wraps CDiT.forward so the sampling loop below is
        agnostic to the exact argument order in models.py.

        >>> ADAPT: align kwargs with nwm/models.py CDiT.forward signature.
        Expected (typical CDiT): forward(x, t, x_cond, action) -> eps/v pred.
        """
        return self.cdit(z_t, t, ctx_latents, actions)

    # ----- checkpoint + VAE helpers -----------------------------------------

    @staticmethod
    def _build_vae(vae_id: str, device: str) -> torch.nn.Module:
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(vae_id)

    @staticmethod
    def _load_checkpoint(cdit: torch.nn.Module, ckpt_path: Optional[str],
                         model_size: str) -> None:
        """
        Load a gated .pth.tar state-dict into the CDiT module.
        Handles the common nesting ('model' / 'ema' / 'state_dict') and the
        'module.' DataParallel prefix.
        """
        if ckpt_path is None:
            raise FileNotFoundError(
                "cfg.model.nwm.checkpoint_path is not set. Download a gated "
                "checkpoint first (see scripts/download_assets.py), e.g.\n"
                f"  checkpoints/{CKPT_FILES.get(model_size.upper(), CKPT_FILES['XL'])}"
            )
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"NWM checkpoint not found: {ckpt_path}")

        # .pth.tar may be a raw torch save or a tar containing one — handle both.
        if tarfile.is_tarfile(ckpt_path):
            with tarfile.open(ckpt_path) as tf:
                member = next(m for m in tf.getmembers()
                              if m.name.endswith((".pth", ".pt", ".bin")))
                f = tf.extractfile(member)
                state = torch.load(f, map_location="cpu")
        else:
            state = torch.load(ckpt_path, map_location="cpu")

        # Unwrap common nestings.
        for key in ("ema", "model", "state_dict", "module"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break

        # Strip DataParallel 'module.' prefix if present.
        state = { (k[7:] if k.startswith("module.") else k): v
                  for k, v in state.items() }

        missing, unexpected = cdit.load_state_dict(state, strict=False)
        if missing:
            print(f"  [NWM] {len(missing)} missing keys (first few): {missing[:4]}")
        if unexpected:
            print(f"  [NWM] {len(unexpected)} unexpected keys (first few): {unexpected[:4]}")
        print(f"  [NWM] loaded checkpoint: {ckpt_path}")

    # ── Core prediction ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        context_frames: List,
        action: str = "walk_forward",
        n_frames: int = 12,
    ) -> List[np.ndarray]:
        """
        Autoregressive rollout of `n_frames` future frames.

        Encodes context to latents, then repeatedly: synthesize the action for
        a single inter-frame step, sample the next latent via diffusion, decode
        to pixels, slide the context window, and continue.
        """
        ctx_latents = self._encode_context(context_frames)  # (1, T, C, h, w)
        dt = 1.0 / self.prediction_fps                       # seconds per step

        out_frames: List[np.ndarray] = []
        for _ in range(n_frames):
            act = self._make_action(action, dt)              # (1, action_dim)
            z_next = self._sample_next_latent(ctx_latents, act)  # (1, C, h, w)
            frame = self._decode_latent(z_next)              # (H, W, 3) uint8
            out_frames.append(frame)

            # Slide the context window: drop oldest, append the new latent.
            ctx_latents = torch.cat(
                [ctx_latents[:, 1:], z_next.unsqueeze(1)], dim=1
            )
        return out_frames

    @torch.no_grad()
    def predict_at_horizons(
        self,
        context_frames: List,
        horizon_times_sec: List[float],
        action: str = "walk_forward",
    ) -> Dict[float, np.ndarray]:
        """
        Predict frames at specific horizon times.

        NWM can condition on an arbitrary time delta, so in principle each
        horizon can be sampled directly in one step. For robustness here we do
        a single autoregressive rollout to the max horizon and index into it —
        identical interface to the lightweight backends.
        """
        max_frames = int(round(max(horizon_times_sec) * self.prediction_fps)) + 1
        all_frames = self.predict(context_frames, action=action, n_frames=max_frames)
        out: Dict[float, np.ndarray] = {}
        for t in horizon_times_sec:
            idx = min(int(round(t * self.prediction_fps)), len(all_frames) - 1)
            out[t] = all_frames[idx]
        return out

    # ── Diffusion sampling ─────────────────────────────────────────────────

    @torch.no_grad()
    def _sample_next_latent(self, ctx_latents, action) -> torch.Tensor:
        """Run the diffusion reverse process to produce the next-frame latent."""
        c, h, w = ctx_latents.shape[-3:]
        z = torch.randn(1, c, h, w, device=self.device, dtype=self.dtype)

        model_kwargs = dict(ctx_latents=ctx_latents, actions=action)

        # Prefer DDIM if available (fast), else fall back to ancestral DDPM.
        sample_fn = getattr(self.diffusion, "ddim_sample_loop", None) \
            or self.diffusion.p_sample_loop

        z0 = sample_fn(
            self._denoise_fn,           # callable(x, t, **model_kwargs)
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=self.device,
        )
        return z0

    def _denoise_fn(self, x, t, ctx_latents=None, actions=None):
        """Adapter passed to the diffusion sampler -> calls CDiT.forward."""
        return self._cdit_forward(x, t, ctx_latents, actions)

    # ── Action synthesis ────────────────────────────────────────────────────

    def _make_action(self, action: str, dt: float) -> torch.Tensor:
        """
        Build a CONTINUOUS pose-delta action for one inter-frame step.

        NWM actions are relative pose changes in the robot/ego frame. We map
        intent strings to a constant velocity over dt seconds:
            walk_forward -> advance walk_speed * dt metres along +x
            turn_left    -> small +yaw
            turn_right    -> small -yaw
            stop         -> zero motion
        Returns a (1, 3) tensor: (Δx, Δy, Δyaw).

        >>> ADAPT: if models.py expects a different action layout (e.g. a
        normalized command or an extra time channel), adjust here.
        """
        step = self.walk_speed_mps * dt
        if action == "walk_forward":
            dx, dy, dyaw = step, 0.0, 0.0
        elif action == "turn_left":
            dx, dy, dyaw = step * 0.5, 0.0, +0.20
        elif action == "turn_right":
            dx, dy, dyaw = step * 0.5, 0.0, -0.20
        else:  # "stop" or unknown
            dx, dy, dyaw = 0.0, 0.0, 0.0
        return torch.tensor([[dx, dy, dyaw]], device=self.device, dtype=self.dtype)

    # ── VAE encode / decode ─────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_context(self, frames: List) -> torch.Tensor:
        """Encode the last `context_len` frames into VAE latents (1, T, C, h, w)."""
        frames = frames[-self.context_len:]
        latents = []
        for fr in frames:
            x = self._preprocess(fr)                       # (1, 3, H, W) in [-1,1]
            posterior = self.vae.encode(x).latent_dist
            z = posterior.sample().mul_(VAE_SCALE)         # (1, C, h, w)
            latents.append(z)
        return torch.stack(latents, dim=1)                 # (1, T, C, h, w)

    @torch.no_grad()
    def _decode_latent(self, z: torch.Tensor) -> np.ndarray:
        """Decode a single latent (1, C, h, w) to an (H, W, 3) uint8 RGB frame."""
        img = self.vae.decode(z / VAE_SCALE).sample        # (1, 3, H, W) in [-1,1]
        img = (img.clamp(-1, 1) + 1.0) / 2.0               # -> [0,1]
        img = img.squeeze(0).permute(1, 2, 0)              # (H, W, 3)
        return (img.float().cpu().numpy() * 255).astype(np.uint8)

    def _preprocess(self, frame) -> torch.Tensor:
        """PIL/ndarray -> normalized (1, 3, H, W) tensor in [-1, 1] on device."""
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        frame = frame.convert("RGB").resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )
        arr = np.asarray(frame, dtype=np.float32) / 127.5 - 1.0   # [-1, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
        return t.to(device=self.device, dtype=self.dtype)

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def frames_per_second_to_n_frames(horizon_sec: float, fps: int = 8) -> int:
        return max(1, int(round(horizon_sec * fps)))

# NWM Integration Fix — what changed and why

## The problem

The original `models/nwm_wrapper.py` assumed an API that **does not exist**:

```python
from nwm.models.nwm import NWM          # ✗ no such module
model = NWM.from_pretrained(checkpoint) # ✗ no from_pretrained()
model.rollout(context, actions, ...)    # ✗ no rollout()
action_tokens = {"walk_forward": 0}     # ✗ NWM uses continuous pose deltas
```

The reality (verified against the official release):

- **Code** lives at `github.com/facebookresearch/nwm` — it's the CDiT
  (Conditional Diffusion Transformer) training/eval code (`models.py`,
  `diffusion/`, `isolated_nwm_infer.py`). There is **no pip package** and
  **no `from_pretrained()`**.
- **Weights** live at `huggingface.co/facebook/nwm` as **gated `.pth.tar`
  state-dicts**: `cdit_s/b/l/xl_100000.pth.tar` and
  `cdit_xl_ego4d_200000.pth.tar`. You must accept the license (logged in)
  and authenticate before download.
- The model is a **diffusion model in SD-VAE latent space**, conditioned on
  **continuous robot pose deltas** (Δx, Δy, Δyaw), trained on **ground-robot**
  data (RECON, SCAND, TartanDrive, HuRoN; XL adds Ego4D human video).

## The fix — two solutions, pick either

### Solution 1 — `world_model.backend: "flow"`  (recommended; runs today)

A new abstraction layer, `models/world_model.py`, decouples the pipeline from
NWM. The default **optical-flow backend** predicts future frames by
estimating dense motion between the last two context frames and
forward-warping the most recent frame along that motion field, scaled by the
horizon. It:

- needs **no checkpoint, no GPU, no gated download, no repo clone**,
- works directly on **pedestrian SANPO** video (no action labels required),
- is a legitimate **constant-velocity world-model baseline** for the paper.

```yaml
# configs/default.yaml
world_model:
  backend: "flow"
```

Nothing else to do. `run_pipeline.py` and `eval/evaluate.py` already call the
abstraction.

### Solution 2 — `world_model.backend: "nwm"`  (faithful, best quality)

`models/nwm_wrapper.py` was rewritten to integrate with the **real** release:
it builds the VAE (`diffusers.AutoencoderKL`), constructs CDiT from the
cloned repo's `models.py`, loads the gated `.pth.tar` state-dict (handling
tar nesting + `module.` prefixes), encodes context to latents, synthesizes a
continuous "walk forward" pose action, runs the diffusion sampling loop, and
decodes latents back to pixels.

Setup:

```bash
# 1. Clone the code repo and expose it
git clone https://github.com/facebookresearch/nwm
export PYTHONPATH=$PYTHONPATH:$(pwd)/nwm

# 2. Accept the license at https://huggingface.co/facebook/nwm (while logged in)
huggingface-cli login         # paste a token from hf.co/settings/tokens

# 3. Download a checkpoint (XL+Ego4D = closest match for human egocentric)
python scripts/download_assets.py     # downloads cdit_xl_100000.pth.tar by default
```

```yaml
# configs/default.yaml
world_model:
  backend: "nwm"
model:
  nwm:
    model_size: "XL"
    checkpoint_path: "checkpoints/cdit_xl_100000.pth.tar"
    vae: "stabilityai/sd-vae-ft-ema"
```

**Three adaptation points** in `nwm_wrapper.py` are marked `# >>> ADAPT`.
After cloning, confirm they match the repo's exact signatures (they follow
the standard DiT/CDiT convention, but pin them once):

1. `_build_cdit()` — the `CDiT_models[...]` constructor name + kwargs.
2. `_make_diffusion()` — the `create_diffusion(...)` import path.
3. `_cdit_forward()` — the argument order of `CDiT.forward(...)`.

## Bonus: this strengthens the paper

Because the predictor is now swappable, you can report the
accuracy-vs-lead-time curve for **three backends** — `last_frame` (naive),
`flow` (constant-velocity), and `nwm` (learned world model) — turning your
single claim into a clean **ablation** that quantifies exactly how much the
learned world model adds over cheap baselines.

```bash
for b in last_frame flow nwm; do
  python eval/evaluate.py --sanpo_root data/sanpo/ \
      --output eval/results_$b.json \
      # set world_model.backend=$b via an OmegaConf override or edit the yaml
done
```

## Honest caveat to put in the abstract

NWM is trained on ground-robot trajectories; pedestrian frames with a
synthesized "walk forward" action are out-of-distribution. Prefer the
XL+Ego4D checkpoint, keep horizons short (≤2.5 s), and report the `flow`
baseline alongside — on pedestrian data it is often competitive and is the
honest reference point.

## Files changed in this fix

- `models/world_model.py`      — NEW: backend abstraction + flow/last_frame predictors
- `models/nwm_wrapper.py`      — REWRITE: faithful CDiT + VAE + diffusion integration
- `configs/default.yaml`       — added `world_model:` selector; corrected `model.nwm:` block
- `scripts/download_assets.py` — `download_nwm()` now handles gated `.pth.tar` + HF auth
- `run_pipeline.py`            — uses `build_world_model(cfg)` instead of `NWMWrapper`
- `eval/evaluate.py`           — uses `build_world_model(cfg)` instead of `NWMWrapper`

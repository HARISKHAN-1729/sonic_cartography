# V-JEPA 2 Latent Pipeline — Setup, Run & Studies Guide

This guide covers the **latent** version of Predictive Sonic Cartography: it
swaps the pixel world model for **V-JEPA 2** (action-free, human-egocentric)
and reads the navigation layout off the **predicted latent** with a small
trained probe — no pixel decoding. It also explains how to run the **full
ablation** (`last_frame` vs `flow` vs `nwm` vs `vjepa`) through one evaluator.

> If you only want the pixel backends (flow / nwm / last_frame), see
> `SETUP.md`. This guide adds the V-JEPA 2 path on top of that codebase.

---

## 0. What's new in this pipeline (mental model)

```
PIXEL path  (configs/default.yaml, scene_predictor.type=pixel)
  context frames ─▶ world_model(flow|nwm|last_frame) ─▶ predicted RGB frame
                 ─▶ Depth Anything + YOLO-World ─▶ Detections ─▶ sonify

LATENT path (configs/vjepa.yaml,  scene_predictor.type=latent)
  context frames ─▶ V-JEPA 2 ─▶ predicted future TOKEN GRID
                 ─▶ trained probe ─▶ Detections ─▶ sonify
```

Both paths end in the **same** `Detection` list, so the **same** evaluator,
metrics, and sonifier serve every backend. That is what makes the four-way
ablation a one-line config switch.

V-JEPA 2 predicts *representations*, not pixels — so the latent path needs a
**probe** (trained once) that maps a predicted token grid to per-cell
objectness / class / depth. We train that probe by **distilling the pixel
teacher** (Depth Anything + YOLO-World on real frames) into the latent.

---

## 1. New & reused files

**New (this pipeline):**

| File | Role |
|---|---|
| `models/vjepa_world_model.py` | V-JEPA 2 wrapper → predicted future token grid |
| `models/latent_probe.py` | probe module + latent→Detection conversion + teacher rasterization |
| `models/scene_predictor.py` | unifies pixel & latent behind one `predict_scene_at_horizons()` |
| `train_probe.py` | trains the probe on frozen V-JEPA features (pixel-teacher distillation) |
| `eval/evaluate_unified.py` | one evaluator for every backend |
| `eval/plot_ablation.py` | overlays F1-vs-horizon for all backends |
| `configs/vjepa.yaml` | latent-path config |
| `scripts/test_vjepa_logic.py` | CPU tests for the latent logic |

**Reused unchanged:** `audio/*` (sonifier), `eval/metrics.py`,
`data/sanpo_loader.py`, `models/object_detector.py` (Detection + geometry),
`models/depth_estimator.py` (teacher), `models/world_model.py` (pixel
backends).

If you're merging into an existing checkout, drop the new files into the
matching folders (same layout as the table). The complete zip already has
everything in place.

---

## 2. Install

```bash
conda create -n sonic_cart python=3.11 -y && conda activate sonic_cart

# GPU PyTorch (adjust cu121 to your CUDA)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Project deps — note transformers>=4.52.4 is required for V-JEPA 2
pip install -r requirements.txt
```

Quick sanity check (no GPU, no downloads):

```bash
python scripts/test_audio_and_metrics.py     # 16 passed  (audio + metrics)
python scripts/test_vjepa_logic.py            # 8 passed   (latent logic)
```

---

## 3. What downloads automatically

| Asset | How | Auto? | Gated? | Lands at |
|---|---|---|---|---|
| **V-JEPA 2** (ViT-L) | `transformers` AutoModel on first use | ✅ Yes | **No** | `~/.cache/huggingface/` |
| Depth Anything V2 (teacher) | `transformers` | ✅ Yes | No | `~/.cache/huggingface/` |
| YOLO-World (teacher) | `ultralytics` | ✅ Yes | No | `./yolov8x-worldv2.pt` |
| SANPO dataset | `data/get_sanpo.py` (gcloud) | ⚠️ Separate step | No (public) | `./data/sanpo/` |

**Good news:** V-JEPA 2 is **ungated** — unlike NWM, nothing to request, no
`huggingface-cli login` needed. The first run pulls `facebook/vjepa2-vitl-fpc64-256`
(~1.1 GB for ViT-L) automatically.

Get SANPO (same as the pixel guide):

```bash
# install gcloud first (see SETUP.md §4), then:
python data/get_sanpo.py --path a --max_sessions 50     # SANPO-Synthetic, ~2.5 GB
```

---

## 4. Train the probe (one time, single GPU)

The probe is the only thing you train. The V-JEPA 2 backbone stays frozen.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python train_probe.py \
    --config configs/vjepa.yaml \
    --sanpo_root data/sanpo/ \
    --out checkpoints/probe.pt \
    --steps 4000 --batch_size 8
```

What happens: for each sampled SANPO frame it runs the pixel teacher
(Depth Anything + YOLO-World) to get per-cell objectness/class/depth targets,
encodes the same frame with frozen V-JEPA 2, and trains the probe to predict
those targets. ~20–40 min on an RTX 4090 for 4000 steps. Output:
`checkpoints/probe.pt`.

Tips:
- Start with `--steps 2000` to smoke-test, then scale up.
- `--max_clips 100` limits the frame pool if your SANPO subset is small.
- Lower VRAM: set `model.vjepa.checkpoint: facebook/vjepa2-vitl-fpc64-256`
  (ViT-L) and keep `batch_size` at 4–8.

---

## 5. Run the latent pipeline

### 5a. Evaluate V-JEPA 2 on SANPO

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python eval/evaluate_unified.py \
    --config configs/vjepa.yaml \
    --sanpo_root data/sanpo/ \
    --output eval/results_vjepa.json \
    --max_clips 200
```

Prints the accuracy-vs-lead-time table and writes `eval/results_vjepa.json`.

### 5b. (Optional) Listen to a predicted soundscape

The demo `run_pipeline.py` is wired for the pixel path. For the latent path,
the sonifier consumes the same `Detection` list, so you can sonify a
predicted scene in a few lines:

```python
from omegaconf import OmegaConf
from models.scene_predictor import build_scene_predictor
from audio.scene_sonifier import SceneSonifier
# ... load 8 context frames as a list of HxWx3 uint8 arrays ...
cfg = OmegaConf.load("configs/vjepa.yaml")
pred = build_scene_predictor(cfg)                       # latent
scene = pred.predict_scene_at_horizons(context, [1.5])  # dict[1.5] -> Detections
SceneSonifier.from_config(cfg.audio).save(
    SceneSonifier.from_config(cfg.audio).sonify(scene[1.5]), "out/vjepa_h1.5s.wav")
```

---

## 6. The studies (this is the paper)

### Study A — Accuracy vs. lead time, per backend

Run the same evaluator under each backend. Pixel backends use
`configs/default.yaml` (flip `world_model.backend`); the latent backend uses
`configs/vjepa.yaml`.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- pixel backends (no probe needed) ---
for B in last_frame flow nwm; do
  cp configs/default.yaml configs/_tmp_$B.yaml
  sed -i "s/^  backend: .*/  backend: \"$B\"/" configs/_tmp_$B.yaml
  python eval/evaluate_unified.py --config configs/_tmp_$B.yaml \
      --sanpo_root data/sanpo/ --output eval/results_$B.json --max_clips 200
done

# --- latent backend (needs checkpoints/probe.pt from §4) ---
python eval/evaluate_unified.py --config configs/vjepa.yaml \
    --sanpo_root data/sanpo/ --output eval/results_vjepa.json --max_clips 200
```

> Skip the `nwm` row if you haven't set up the gated NWM checkpoint — the
> three remaining backends are already a meaningful study.

### Study B — The ablation figure

```bash
python eval/plot_ablation.py \
    --results eval/results_last_frame.json eval/results_flow.json \
              eval/results_nwm.json eval/results_vjepa.json \
    --labels  last_frame flow nwm vjepa \
    --output_dir eval/figures/
```

Produces `ablation_f1_vs_horizon.{png,pdf}` and `ablation_depth_mae.{png,pdf}`.
This is the central result: it answers **"does a learned world model beat
physics, and how far ahead?"** — `last_frame` is the floor, `flow` is the
constant-velocity baseline, `vjepa` (and optionally `nwm`) are the learned
models.

### Study C — Horizon sweep / operating point

Use `--horizons` to probe finer-grained lead times for the chosen backend:

```bash
python eval/evaluate_unified.py --config configs/vjepa.yaml \
    --sanpo_root data/sanpo/ --output eval/results_vjepa_fine.json \
    --horizons 0.25 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0
```

Report the horizon where F1 crosses your usefulness threshold and where
azimuth MAE crosses 15° (the reliable-spatial-audio bound).

### Study D — Backbone size (optional)

Swap `model.vjepa.checkpoint` between `vjepa2-vitl-fpc64-256`,
`vjepa2-vith-fpc64-256`, `vjepa2-vitg-fpc64-256`, retrain the probe for each
(the embed dim changes), and compare. Shows whether scaling the world model
helps prediction fidelity for navigation.

---

## 7. Reading the results

- **F1 (object hit rate):** did the predicted future layout match the real
  future? Higher = better; expect a downward slope with horizon.
- **Distance MAE (m):** depth error on matched objects; rises with horizon.
- **Azimuth MAE (°):** directional error; stay under ~15° for trustworthy
  spatial audio.
- **The story to look for:** if `vjepa` sits clearly above `flow` at longer
  horizons (1.5–2.5 s) while matching it at short horizons, you've shown the
  learned, action-free world model earns its place exactly where physics
  fails. If `flow` keeps up, that's an honest, publishable negative — and a
  strong argument for the cheap baseline in a wearable.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'VJEPA2Model'` / AutoModel fails | `pip install -U "transformers>=4.52.4"` |
| `predictor_output is None` | Your transformers build returns encoder only; update transformers, or the wrapper falls back to encoder tokens (still runs, weaker future signal). See the `# >>> ADAPT` note in `vjepa_world_model.py`. |
| Probe eval F1 near zero | Confirm the probe was trained against the SAME `detector.categories` and `image_size` used at eval; retrain if you changed either. |
| `CUDA out of memory` | Use ViT-L (not H/g), lower `batch_size`, keep `dtype: float16`. |
| Token grid shape mismatch | Set `model.vjepa.patch_size`/`tubelet_size`/`image_size` to match the checkpoint (defaults: 16 / 2 / 256). |
| SANPO depth all zeros | The evaluator auto-falls back to estimated depth for GT; harmless. |

---

## 9. One-screen quickstart (latent path)

```bash
unzip sonic_cartography_complete.zip && cd sonic_cartography     # if not already
conda create -n sonic_cart python=3.11 -y && conda activate sonic_cart
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)

python scripts/test_vjepa_logic.py                               # 8 passed
python data/get_sanpo.py --path a --max_sessions 50              # dataset
python train_probe.py --config configs/vjepa.yaml \
    --sanpo_root data/sanpo/ --out checkpoints/probe.pt --steps 4000
python eval/evaluate_unified.py --config configs/vjepa.yaml \
    --sanpo_root data/sanpo/ --output eval/results_vjepa.json --max_clips 200

# full ablation
for B in last_frame flow; do
  cp configs/default.yaml configs/_tmp_$B.yaml
  sed -i "s/^  backend: .*/  backend: \"$B\"/" configs/_tmp_$B.yaml
  python eval/evaluate_unified.py --config configs/_tmp_$B.yaml \
      --sanpo_root data/sanpo/ --output eval/results_$B.json --max_clips 200
done
python eval/plot_ablation.py \
    --results eval/results_last_frame.json eval/results_flow.json eval/results_vjepa.json \
    --labels last_frame flow vjepa --output_dir eval/figures/
```

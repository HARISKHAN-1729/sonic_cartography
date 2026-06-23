# Predictive Sonic Cartography — Complete Setup & Run Guide

This guide takes you from two zip files to a finished evaluation with figures.
Read **Section 0** first to pick your path — it determines how much you download.

---

## 0. Pick your path first

There are two ways to run the world-model stage. Your choice changes what you
need to download.

| | **Path FLOW** (recommended to start) | **Path NWM** (full, gated) |
|---|---|---|
| World model | Optical-flow extrapolation | Navigation World Model (CDiT) |
| GPU needed | No (CPU fine) | Yes (≥16 GB VRAM) |
| NWM repo clone | Not needed | Needed |
| NWM gated checkpoint | Not needed | Needed (you said you have access ✓) |
| Works on pedestrian SANPO | Yes, directly | Yes, but out-of-domain (approximate) |
| Setup time | ~10 min | ~40 min + checkpoint download |

**Recommendation:** get everything working on **Path FLOW** first (it's the
default in the config), confirm the full pipeline runs end to end, then switch
to **Path NWM** by changing one line in the config. You can run **both** and
report them as an ablation in the paper.

The only difference at runtime is this line in `configs/default.yaml`:

```yaml
world_model:
  backend: "flow"      # Path FLOW   (default)
  # backend: "nwm"     # Path NWM
```

---

## 1. Put the files in the right place

You have two zips:

- `sonic_cartography.zip` — the full codebase
- `nwm_fix_updated_files.zip` — the 7 corrected/new files

### 1a. Unzip the full codebase

```bash
unzip sonic_cartography.zip
cd sonic_cartography
```

### 1b. Overlay the updated files

Unzip the fix **on top of** the codebase so the 6 changed files overwrite the
originals and the 1 new file (`world_model.py`) is added:

```bash
# from inside the sonic_cartography/ directory
unzip -o ../nwm_fix_updated_files.zip -d .
```

That's it. For reference, here is exactly where each updated file belongs:

| File in the fix zip | Destination | Action |
|---|---|---|
| `models/world_model.py` | `sonic_cartography/models/world_model.py` | **NEW** |
| `models/nwm_wrapper.py` | `sonic_cartography/models/nwm_wrapper.py` | overwrite |
| `configs/default.yaml` | `sonic_cartography/configs/default.yaml` | overwrite |
| `scripts/download_assets.py` | `sonic_cartography/scripts/download_assets.py` | overwrite |
| `run_pipeline.py` | `sonic_cartography/run_pipeline.py` | overwrite |
| `eval/evaluate.py` | `sonic_cartography/eval/evaluate.py` | overwrite |
| `NWM_FIX.md` | `sonic_cartography/NWM_FIX.md` | **NEW** (reference doc) |

### 1c. Verify the layout

```bash
ls models/        # should show world_model.py AND nwm_wrapper.py
grep backend configs/default.yaml | head -1     # -> backend: "flow"
```

The final project tree should look like:

```
sonic_cartography/
├── configs/default.yaml
├── models/
│   ├── world_model.py          ← NEW (backend selector + flow predictor)
│   ├── nwm_wrapper.py          ← rewritten (faithful NWM)
│   ├── depth_estimator.py
│   └── object_detector.py
├── audio/   (earcon_bank, hrtf_renderer, scene_sonifier)
├── data/    (get_sanpo.py, sanpo_loader.py)
├── eval/    (evaluate.py, metrics.py, plot_results.py)
├── scripts/ (download_assets.py, mock_evaluate.py, test_audio_and_metrics.py)
├── run_pipeline.py
├── requirements.txt
└── *.md     (README, GPU_SETUP, NWM_FIX, SETUP — this file)
```

---

## 2. Prerequisites

| Tool | Why | Install |
|---|---|---|
| Python 3.10–3.11 | everything | conda or system |
| CUDA-capable GPU | only Path NWM, and faster depth/detection | NVIDIA driver + CUDA 12.x |
| `ffmpeg` | frame extraction from video | `sudo apt install ffmpeg` or `conda install ffmpeg` |
| `gcloud` CLI | downloading SANPO from Google Cloud | see Section 4 |
| HuggingFace account | NWM gated checkpoint (Path NWM only) | huggingface.co |

Create the environment:

```bash
conda create -n sonic_cart python=3.11 -y
conda activate sonic_cart

# GPU PyTorch (adjust cu121 to your CUDA). For CPU-only Path FLOW use the CPU wheel.
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install -r requirements.txt
```

---

## 3. What downloads automatically vs. manually

**Short answer to your question:** with your gated access and one
`huggingface-cli login`, **all the models download automatically**. The
**dataset (SANPO) is the one thing that needs a separate step** because it
lives on Google Cloud Storage, not HuggingFace — it's public (no gate) but
needs the `gcloud` CLI. Details:

| Asset | Downloaded by | Automatic? | Gated? | Lands at |
|---|---|---|---|---|
| Depth Anything V2 (Large) | `transformers` on first use | ✅ Yes | No | `~/.cache/huggingface/` |
| YOLO-World v2-X | `ultralytics` on first use | ✅ Yes | No | `./yolov8x-worldv2.pt` + ultralytics cache |
| SD VAE (for NWM only) | `diffusers` on first use | ✅ Yes | No | `~/.cache/huggingface/` |
| NWM code repo | `scripts/download_assets.py` (git clone) | ✅ Yes | No | `./nwm/` (add to PYTHONPATH) |
| **NWM weights** (Path NWM) | `scripts/download_assets.py` (HF Hub) | ✅ Yes* | **Yes** | `./checkpoints/cdit_xl_100000.pth.tar` |
| **SANPO dataset** | `data/get_sanpo.py` (gcloud) | ⚠️ Separate step | No (public) | `./data/sanpo/...` |
| Sample demo video | `scripts/download_assets.py` | ⚠️ Best-effort | No | `./data/sample.mp4` |

\* NWM weights download automatically **only after** you accept the license on
the model page (while logged in) and run `huggingface-cli login`. You said you
have gated access, so this will work once you authenticate (Section 5).

> **Note:** `scripts/download_assets.py` does **not** reliably fetch SANPO (its
> SANPO helper is a placeholder). Always use `data/get_sanpo.py` for the
> dataset — Section 4.

---

## 4. Get the SANPO dataset

SANPO is on a public Google Cloud bucket (`gs://gresearch/sanpo_dataset/v0`).
Install the `gcloud` CLI first:

```bash
# Linux/macOS quick install
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init        # you can skip login; the bucket is public
```

Then pick one of three sub-options.

### Option A — SANPO-Synthetic (recommended for the paper)

Pixel-perfect depth + panoptic masks, smaller, cleanest ground truth.

```bash
# ~10 GB for the full synthetic test split; or cap it:
python data/get_sanpo.py --path a --max_sessions 50      # ~2.5 GB, fast
# python data/get_sanpo.py --path a                       # full test split
```

### Option B — SANPO-Real (real pedestrian footage)

```bash
python data/get_sanpo.py --path b --max_sessions 10      # ~28 GB
```

### Option C — Your own phone video (zero dataset download)

```bash
python data/get_sanpo.py --path c --input_dir ~/my_walking_videos/
```

All three write into `./data/sanpo/` in the layout the loader expects.

### If `gcloud` is unavailable — fully manual SANPO

1. On any machine with a browser, download the test sessions from the bucket
   page: `https://github.com/google-research-datasets/sanpo_dataset`
   (the README links to the GCS bucket browser).
2. Copy them to your GPU machine under:
   ```
   sonic_cartography/data/sanpo/sanpo-synthetic/sessions/<session_id>/
       rgb/      ← frame_000000.jpg ...
       depth/    ← frame_000000.npz (or .png) ...
       metadata.json
   ```
3. Add a split list at `data/sanpo/sanpo-synthetic/splits/test.txt`
   (one session id per line), or just point `--sanpo_root` at the folder and
   the loader will scan it.

---

## 5. Path-specific setup

### Path FLOW (default — do this first)

Nothing extra to download for the world model. Just fetch the two ungated
models and (optionally) a sample video:

```bash
# downloads Depth Anything V2 + YOLO-World; skips NWM checkpoint
python scripts/download_assets.py --no_sanpo
# (it will still try to git-clone NWM; that's harmless. To skip entirely:)
# python scripts/download_assets.py --no_sanpo --no_models   # then download depth/yolo on first run
```

Confirm `configs/default.yaml` has `backend: "flow"` (it does by default).
Done — skip to Section 6.

### Path NWM (gated, full)

1. **Accept the license** at `https://huggingface.co/facebook/nwm` while
   logged in to your HuggingFace account (you have access).

2. **Authenticate** so the gated download works:
   ```bash
   huggingface-cli login        # paste a token from hf.co/settings/tokens
   ```

3. **Download the repo + checkpoint** (automatic):
   ```bash
   python scripts/download_assets.py --no_sanpo
   ```
   This clones `./nwm/` and downloads `./checkpoints/cdit_xl_100000.pth.tar`.
   For the best pedestrian match, grab the Ego4D variant instead by editing
   `NWM_CKPTS` selection or downloading it directly:
   ```bash
   python -c "from huggingface_hub import hf_hub_download as d; \
   d(repo_id='facebook/nwm', filename='cdit_xl_ego4d_200000.pth.tar', local_dir='checkpoints')"
   ```

4. **Expose the repo** on PYTHONPATH every session (the wrapper imports
   `models.py` and `diffusion/` from it):
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/nwm
   ```

5. **Point the config** at the checkpoint and switch the backend:
   ```yaml
   # configs/default.yaml
   world_model:
     backend: "nwm"
   model:
     nwm:
       model_size: "XL"
       checkpoint_path: "checkpoints/cdit_xl_100000.pth.tar"   # or the ego4d file
       vae: "stabilityai/sd-vae-ft-ema"
   ```

6. **One-time adaptation check.** Open `models/nwm_wrapper.py` and verify the
   three spots marked `# >>> ADAPT` against the freshly cloned `nwm/models.py`
   and `nwm/diffusion/`:
   - `_build_cdit()` — the `CDiT_models[...]` key + constructor kwargs
   - `_make_diffusion()` — the `create_diffusion(...)` import
   - `_cdit_forward()` — the argument order of `CDiT.forward(...)`
   They follow the standard DiT/CDiT convention, but pin them once. If
   anything mismatches you'll see a shape/keyword error on the first
   prediction — fix it in that one method.

---

## 6. Run it

### 6a. Sanity checks (no GPU, no downloads — do these first)

```bash
# Unit tests for audio + metrics (should print 16 passed)
python scripts/test_audio_and_metrics.py

# Full pipeline logic on synthetic data (validates the eval loop end to end)
python scripts/mock_evaluate.py --n_clips 50 --plot
```

If both pass, your install is healthy regardless of models/data.

### 6b. Single-video demo (produces a binaural .wav + a visualization)

```bash
python run_pipeline.py \
    --video data/sample.mp4 \
    --output_dir out/ \
    --horizon 1.5
# all horizons at once:
python run_pipeline.py --video data/sample.mp4 --output_dir out/ --all_horizons
```

Outputs land in `out/`: `*_h1.5s.wav` (the predictive soundscape) and
`*_h1.5s_vis.jpg` (context | predicted frame + detections | depth map).

### 6c. Full offline evaluation on SANPO

```bash
python eval/evaluate.py \
    --sanpo_root data/sanpo/ \
    --output eval/results.json \
    --max_clips 500
```

Prints the accuracy-vs-lead-time table and writes `eval/results.json`.

### 6d. Generate the paper figures

```bash
python eval/plot_results.py \
    --results eval/results.json \
    --output_dir eval/figures/
```

Produces `fig1_accuracy_vs_lead_time.pdf/png`, `fig2_per_class_mae.*`, and
`fig3_azimuth_mae.*`.

### 6e. The ablation (strongest result for the abstract)

Run the same evaluation under each backend and compare. The clean way is to
copy the config and flip `backend`:

```bash
for B in last_frame flow nwm; do
  cp configs/default.yaml configs/_tmp.yaml
  sed -i "s/^  backend: .*/  backend: \"$B\"/" configs/_tmp.yaml
  python eval/evaluate.py --config configs/_tmp.yaml \
      --sanpo_root data/sanpo/ --output eval/results_$B.json --max_clips 500
done
```

`last_frame` is the trivial lower bound, `flow` the constant-velocity
baseline, `nwm` the learned world model. Plot all three F1-vs-horizon curves
on one axis to show how much the world model adds.

> If you only have Path FLOW set up, just run the `last_frame` and `flow`
> rows — that's already a meaningful ablation.

---

## 7. Expected runtimes (single RTX 4090, fp16)

| Step | Path FLOW | Path NWM |
|---|---|---|
| Per future frame (predict) | ~0.01 s (CPU) | ~0.8 s (diffusion) |
| Depth + detection per frame | ~0.08 s | ~0.08 s |
| 500-clip × 5-horizon eval | ~20–40 min | a few hours |

For a quick first eval, use `--max_clips 20`.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: No module named 'models'` (from NWM) | `export PYTHONPATH=$PYTHONPATH:$(pwd)/nwm` |
| NWM download 403 / gated error | Accept license at hf.co/facebook/nwm, then `huggingface-cli login` |
| `CUDA out of memory` (Path NWM) | Set `model.nwm.model_size: "L"` or `num_sampling_steps: 50`; use fp16 |
| Shape/keyword error on first NWM predict | Align the `# >>> ADAPT` spots in `nwm_wrapper.py` with `nwm/models.py` |
| `gcloud: command not found` | Install gcloud (Section 4) or use the manual SANPO option |
| SANPO depth all zeros for some clips | The loader auto-falls back to estimated depth; harmless |
| `ffmpeg not found` (Path C / demo) | `sudo apt install ffmpeg` (OpenCV fallback also works) |
| Want zero risk / fastest path | Stay on `backend: "flow"`; it needs none of NWM |

---

## 9. One-screen quickstart (Path FLOW)

```bash
unzip sonic_cartography.zip && cd sonic_cartography
unzip -o ../nwm_fix_updated_files.zip -d .
conda create -n sonic_cart python=3.11 -y && conda activate sonic_cart
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/test_audio_and_metrics.py          # 16 passed
python data/get_sanpo.py --path a --max_sessions 50
python scripts/download_assets.py --no_sanpo --no_models   # depth/yolo pull on first run
python eval/evaluate.py --sanpo_root data/sanpo/ --output eval/results.json --max_clips 50
python eval/plot_results.py --results eval/results.json --output_dir eval/figures/
```

Switch to Path NWM later by doing Section 5 (Path NWM) and flipping
`backend: "nwm"`.

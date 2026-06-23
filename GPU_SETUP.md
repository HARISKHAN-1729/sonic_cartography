# GPU Setup Guide
## Predictive Sonic Cartography — single-GPU deployment

---

## VRAM requirements

| Model                      | VRAM (fp16) | VRAM (fp32) |
|----------------------------|-------------|-------------|
| NWM (outdoor_v1)           | ~6 GB       | ~12 GB      |
| Depth Anything V2 ViT-L    | ~3 GB       | ~6 GB       |
| YOLO-World v2 X            | ~2 GB       | ~4 GB       |
| **Total (concurrent)**     | **~11 GB**  | **~22 GB**  |

**RTX 4090 (24 GB)** — runs all three in fp16 with headroom.  
**RTX 3090 / A5000 (24 GB)** — same.  
**RTX 4080 (16 GB)** — runs fine in fp16; use `vitb` encoder for depth instead of `vitl`.  
**RTX 3080 (10 GB)** — use `vits` depth encoder + run models sequentially (not concurrent).

---

## Step-by-step setup

```bash
# 1. Clone this repo
git clone https://github.com/[your-repo]/sonic_cartography
cd sonic_cartography

# 2. Create conda environment
conda create -n sonic_cartography python=3.11 -y
conda activate sonic_cartography

# 3. Install CUDA-enabled PyTorch (adjust cu121 for your CUDA version)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Clone and install the NWM repo
git clone https://github.com/facebookresearch/nwm
pip install -e nwm/

# 6. Download pretrained checkpoints + sample data
python scripts/download_assets.py

# 7. Run CPU-only tests (validates audio + metrics, no GPU needed)
python scripts/test_audio_and_metrics.py

# 8. Run mock evaluation (validates full eval loop, no GPU needed)
python scripts/mock_evaluate.py --n_clips 50 --plot

# 9. Run real demo on sample video
python run_pipeline.py --video data/sample.mp4 --output_dir out/

# 10. Run full offline evaluation on SANPO
python eval/evaluate.py \
    --sanpo_root data/sanpo/ \
    --output eval/results.json \
    --max_clips 500

# 11. Generate final paper figures
python eval/plot_results.py \
    --results eval/results.json \
    --output_dir eval/figures/
```

---

## Memory optimisation tips

### If you're tight on VRAM

```python
# In configs/default.yaml, change:
model:
  depth:
    encoder: "vits"          # was "vitl" — saves ~2GB, small accuracy drop
  nwm:
    dtype: "float16"         # already default; double-check
  detector:
    checkpoint: "yolov8l-worldv2.pt"  # was yolov8x — saves ~0.5GB
```

### Sequential model execution (10 GB GPU)
Add `--sequential` flag to `run_pipeline.py` (loads/unloads each model in turn):
```bash
python run_pipeline.py --video data/sample.mp4 --sequential
```

### torch.compile for faster inference
```python
# In nwm_wrapper.py, after loading model:
self.model = torch.compile(self.model, mode="reduce-overhead")
```
Adds ~30s warmup on first run; reduces per-frame latency ~25% on RTX 4090.

---

## Expected runtimes (RTX 4090, fp16)

| Stage                          | Time per frame |
|--------------------------------|---------------|
| NWM rollout (12 future frames) | ~0.8s         |
| Depth Anything V2 ViT-L        | ~0.05s        |
| YOLO-World inference           | ~0.03s        |
| Audio synthesis + HRTF         | ~0.002s (CPU) |
| **Total per inference step**   | **~0.9s**     |

At 1 inference/second, the system provides continuous 1.5s-ahead previews
at a comfortable update rate for real-time pedestrian navigation.

---

## Troubleshooting

**`ImportError: No module named 'nwm'`**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/nwm
# Or add to your .bashrc / conda env activation script
```

**`CUDA out of memory`**
```python
# Add to run_pipeline.py before model loading:
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

**`NWM checkpoint not found`**
The NWM HuggingFace repo ID may have changed. Check:
https://github.com/facebookresearch/nwm for the latest checkpoint name
and update `configs/default.yaml` → `model.nwm.checkpoint`.

**`SANPO depth maps all zeros`**
Some SANPO clips have missing depth; the pipeline falls back to estimated
depth from Depth Anything V2 automatically. Set `--verbose` for per-clip logs.

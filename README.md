# Predictive Sonic Cartography
### Navigation World Models for Visually Impaired Auditory Navigation

End-to-end pipeline: egocentric video → future frame prediction → scene parsing → binaural audio.

---

## Hardware requirements
- GPU: single RTX 4090 or A100 (24GB VRAM recommended; 16GB works with fp16)
- CPU: any modern multi-core
- RAM: 32GB recommended

## Install

```bash
git clone https://github.com/facebookresearch/nwm   # Navigation World Models
pip install -r requirements.txt
```

## Quick start

```bash
# Run demo on a single video
python run_pipeline.py --video data/sample.mp4 --output_dir out/ --horizon 2.0

# Run full offline evaluation on SANPO dataset
python eval/evaluate.py --sanpo_root data/sanpo/ --output eval/results.json

# Plot accuracy-vs-lead-time curve
python eval/plot_results.py --results eval/results.json
```

## Project structure

```
sonic_cartography/
├── models/
│   ├── nwm_wrapper.py          # Navigation World Model wrapper
│   ├── depth_estimator.py      # Depth Anything V2 wrapper
│   └── object_detector.py      # YOLO-World wrapper
├── audio/
│   ├── earcon_bank.py          # Earcon definitions per object class
│   ├── hrtf_renderer.py        # Binaural HRTF renderer
│   └── scene_sonifier.py       # Spatial layout → audio scene
├── eval/
│   ├── evaluate.py             # Main offline evaluation script
│   ├── metrics.py              # Object hit rate, distance MAE, etc.
│   └── plot_results.py         # Accuracy vs. lead-time curve
├── data/
│   └── sanpo_loader.py         # SANPO dataset loader
├── configs/
│   └── default.yaml            # All hyperparameters
├── run_pipeline.py             # End-to-end demo (video → audio)
└── requirements.txt
```

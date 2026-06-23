"""
scripts/download_assets.py
───────────────────────────
Downloads all required pretrained checkpoints and sample data.

What this script downloads:
  1. NWM pretrained checkpoint (HuggingFace Hub)
  2. Depth Anything V2 Large (HuggingFace Hub — auto via transformers)
  3. YOLO-World v2 x (ultralytics auto-download)
  4. SANPO dataset subset (Google Drive via gdown — outdoor test split)
  5. A sample video for demo mode

Usage:
    python scripts/download_assets.py [--no_sanpo] [--no_models]
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list, cwd=None):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


# NWM checkpoint files on HuggingFace facebook/nwm (all GATED).
# Pick ONE. For pedestrian / human-egocentric scenes, the XL+Ego4D weights
# are the closest domain match.
NWM_CKPTS = {
    "S":        "cdit_s_100000.pth.tar",
    "B":        "cdit_b_100000.pth.tar",
    "L":        "cdit_l_100000.pth.tar",
    "XL":       "cdit_xl_100000.pth.tar",
    "XL_EGO4D": "cdit_xl_ego4d_200000.pth.tar",
}


def download_nwm(model_size: str = "XL"):
    """
    Set up the Navigation World Model.

    IMPORTANT — two facts about the real release:
      1. github.com/facebookresearch/nwm provides the CODE (models.py,
         diffusion/) — there is NO pip package and NO `from_pretrained()`.
      2. huggingface.co/facebook/nwm provides only GATED `.pth.tar`
         STATE-DICTS. You must (a) accept the license on the model page while
         logged in, and (b) authenticate (`huggingface-cli login`) before the
         download will succeed.

    This function clones the repo and downloads one gated checkpoint into
    ./checkpoints/. Point cfg.model.nwm.checkpoint_path at the result.

    If you'd rather not deal with the gate at all, skip this entirely and use
    world_model.backend == "flow" (no checkpoint required).
    """
    print("\n── Navigation World Model ─────────────────────────────────────")

    # 1) Clone the code repo (needed for models.py + diffusion/).
    nwm_dir = ROOT / "nwm"
    if not nwm_dir.exists():
        run(["git", "clone", "https://github.com/facebookresearch/nwm", str(nwm_dir)])
    else:
        print("  NWM repo already present.")
    print(f"  Add it to PYTHONPATH before running NWM inference:")
    print(f"    export PYTHONPATH=$PYTHONPATH:{nwm_dir}")

    # 2) Download ONE gated checkpoint via the HF Hub (requires auth).
    ckpt_file = NWM_CKPTS.get(model_size.upper(), NWM_CKPTS["XL"])
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    dest = ckpt_dir / ckpt_file

    if dest.exists():
        print(f"  Checkpoint already present: {dest}")
        return

    print(f"  Downloading gated checkpoint: {ckpt_file}")
    print("  (Requires: accept license at https://huggingface.co/facebook/nwm")
    print("   while logged in, then `huggingface-cli login` with a token.)")
    py_cmd = (
        "from huggingface_hub import hf_hub_download; "
        f"p = hf_hub_download(repo_id='facebook/nwm', filename='{ckpt_file}', "
        f"local_dir='{ckpt_dir}'); print('saved', p)"
    )
    try:
        run([sys.executable, "-c", py_cmd])
        print(f"  ✓ NWM checkpoint saved → {dest}")
        print(f"  Set in configs/default.yaml:")
        print(f"    model.nwm.checkpoint_path: checkpoints/{ckpt_file}")
        print(f"    world_model.backend: nwm")
    except Exception as e:
        print(f"  [WARN] Gated download failed ({e}).")
        print("  Steps to fix:")
        print("    1. Visit https://huggingface.co/facebook/nwm and accept the license")
        print("    2. Run: huggingface-cli login   (paste a token from hf.co/settings/tokens)")
        print(f"    3. Re-run: python scripts/download_assets.py")
        print("  Or skip NWM entirely and use world_model.backend == 'flow'.")


def download_depth_anything():
    """Depth Anything V2 is downloaded automatically by transformers pipeline."""
    print("\n── Depth Anything V2 ──────────────────────────────────────────")
    print("  Downloading Depth Anything V2 Large via transformers pipeline...")
    py_cmd = (
        "from transformers import pipeline as P; "
        "P(task='depth-estimation', "
        "model='depth-anything/Depth-Anything-V2-Large-hf', "
        "device=-1)"
    )
    run([sys.executable, "-c", py_cmd])
    print("  ✓ Depth Anything V2 cached in ~/.cache/huggingface/")


def download_yolo_world():
    """YOLO-World downloads on first .predict() call via ultralytics."""
    print("\n── YOLO-World ─────────────────────────────────────────────────")
    print("  Pre-downloading YOLO-World v2 x...")
    py_cmd = (
        "from ultralytics import YOLOWorld; "
        "m = YOLOWorld('yolov8x-worldv2.pt'); "
        "print('  ✓ YOLO-World ready')"
    )
    run([sys.executable, "-c", py_cmd])


def download_sanpo(subset_only: bool = True):
    """
    Download SANPO dataset.
    Full dataset is ~200GB; we offer a small subset (~2GB) for testing.
    """
    print("\n── SANPO Dataset ──────────────────────────────────────────────")
    data_dir = ROOT / "data" / "sanpo"
    data_dir.mkdir(parents=True, exist_ok=True)

    if subset_only:
        # Small curated subset hosted by the SANPO authors (outdoor test clips)
        # See: https://github.com/google-research-datasets/sanpo-dataset
        print("  Downloading SANPO outdoor test subset (~2GB)...")
        print("  (For full dataset, follow instructions at:")
        print("   https://github.com/google-research-datasets/sanpo-dataset)")
        # Placeholder: replace with actual GDrive ID from dataset page
        gdrive_id = "SANPO_OUTDOOR_TEST_SUBSET_GDRIVE_ID"
        try:
            import gdown
            gdown.download(
                id=gdrive_id,
                output=str(data_dir / "sanpo_outdoor_test.zip"),
                quiet=False,
            )
            import zipfile
            with zipfile.ZipFile(data_dir / "sanpo_outdoor_test.zip") as z:
                z.extractall(data_dir)
            print(f"  ✓ SANPO subset extracted → {data_dir}/")
        except Exception as e:
            print(f"  [WARN] Auto-download failed ({e}).")
            print("  Please manually download SANPO from:")
            print("    https://github.com/google-research-datasets/sanpo-dataset")
            print(f"  and extract to: {data_dir}/")
    else:
        print("  For full SANPO dataset, follow instructions at:")
        print("    https://github.com/google-research-datasets/sanpo-dataset")


def download_sample_video():
    """Download a short public-domain egocentric walking video for demo."""
    print("\n── Sample video ────────────────────────────────────────────────")
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = data_dir / "sample.mp4"

    if sample_path.exists():
        print(f"  Sample video already present: {sample_path}")
        return

    # Download a short CC0 walking video from Wikimedia Commons
    url = (
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/"
        "5/50/Walking_man.webm/Walking_man.webm.360p.webm"
    )
    try:
        import urllib.request
        print(f"  Downloading sample video...")
        urllib.request.urlretrieve(url, str(sample_path))
        print(f"  ✓ Saved → {sample_path}")
    except Exception as e:
        print(f"  [WARN] Could not download sample video: {e}")
        print(f"  Place any walking video at: {sample_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no_sanpo",  action="store_true",
                   help="Skip SANPO download (for quick model-only setup)")
    p.add_argument("--no_models", action="store_true",
                   help="Skip model downloads")
    p.add_argument("--full_sanpo", action="store_true",
                   help="Print full SANPO download instructions")
    args = p.parse_args()

    print("Predictive Sonic Cartography — asset downloader")
    print("=" * 55)

    if not args.no_models:
        download_nwm()
        download_depth_anything()
        download_yolo_world()

    if not args.no_sanpo:
        download_sanpo(subset_only=not args.full_sanpo)

    download_sample_video()

    print("\n" + "=" * 55)
    print("Setup complete. Run the demo with:")
    print("  python run_pipeline.py --video data/sample.mp4 --output_dir out/")
    print("\nRun evaluation with:")
    print("  python eval/evaluate.py --sanpo_root data/sanpo/ --output eval/results.json")


if __name__ == "__main__":
    main()

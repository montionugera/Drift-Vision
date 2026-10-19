# Drift-Vision

Computer vision for dashcam / live car camera footage. Current goal: recognise road lines,
turns, road signs and the general environment. Long-term goal: predict the speed and angle
needed to drift an upcoming turn.

## Setup

Requires Python 3.11+.

```bash
# Ubuntu/WSL only, first time: make sure venv is available
sudo apt install python3-venv

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies (listed in pyproject.toml)
# Includes the GPU version of PyTorch (~5 GB) for NVIDIA cards
pip install -e ".[dev]"

# Check that PyTorch can see your GPU (should print True)
python -c "import torch; print(torch.cuda.is_available())"
```

No NVIDIA GPU? Install the much smaller CPU-only PyTorch (~200 MB) **before** the step above:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If pip fails with `No space left on device`, your `/tmp` is too small (common on WSL).
Point pip's temporary files at your main disk instead:

```bash
mkdir -p ~/.pip-tmp
TMPDIR=~/.pip-tmp pip install -e ".[dev]"
rm -rf ~/.pip-tmp
```

Run `source .venv/bin/activate` each time you open a new terminal.

## Project layout

```
main.py                     entry point
configs/default.yaml        settings
src/drift_vision/
  io/video_source.py        read dashcam file or live camera
  perception/lanes.py       road line detection
  perception/objects.py     signs, cars, pedestrians (YOLO)
  perception/scene.py       general environment segmentation
  geometry/turns.py         detect upcoming turns from lane curvature
  viz/overlay.py            draw results on frames
  pipeline.py               runs everything frame by frame
data/                       footage (not committed)
models/                     trained weights (not committed)
notebooks/                  experiments
scripts/                    helper scripts
tests/                      tests
```

## Adding a dependency

Add it to the `dependencies` list in `pyproject.toml`, then re-run `pip install -e ".[dev]"`.

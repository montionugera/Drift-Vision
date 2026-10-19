# Drift-Vision

Computer vision for dashcam / live car camera footage. Current goal: recognise road lines,
turns, road signs and the general environment. Long-term goal: predict the speed and angle
needed to drift an upcoming turn.

## Setup

Requires Python 3.11+ and git. The install is ~6 GB (mostly GPU PyTorch) and only needs to
be done once per computer.

### Linux / WSL (Ubuntu)

```bash
# First time only: make sure venv is available
sudo apt install python3-venv

# Get the code
git clone https://github.com/VOCTOV/Drift-Vision.git
cd Drift-Vision

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies (listed in pyproject.toml)
# On Linux this includes the GPU version of PyTorch for NVIDIA cards
pip install -e ".[dev]"

# Check that PyTorch can see your GPU (should print True)
python -c "import torch; print(torch.cuda.is_available())"
```

If pip fails with `No space left on device`, your `/tmp` is too small (common on WSL).
Point pip's temporary files at your main disk instead:

```bash
mkdir -p ~/.pip-tmp
TMPDIR=~/.pip-tmp pip install -e ".[dev]"
rm -rf ~/.pip-tmp
```

### Windows (PowerShell)

Install Python from [python.org](https://www.python.org/downloads/) and tick
**"Add python.exe to PATH"** during setup.

```powershell
# Get the code
git clone https://github.com/VOCTOV/Drift-Vision.git
cd Drift-Vision

# Create and activate a virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1

# On Windows, plain pip installs CPU-only PyTorch, so install the GPU version first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install all other dependencies (listed in pyproject.toml)
pip install -e ".[dev]"

# Check that PyTorch can see your GPU (should print True)
python -c "import torch; print(torch.cuda.is_available())"
```

If activating fails with "running scripts is disabled on this system", run this once, then
try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Using Command Prompt instead of PowerShell? Activate with `.venv\Scripts\activate.bat`.

### No NVIDIA GPU?

Install the much smaller CPU-only PyTorch (~200 MB) **before** `pip install -e ".[dev]"`
(on Windows, use this instead of the `cu130` line):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Every time you open a new terminal

Activate the virtual environment first:

- Linux / WSL: `source .venv/bin/activate`
- Windows: `.venv\Scripts\Activate.ps1`

## Project layout

```
Drift-Vision/
├── main.py                    entry point: loads the config and runs the pipeline
├── pyproject.toml             project info + dependencies (like package.json)
├── configs/
│   └── default.yaml           settings: video source, thresholds, model choice
├── src/drift_vision/          the main Python package
│   ├── pipeline.py            runs every stage frame by frame
│   ├── io/
│   │   └── video_source.py    reads a dashcam file or live camera
│   ├── perception/            understanding what is in each frame
│   │   ├── lanes.py           road line detection
│   │   ├── objects.py         road signs, cars, pedestrians, traffic lights (YOLO)
│   │   └── scene.py           environment: road, sidewalk, sky, buildings
│   ├── geometry/
│   │   └── turns.py           detects upcoming turns from lane curvature
│   └── viz/
│       └── overlay.py         draws results on top of each frame
├── data/                      footage (not committed to git)
│   ├── samples/               short test clips
│   ├── raw/                   full original footage / datasets
│   └── processed/             annotated output videos
├── models/                    trained model weights (not committed to git)
├── notebooks/                 Jupyter experiments
├── scripts/                   helper scripts (e.g. downloading datasets)
└── tests/                     tests, run with `pytest`
```

How a frame flows through the pipeline:

```
video_source → lanes / objects / scene → turns → overlay → screen or saved video
```

## Adding a dependency

Add it to the `dependencies` list in `pyproject.toml`, then re-run `pip install -e ".[dev]"`.

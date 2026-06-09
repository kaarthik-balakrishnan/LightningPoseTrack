# Project Context for AI Coding Agents

## Workflow Requirement
**Before making any changes, ALWAYS:**
1. `git pull` from GitHub to get the latest
2. Make edits
3. `git push` to GitHub after changes are complete

The user runs notebooks on Google Colab which clones/pulls from GitHub. If code is not pushed, Colab will run stale versions.

## Project Summary
End-to-end pig behavior analysis pipeline using Lightning Pose for pose estimation, running entirely on Google Colab (GPU) with data on Google Drive and code on GitHub.

- **GitHub**: https://github.com/kaarthik-balakrishnan/LightningPoseTrack
- **Drive folder ID**: `1X_41ZW3HfwVeft2lPld3XNqXsdxRDIwb` (shared, contains raw videos)
- **Raw videos**: Organized by session folders, camera encoded in filename as `_N.asf` (N=1-4, hyphen-delimited, e.g. `180422-3.asf`)
- **Docker Hub**: `kaarthikbalakrishnan/lightningposetrack:latest` — GPU image auto-built by GH Action on push to main

## Pipeline (9 notebooks)
| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Data Audit | Scan videos, parse metadata, inventory report |
| 02 | Frame Sampling | Sample frames with animal present (background subtraction) |
| 02b | Cross-Camera Calibration | Probe, timeline, motion sync, unified frame index |
| 03 | Pose Training | Train Lightning Pose model (GPU) |
| 04 | Pose Inference | Run model on all videos |
| 05 | Kinematics | Speed/acceleration per keypoint |
| 06 | Orientation | Heading, angular velocity, turning rate |
| 07 | Behavior Detection | Feeding detection (snout + feeder + duration rules) |
| 08 | Reporting | HTML/PDF daily summary reports |

## Keypoint Definitions (8 total)
1. snout
2. left_ear
3. right_ear
4. neck
5. shoulders
6. mid_back
7. hip
8. tail_base

Bilateral keypoints (left_ear, right_ear) are marked at the midline midpoint when both would be visible.

## Key Decisions
- **All cameras trained together**: Labeled frames from all cameras pooled for a single Lightning Pose model
- **Camera label is metadata only**: Not used during training; extracted from filename for filtering/reporting
- **LabelMe for offline labeling**: JSON files converted to Lightning Pose CSV via `src/pose/label_converter.py`
- **Hyphen-delimited camera parsing**: Parser splits on both hyphens and underscores (`re.split(r"[-_]+", stem)`)
- **Background subtraction for frame sampling**: In notebook 02, median background is computed per video to detect animal presence before saving frames
- **LabelMe labels**: No group ID needed; just the exact keypoint name string per point annotation
- **Docker for local dev only**: CPU Dockerfile for VS Code editing; GPU training stays on Colab

## Dev Container Setup

### Files
| File | Purpose |
|------|---------|
| `.devcontainer/Dockerfile` | CPU-only, for local VS Code dev (Python 3.12, PyTorch CPU, Jupyter) |
| `.devcontainer/Dockerfile.gpu` | GPU image based on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` |
| `.devcontainer/devcontainer.json` | VS Code config — Python/Jupyter extensions, port forwarding |
| `.devcontainer/build_on_colab.ipynb` | Colab launchpad — installs everything + opencode, has notebook runner |
| `.devcontainer/README.md` | Documentation for all workflows |
| `.github/workflows/docker-gpu.yml` | GH Action: on push to main, builds GPU image, pushes to Docker Hub |
| `dev/colab_runner.py` | Colab notebook runner — executes notebooks headlessly, captures errors |

### Three Workflows

**1. Local Dev (CPU)** — VS Code Dev Containers: `Ctrl+Shift+P → "Reopen in Container"`
- Edit code, lint, test non-GPU modules
- No GPU — training stays on Colab

**2. Colab Direct (recommended)** — Open `build_on_colab.ipynb` on Colab with GPU runtime, Run All
- Installs everything directly on Colab VM (no Docker)
- Sets up Drive mount, repo clone, packages, opencode CLI

**3. Colab Docker (experimental)** — Same notebook, set `MODE = "docker"`
- Most Colab runtimes don't support nested Docker
- Falls back to direct mode automatically

### The Fix Loop (opencode + Colab)
This is the core pattern for debugging notebooks:

1. On Colab: open `build_on_colab.ipynb` → Run All
2. Change `NOTEBOOK_NUM = "01"` (or any number 01-08) and run the runner cell
3. `dev/colab_runner.py` executes the notebook via `jupyter nbconvert --execute`
4. On success → move to next notebook number
5. On failure → full traceback saved to `/tmp/colab_error.txt`
6. User copies the error from the display cell and pastes it to opencode
7. opencode fixes the code → `git add/commit/push`
8. User re-runs the same runner cell — it auto-runs `git pull` before executing
9. Loop repeats until notebook passes

### Known Issues & Fixes

#### Notebook 03: `ConfigAttributeError: Missing key dali`
- **Cause**: `losses_to_use: ["pca_singleview"]` makes Lightning Pose think it's semi-supervised, which requires a `dali` config section
- **Fix**: Set `losses_to_use: []` for purely supervised training (no unlabeled data needed)
- Also add `train_frames: null` to training config
- Already applied to `notebooks/03_Pose_Training.ipynb`

#### Docker on Colab
- **Cause**: Colab runs inside its own container — nested Docker requires privileged mode
- **Fix**: Use direct install mode (`MODE = "direct"`) instead. Docker image is for local dev and GPU cloud servers, not Colab.

#### GitHub Actions Node.js 20 deprecation
- Warning about actions using Node.js 20 (will stop working Sept 2026)
- Fix: update action versions or set `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true`

## Directory Structure
```
LightningPoseTrack/
├── .devcontainer/       # Dev Container config (Dockerfiles, VS Code, Colab runner notebook)
│   ├── Dockerfile       # CPU local dev
│   ├── Dockerfile.gpu   # GPU for cloud
│   ├── devcontainer.json
│   ├── build_on_colab.ipynb
│   └── README.md
├── .github/workflows/
│   └── docker-gpu.yml   # Auto-build GPU image on push to main
├── dev/
│   └── colab_runner.py  # Headless notebook executor for the fix loop
├── notebooks/           # 9 Colab notebooks (01, 02, 02b, 03-08)
├── src/
│   ├── calibration/
│   │   └── pipeline.py         # Cross-camera calibration pipeline
│   ├── io/
│   │   └── video_inventory.py   # scan_videos(), parse_camera_from_filename()
│   ├── pose/
│   │   ├── frame_sampler.py     # compute_background(), detect_animal(), sample_*()
│   │   ├── label_converter.py   # LabelMe JSON → LP CSV conversion
│   │   └── clean_pose.py        # Confidence filtering, interpolation, smoothing
│   ├── features/
│   │   ├── kinematics.py        # Speed/acceleration per keypoint
│   │   ├── orientation.py       # Heading, angular velocity
│   │   └── spatial.py           # Distances to feeder/walls, zone occupancy
│   ├── behaviors/
│   │   └── feeding.py           # Rule-based feeding detection
│   └── reports/
│       └── daily_report.py      # HTML/PDF report generation
├── requirements-colab.txt
├── setup_colab.sh
├── .gitignore
└── AGENTS.md
```

## Progress
- Git repo initialized and pushed with all 8 notebooks and source modules
- Parser fixed to handle hyphen-delimited camera numbers (`180422-3.asf` → camera=3)
- Label converter created at `src/pose/label_converter.py`
- Background subtraction functions added to `src/pose/frame_sampler.py`
- Notebook 02 rewritten to use background subtraction for animal detection
- Notebook 03 includes auto-converter cell (LabelMe JSON → LP CSV)
- Notebook 03 fixed: `losses_to_use: []` + `train_frames: null` (dali key error)
- Dev Container setup: CPU Dockerfile, GPU Dockerfile, devcontainer.json, GH Action, colab_runner.py
- Calibration pipeline (`src/calibration/pipeline.py`) with 9-stage workflow
- Notebook 02b (Cross-Camera Calibration) with dynamic tick intervals, read-only FS fallback, and custom time-range filtering
- TODO: Label frames, run notebooks 03-08

## Labeling Instructions
```bash
# On Mac (offline)
python3 -m venv ~/labeling_env
source ~/labeling_env/bin/activate
pip install labelme
labelme /path/to/labeled_frames
```
- Create POINT annotations with the exact 8 keypoint names above
- No group ID needed
- Save to `labeled_frames/` on Drive at `My Drive/PigBehavior/labeled_frames/`

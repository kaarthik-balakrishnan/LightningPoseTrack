# Dev Container for LightningPoseTrack

Two environments are provided:

| Container | File | Use Case | GPU |
|-----------|------|----------|-----|
| **CPU** | `Dockerfile` | Local dev on Mac/Windows (VS Code) | ❌ |
| **GPU** | `Dockerfile.gpu` | Training/inference on Colab or GPU cluster | ✅ |

---

## Workflow A: Local Development (CPU)

Open the repo in VS Code with the **Dev Containers** extension installed:

```bash
# Option 1: Command Palette
Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# Option 2: Terminal
code /path/to/LightningPoseTrack
# Click "Reopen in Container" when prompted
```

This builds `Dockerfile` (CPU) and mounts the workspace at `/workspace`. Use for:
- Editing Python modules in `src/`
- Running linters (pylint, autopep8)
- Testing non-GPU code
- Authoring notebooks (Jupyter runs inside the container on ports 8888/8899)

> **Note**: The CPU container cannot run `lightning-pose` training. GPU training happens on Colab.

---

## Workflow B: GPU Training on Colab

### Prerequisites

1. Docker Hub token added as GitHub secret `DOCKER_HUB_TOKEN` at:
   `github.com/kaarthik-balakrishnan/LightningPoseTrack/settings/secrets/actions`

2. Every push to `main` auto-builds the GPU image and pushes to:
   `kaarthikbalakrishnan/lightningposetrack:latest`

### Colab Steps

| Step | Action |
|------|--------|
| 1 | Open `colab.research.google.com` with **GPU runtime** (T4/V100/A100) |
| 2 | File → Open notebook → GitHub → select `LightningPoseTrack/.devcontainer/build_on_colab.ipynb` |
| 3 | **Runtime → Run all** |
| 4 | The notebook installs Docker, pulls the GPU image, and launches Jupyter inside the container |

### Running Notebooks 03-08

Once the container is running, you have two options:

**Option A — Inside the Docker container (recommended for reproducibility):**
```bash
docker exec -it lightningposetrack bash
cd /workspace
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Option B — Standard Colab workflow (no Docker):**
Just open the notebooks directly from GitHub in Colab — they already handle all dependencies via `pip install` cells.

---

## GitHub Actions Pipeline

`.github/workflows/docker-gpu.yml`:

| Trigger | Action |
|---------|--------|
| Push to `main` (excluding docs) | Build `Dockerfile.gpu`, push to Docker Hub |
| Manual via `workflow_dispatch` | Same as above |

Uses GitHub Actions cache (`type=gha`) for faster subsequent builds.

---

## Image Details

### CPU Image (`Dockerfile`)
- **Base**: `python:3.12-slim-bookworm`
- **PyTorch**: CPU-only from `download.pytorch.org/whl/cpu`
- **Lightning Pose**: base install (no `[all]` extras)
- **Size**: ~2.5 GB

### GPU Image (`Dockerfile.gpu`)
- **Base**: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
- **PyTorch**: 2.5.1 with CUDA 12.1 + cuDNN 9
- **Lightning Pose**: `[all]` extras
- **Size**: ~6 GB

Both images include:
- All project dependencies (numpy, pandas, OpenCV, scikit-learn, xgboost, etc.)
- Jupyter Notebook + Jupyter Lab
- Code-quality tools (pylint, autopep8)
- `PYTHONPATH=/workspace/src` pre-configured

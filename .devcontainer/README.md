# Dev Container for LightningPoseTrack

## Three Ways to Use This Repo

| Workflow | File | GPU | When To Use |
|----------|------|-----|-------------|
| **Local Dev (CPU)** | `Dockerfile` | ❌ | VS Code on Mac for editing, linting, testing |
| **Colab Direct** | `build_on_colab.ipynb` (Mode A) | ✅ | Daily notebook work (recommended for Colab) |
| **Colab Docker** | `build_on_colab.ipynb` (Mode B) | ✅ | Experimental — if you need pinned deps + VS Code attach |

---

## Workflow 1: Local Development (CPU)

Open in VS Code with **Dev Containers** extension:

```
Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

Uses `Dockerfile` (CPU). Good for:
- Editing `src/` Python modules
- Running pylint / autopep8
- Testing non-GPU code
- Authoring notebooks

> GPU training stays on Colab.

---

## Workflow 2: Colab (Direct — Recommended)

1. Open `colab.research.google.com` with **GPU runtime**
2. File → Open notebook → GitHub → `LightningPoseTrack/.devcontainer/build_on_colab.ipynb`
3. Ensure `MODE = "direct"` at the top
4. **Runtime → Run all**

This installs everything directly on the Colab VM. Then open any notebook from `notebooks/`.

---

## Workflow 3: Colab (Docker — Experimental)

Same steps as above but set `MODE = "docker"`.

Docker requires nested container support on the Colab VM. Some runtimes support it,
most don't. If `dockerd` fails to start, the notebook prints a message and you
should switch to Mode A.

When it works:
- `docker exec -it lightningposetrack bash` to shell in
- VS Code: **Dev Containers: Attach to Running Container** → `lightningposetrack`

---

## GitHub Actions

`.github/workflows/docker-gpu.yml`:

| Trigger | Action |
|---------|--------|
| Push to `main` | Build `Dockerfile.gpu`, push to `kaarthikbalakrishnan/lightningposetrack:latest` |
| Manual (`workflow_dispatch`) | Same |

Requires `DOCKER_HUB_TOKEN` secret in GitHub repository settings.

---

## Image Details

### CPU (`Dockerfile`)
- **Base**: `python:3.12-slim-bookworm`
- **PyTorch**: CPU-only
- **Lightning Pose**: base (no `[all]`)
- **Size**: ~2.5 GB

### GPU (`Dockerfile.gpu`)
- **Base**: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
- **PyTorch**: 2.5.1 + CUDA 12.1 + cuDNN 9
- **Lightning Pose**: `[all]`
- **Size**: ~6 GB

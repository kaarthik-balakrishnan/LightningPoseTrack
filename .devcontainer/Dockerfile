FROM python:3.12-slim-bookworm

LABEL description="LightningPoseTrack Dev Container for pig behavior analysis"

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies: video processing, OpenCV GL rendering, OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY requirements-colab.txt /tmp/requirements-colab.txt

# Core data-science stack
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /tmp/requirements-colab.txt

# Training stack (CPU-only for local dev; GPU training on Colab)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Lightning Pose (base install, without [all] extras to keep image size smaller)
RUN pip install --no-cache-dir lightning-pose omegaconf

# Jupyter ecosystem for notebook development
RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    jupyterlab \
    ipykernel \
    ipywidgets

# ML libs used in notebook 07
RUN pip install --no-cache-dir \
    scikit-learn \
    xgboost \
    joblib

# Code-quality tools
RUN pip install --no-cache-dir \
    pylint \
    autopep8

# Register the IPython kernel for Jupyter
RUN python -m ipykernel install --user --name=lightningposetrack --display-name="Python 3.12 (LightningPoseTrack)"

# Clean pip cache
RUN rm -rf /root/.cache/pip

WORKDIR /workspace

# Set PYTHONPATH so that src/ is importable
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

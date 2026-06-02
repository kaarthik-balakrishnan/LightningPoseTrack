#!/bin/bash
# Setup script for Google Colab environment
# This is used within Colab notebooks to install dependencies

set -e

echo "=== LightningPoseTrack Colab Setup ==="

# System dependencies
apt-get update -qq
apt-get install -y -qq ffmpeg libgl1-mesa-glx libglib2.0-0 tesseract-ocr > /dev/null 2>&1

# Core Python packages
pip install --quiet \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    xgboost \
    opencv-python \
    pyarrow \
    seaborn \
    pyyaml \
    tqdm \
    scipy \
    pytesseract

echo "=== Setup Complete ==="
echo "Run 'pip install lightning-pose[all]' separately in the training notebook."

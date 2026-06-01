# AGENTS.md

# Pig Behavior Analysis Platform

## Objective

Build an end-to-end animal behavior analysis pipeline for top-down pig videos.

The system shall:

1. Train a pose estimation model using Lightning Pose.
2. Run batch pose inference on videos.
3. Extract position and gait features.
4. Detect feeding behavior.
5. Train behavior classifiers.
6. Generate reports and visualizations.
7. Run entirely using:

   * VS Code Devcontainer (development)
   * Google Colab GPU (training/inference)
   * Google Drive (storage)

---

# System Architecture

```text
Videos
    ↓
Pose Estimation
    ↓
Pose Time Series
    ↓
Feature Extraction
    ↓
Behavior Classification
    ↓
Reports + Visualizations
```

---

# Technology Stack

## Development

* VS Code
* Docker Desktop
* Dev Containers extension

## Storage

* Google Drive

## Source Control

* GitHub

## Training

* Google Colab
* Lightning Pose

## Analysis

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* OpenCV
* Matplotlib

---

# Repository Structure

```text
pig-behavior-analysis/

├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_frame_sampling.ipynb
│   ├── 03_pose_training.ipynb
│   ├── 04_pose_inference.ipynb
│   ├── 05_feature_extraction.ipynb
│   ├── 06_feeding_detection.ipynb
│   ├── 07_behavior_classification.ipynb
│   └── 08_reporting.ipynb
│
├── src/
│   ├── io/
│   ├── pose/
│   ├── features/
│   ├── behaviors/
│   ├── visualization/
│   └── reports/
│
├── configs/
├── raw_videos/
├── labeled_frames/
├── trained_models/
├── pose_outputs/
├── features/
├── behavior_outputs/
├── reports/
│
└── AGENTS.md
```

---

# Phase 0: Environment Setup

## Goal

Create a reproducible local development environment.

---

## Agent Task 0.1

Create:

```text
.devcontainer/
```

directory.

---

## Agent Task 0.2

Create Dockerfile.

Requirements:

* Ubuntu base image
* Python 3.11
* git
* ffmpeg
* jupyterlab

Install:

* numpy
* pandas
* matplotlib
* scipy
* scikit-learn
* xgboost
* opencv-python
* pyarrow
* seaborn

NOTE:

Do NOT install Lightning Pose locally.

Lightning Pose will run only in Colab.

---

## Agent Task 0.3

Create devcontainer.json.

Requirements:

* Mount repository.
* Forward Jupyter ports.
* Auto-install Python extension.
* Auto-install Jupyter extension.

---

## Agent Task 0.4

Create Makefile commands:

```bash
make setup
make test
make lint
make notebooks
```

---

# Phase 1: Data Organization

## Goal

Standardize all videos.

---

## Agent Task 1.1

Create folder structure:

```text
raw_videos/
camera_01/
camera_02/
camera_03/
```

---

## Agent Task 1.2

Implement:

```python
src/io/video_inventory.py
```

Functionality:

* scan all videos
* compute duration
* compute frame count
* compute fps
* save metadata csv

Output:

```text
reports/video_inventory.csv
```

---

# Phase 2: Frame Sampling

## Goal

Create labeling dataset.

---

## Agent Task 2.1

Implement:

```python
src/pose/frame_sampler.py
```

Requirements:

Sample frames:

* uniformly
* random
* motion-based

Output:

```text
labeled_frames/
```

---

## Agent Task 2.2

Generate:

```text
300 candidate frames
```

distributed across:

* cameras
* lighting conditions
* behaviors

---

# Phase 3: Lightning Pose Training

## Goal

Train first pose model.

---

## Agent Task 3.1

Create Colab notebook:

```text
03_pose_training.ipynb
```

Responsibilities:

* mount Google Drive
* install Lightning Pose
* load labels
* train model
* save checkpoint

Output:

```text
trained_models/
```

---

## Keypoints

Label:

1. snout
2. left ear
3. right ear
4. neck
5. shoulders
6. mid-back
7. hip
8. tail base

---

## Agent Task 3.2

Implement model evaluation.

Metrics:

* PCK
* mean pixel error
* confidence histograms

Generate:

```text
reports/pose_model_report.html
```

---

# Phase 4: Pose Inference

## Goal

Run model on all videos.

---

## Agent Task 4.1

Create:

```text
04_pose_inference.ipynb
```

Responsibilities:

* load trained model
* iterate through videos
* run inference
* save coordinates

Output:

```text
pose_outputs/
```

Format:

```text
frame
timestamp

snout_x
snout_y

...

tailbase_x
tailbase_y
```

Store as Parquet.

---

# Phase 5: Pose Cleanup

## Goal

Create smooth trajectories.

---

## Agent Task 5.1

Implement:

```python
src/pose/clean_pose.py
```

Requirements:

* confidence filtering
* interpolation
* smoothing

Recommended:

Savitzky-Golay filter.

Output:

clean pose files.

---

# Phase 6: Feature Extraction

## Goal

Convert pose into behavioral features.

---

## Agent Task 6.1

Implement:

```python
src/features/kinematics.py
```

Compute:

* position
* velocity
* acceleration

---

## Agent Task 6.2

Implement:

```python
src/features/orientation.py
```

Compute:

```text
snout -> hip vector
```

Outputs:

* heading
* angular velocity
* turning rate

---

## Agent Task 6.3

Implement:

```python
src/features/spatial.py
```

Compute:

* distance to feeder
* distance to walls
* zone occupancy

---

# Phase 7: Feeding Detection

## Goal

Create initial feeding detector.

---

## Agent Task 7.1

Implement:

```python
src/behaviors/feeding.py
```

Rule:

```text
snout near feeder
AND
facing feeder
AND
duration > threshold
```

Output:

feeding events.

---

## Agent Task 7.2

Generate:

```text
feeding_events.parquet
```

Fields:

```text
start_time
end_time
duration
camera
```

---

# Phase 8: Behavior Classification

## Goal

Classify behavior automatically.

---

## Version 1

Labels:

* feeding
* standing
* walking
* turning
* resting

---

## Agent Task 8.1

Create labeling tool.

Input:

pose trajectories.

Output:

behavior labels.

---

## Agent Task 8.2

Implement feature matrix generation.

Window sizes:

* 1 sec
* 3 sec
* 5 sec

---

## Agent Task 8.3

Train XGBoost classifier.

Metrics:

* accuracy
* precision
* recall
* F1

Save:

```text
trained_models/behavior_model.pkl
```

---

# Phase 9: Reporting

## Goal

Generate automated reports.

---

## Agent Task 9.1

Create:

```python
src/reports/daily_report.py
```

Generate:

* distance traveled
* feeding duration
* occupancy heatmaps
* speed distributions

Output:

```text
reports/daily_summary.html
```

---

# Phase 10: Future Roadmap

## Milestone A

Detailed gait analysis.

Add:

* front-left hoof
* front-right hoof
* rear-left hoof
* rear-right hoof

Outputs:

* stride length
* stride frequency
* stance duration

---

## Milestone B

Cross-camera fusion.

Goal:

Single coordinate system across all cameras.

---

## Milestone C

Real-time inference.

Pipeline:

```text
Camera
→ Pose
→ Features
→ Behavior
→ Alerts
```

---

## Milestone D

Health Monitoring.

Detect:

* reduced mobility
* abnormal feeding
* lameness
* illness indicators

---

# Success Criteria

The project is complete when:

1. Videos can be processed automatically.
2. Pose is estimated reliably.
3. Position and orientation are tracked.
4. Feeding behavior is detected.
5. Behavior classification achieves useful performance.
6. Reports are generated automatically.
7. Entire workflow runs using:

   * Devcontainer
   * Google Drive
   * Google Colab
   * GitHub

```
```

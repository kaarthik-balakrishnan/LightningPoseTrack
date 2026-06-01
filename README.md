# LightningPoseTrack — Pig Behavior Analysis Pipeline

An end-to-end pipeline for tracking animal poses and analyzing behavior from top-down pig videos. Everything runs on **Google Colab** (GPU) with data stored on **Google Drive** and code on **GitHub**.

## Architecture

```
Google Drive (Videos)           GitHub (Code)
        │                            │
        └──────────┬─────────────────┘
                   ▼
          Google Colab (GPU)
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    Pose Est.   Feature     Behavior
    (Lightning   Ext.       Classifier
      Pose)                 (XGBoost)
        │          │          │
        └──────────┼──────────┘
                   ▼
          Reports & Visualizations
                   │
                   ▼
          Google Drive (Outputs)
```

---

## Prerequisites

1. **Google Account** with Google Drive and Colab access
2. **Colab Pro/Pro+** (recommended for GPU training — provides longer runtimes and better GPUs)
3. **GitHub Account**
4. **Raw video files** stored on Google Drive in the shared folder:  
   `https://drive.google.com/drive/folders/1X_41ZW3HfwVeft2lPld3XNqXsdxRDIwb`
5. **Video naming convention**: Files contain `_N.asf` where `N` = camera number (1–4)

---

## Step 1: Fork/Create the GitHub Repository

1. Go to [github.com/new](https://github.com/new) and create a new repository named `LightningPoseTrack`
2. Run these commands in your local terminal:

```bash
cd /path/to/LightningPoseTrack
git remote add origin https://github.com/YOUR_USERNAME/LightningPoseTrack.git
git add .
git commit -m "Initial commit: full pipeline with Colab notebooks"
git branch -M main
git push -u origin main
```

> **Note**: The local machine is only used to push code to GitHub. All actual processing happens in Colab.

---

## Step 2: Add Shared Drive Folder to Your Google Drive

1. Open the shared folder link:  
   `https://drive.google.com/drive/folders/1X_41ZW3HfwVeft2lPld3XNqXsdxRDIwb`
2. Right-click → **Add shortcut to Drive**
3. Choose a location (e.g., `My Drive/PigBehavior`) and click **Add shortcut**
4. This makes the folder accessible in Colab at `/content/drive/My Drive/PigBehavior/...`

### Expected Drive Folder Structure After Setup

After running all notebooks, your Google Drive will look like this:

```
My Drive/
└── PigBehavior/
    ├── raw_videos/                  ← Shared folder (shortcut)
    │   ├── session_01/
    │   │   ├── video_1.asf
    │   │   ├── video_2.asf
    │   │   └── ...
    │   └── session_02/
    │       └── ...
    │
    ├── labeled_frames/              ← Created by Notebook 02
    │   └── session_01/
    │       ├── video_cam1_frame000001.jpg
    │       └── ...
    │
    ├── trained_models/              ← Created by Notebook 03 & 07
    │   ├── pose_model.ckpt
    │   └── behavior_model.pkl
    │
    ├── pose_outputs/                ← Created by Notebook 04
    │   └── session_01/
    │       ├── video_cam1_pose.parquet
    │       └── ...
    │
    ├── features/                    ← Created by Notebook 05
    │   └── session_01/
    │       ├── video_cam1_features.parquet
    │       └── ...
    │
    ├── behavior_outputs/            ← Created by Notebook 06
    │   └── feeding_events.parquet
    │
    └── reports/                     ← Created by Notebook 08
        ├── report_session01_cam1.html
        └── ...
```

---

## Step 3: Run Notebooks in Google Colab

Run the notebooks **in order**. Each notebook mounts Google Drive, clones the GitHub repo, installs dependencies, and saves outputs back to Drive — all within Colab.

### Row 1: Open a Notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Sign in with your Google account
3. File → Open notebook → GitHub → paste your repo URL
4. Select the notebook you want to run

**OR** use the direct Colab link (replace `YOUR_USERNAME`):

```
https://colab.research.google.com/github/YOUR_USERNAME/LightningPoseTrack/blob/main/notebooks/01_Data_Audit.ipynb
```

### Row 2: Set Runtime to GPU

For notebooks 03 (training) and 04 (inference):
- Runtime → Change runtime type → Hardware accelerator → **GPU (T4 or better)**
- For Colab Pro+, select **A100** or **V100** when available

---

## Notebook Pipeline

### Notebook 01 — Data Audit

**File**: `notebooks/01_Data_Audit.ipynb`

Scans all videos across session folders and generates a metadata inventory.

**Steps**:
1. Set `GITHUB_REPO_URL` to your repo URL
2. Set `DRIVE_RAW_VIDEOS` to the shortcut path of the shared folder (e.g., `/content/drive/My Drive/PigBehavior/raw_videos`)
3. Run all cells
4. Output: `reports/video_inventory.csv` on Drive

---

### Notebook 02 — Frame Sampling

**File**: `notebooks/02_Frame_Sampling.ipynb`

Extracts candidate frames from videos for manual labeling. Samples ~300 frames total using uniform, random, or motion-based sampling.

**Steps**:
1. Run all cells to extract frames
2. **After Colab run**: Download the `labeled_frames/` folder from Google Drive
3. Label keypoints using:
   - **Lightning Pose Labeling Tool** (recommended): `python -m lightning_pose.label`
   - **LabelMe**: `pip install labelme && labelme`
   - **DeepLabCut** labeling convention
4. **Keypoints to label** (8 total):
   - `snout`, `left_ear`, `right_ear`, `neck`, `shoulders`, `mid_back`, `hip`, `tail_base`
5. Upload labeled data back to `My Drive/PigBehavior/labeled_frames/`

**Output**: `labeled_frames/` on Drive

---

### Notebook 03 — Pose Training (Lightning Pose)

**File**: `notebooks/03_Pose_Training.ipynb`

Trains a pose estimation model using Lightning Pose with GPU acceleration.

**Steps**:
1. **Set runtime to GPU** (T4/V100/A100)
2. Set the paths to your labeled data
3. Update the `csv_file` path in the config to point to your Lightning Pose label file
4. Configure training hyperparameters:
   - `BACKBONE`: `resnet50` (fast) or `resnet101` (more accurate)
   - `MAX_EPOCHS`: Start with 200
   - `BATCH_SIZE`: depends on GPU memory (16–32 for T4)
5. Run all cells — training typically takes 30–90 minutes
6. Model checkpoint is saved to Drive at `trained_models/pose_model.ckpt`

**Keypoints defined**:
| # | Keypoint    |
|---|-------------|
| 1 | snout       |
| 2 | left_ear    |
| 3 | right_ear   |
| 4 | neck        |
| 5 | shoulders   |
| 6 | mid_back    |
| 7 | hip         |
| 8 | tail_base   |

---

### Notebook 04 — Pose Inference

**File**: `notebooks/04_Pose_Inference.ipynb`

Runs the trained pose model on all videos to extract keypoint coordinates.

**Steps**:
1. **Set runtime to GPU** for faster inference
2. Set `MODEL_CHECKPOINT` to the model filename saved in Step 3
3. Run all cells
4. Output: per-video parquet files in `pose_outputs/` on Drive

**Output format** (parquet):
```
frame | timestamp | snout_x | snout_y | snout_likelihood | left_ear_x | ... | tail_base_y | tail_base_likelihood
```

---

### Notebook 05 — Feature Extraction

**File**: `notebooks/05_Feature_Extraction.ipynb`

Extracts kinematic, orientation, and spatial features from pose keypoints.

**Before running**: Set these configuration values:
- `FEEDER_X`, `FEEDER_Y`: The pixel coordinates of the feeder (check one video frame)
- `FRAME_WIDTH`, `FRAME_HEIGHT`: Your video dimensions
- `FPS`: Frames per second of your videos

**Features extracted**:
- **Kinematics**: Speed & acceleration per keypoint, centroid speed/acceleration
- **Orientation**: Heading angle (snout→hip vector), angular velocity, turning rate
- **Spatial**: Distance to feeder, distance to walls, zone occupancy

**Output**: `features/` on Drive (one parquet file per video)

---

### Notebook 06 — Feeding Detection

**File**: `notebooks/06_Feeding_Detection.ipynb`

Detects feeding events using rule-based heuristics on pose data.

**Detection logic**:
```
snout_near_feeder AND facing_feeder AND duration > threshold
```

**Before running**: Set `FEEDER_X`, `FEEDER_Y`, `FPS`, and thresholds.

**Output**: `behavior_outputs/feeding_events.parquet` on Drive

---

### Notebook 07 — Behavior Classification

**File**: `notebooks/07_Behavior_Classification.ipynb`

Trains an XGBoost classifier on pose features to classify behaviors.

**Labels**: `feeding`, `standing`, `walking`, `turning`, `resting`

**Steps**:
1. The notebook generates a labeling template CSV on Drive
2. Download and fill in behavior labels manually
3. Re-upload and run training
4. Model saved to `trained_models/behavior_model.pkl`

---

### Notebook 08 — Reporting

**File**: `notebooks/08_Reporting.ipynb`

Generates automated HTML reports with visualizations.

**Report contents**:
- Summary statistics
- Trajectory plot
- Speed distribution histogram
- Heading angle over time
- Feeding event timeline
- Per-session summary

**Output**: `reports/report_*.html` on Drive (open in browser)

---

## Key Configuration Parameters

All configurable parameters are in the **first code cell** of each notebook under `# CONFIGURATION`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GITHUB_REPO_URL` | — | URL to your GitHub repo |
| `DRIVE_ROOT` | `/content/drive/My Drive/PigBehavior` | Root path on Drive |
| `DRIVE_RAW_VIDEOS` | `{DRIVE_ROOT}/raw_videos` | Path to raw videos |
| `FPS` | 30.0 | Video frame rate |
| `FEEDER_X`, `FEEDER_Y` | — | Feeder position (pixels) |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | 640×480 | Video dimensions |
| `BACKBONE` | `resnet50` | Model backbone |
| `MAX_EPOCHS` | 200 | Training epochs |
| `CONFIDENCE_THRESHOLD` | 0.5 | Pose confidence threshold |

---

## Google Drive Structure

The pipeline expects and creates this structure on Google Drive:

```
My Drive/
└── PigBehavior/
    ├── raw_videos/              ← Your video sessions (shared folder shortcut)
    │   ├── session_01/
    │   │   ├── video_1.asf
    │   │   ├── video_2.asf
    │   │   ├── video_3.asf
    │   │   └── video_4.asf
    │   └── session_02/
    │       └── ...
    │
    ├── labeled_frames/          ← Created by Notebook 02
    ├── trained_models/          ← Created by Notebook 03 & 07
    │   ├── pose_model.ckpt
    │   └── behavior_model.pkl
    ├── pose_outputs/            ← Created by Notebook 04
    ├── features/                ← Created by Notebook 05
    ├── behavior_outputs/        ← Created by Notebook 06
    └── reports/                 ← Created by Notebook 08
```

---

## Troubleshooting

### "CUDA out of memory" in Notebook 03
- Reduce `BATCH_SIZE` (try 8)
- Use `resnet50` instead of `resnet101`
- Restart runtime and clear memory: Runtime → Factory reset runtime
- Upgrade to Colab Pro+ for A100 GPU

### "File not found" when mounting Drive
- Ensure you added the shared folder as a shortcut to your Drive
- Check the path exactly: `ls "/content/drive/My Drive/PigBehavior/"`

### OpenCV can't read .asf files
- Run: `!apt-get install -y ffmpeg libavcodec-extra`
- Or use `ffmpeg` to convert: `!ffmpeg -i input.asf -c:v libx264 output.mp4`

### "pip install lightning-pose" fails
- Use: `!pip install --quiet "lightning-pose[all]"`
- Ensure CUDA is available: `import torch; torch.cuda.is_available()`
- Set runtime to GPU

### Notebook 01 shows 0 videos
- Verify the path to `raw_videos` contains session subfolders
- Check video file extensions (supports: .asf, .mp4, .avi, .mov, .mkv)

---

## Source Code Reference

| File | Purpose |
|------|---------|
| `src/io/video_inventory.py` | Scan videos, compute metadata |
| `src/pose/frame_sampler.py` | Uniform/random/motion-based frame sampling |
| `src/pose/clean_pose.py` | Confidence filtering, interpolation, smoothing |
| `src/features/kinematics.py` | Speed, acceleration per keypoint |
| `src/features/orientation.py` | Heading, angular velocity, turning rate |
| `src/features/spatial.py` | Distance to feeder/walls, zone occupancy |
| `src/behaviors/feeding.py` | Rule-based feeding detection |
| `src/reports/daily_report.py` | HTML/PDF report generation |

---

## Pipelines Quick Reference

```
Notebook 01: Data Audit        → reports/video_inventory.csv
Notebook 02: Frame Sampling    → labeled_frames/ (manual labeling needed)
Notebook 03: Pose Training     → trained_models/pose_model.ckpt
Notebook 04: Pose Inference    → pose_outputs/*.parquet
Notebook 05: Feature Ext.      → features/*.parquet
Notebook 06: Feeding Detection → behavior_outputs/feeding_events.parquet
Notebook 07: Behavior Class.   → trained_models/behavior_model.pkl
Notebook 08: Reporting         → reports/report_*.html
```

## License

MIT

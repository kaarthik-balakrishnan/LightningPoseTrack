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

## Pipeline (8 notebooks)
| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Data Audit | Scan videos, parse metadata, inventory report |
| 02 | Frame Sampling | Sample frames with animal present (background subtraction) |
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

## Directory Structure
```
LightningPoseTrack/
├── notebooks/          # 8 Colab notebooks (01-08)
├── src/
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
├── requirements_colab.txt
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

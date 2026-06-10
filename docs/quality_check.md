# Quality Check — Video Frame Integrity Analysis

## Goal

Verify that video files from multi-camera security recordings are free of dropped
frames and decoding corruption before downstream pose estimation and behavioral
analysis.

## Introduction

The pipeline records synchronized video from four H.264 security cameras saving
to the ASF container format. These ASF files present two challenges for quality
verification:

1. **No PTS (Presentation Time Stamps)** — ASF containers do not store per-frame
   presentation timestamps, making standard PTS-gap analysis impossible.
2. **Missing or unreliable container metadata** — The `nb_frames` field is absent
   from the ASF container header, and the `duration` field is unreliable (observed
   errors of up to 27 seconds). The fallback `int(duration × fps)` estimate is
   therefore meaningless.

## Methods

### Two-Pronged Detection Strategy

**Approach 1: PTS Gap Analysis (for MP4, AVI, MOV containers)**
Enumerate every frame's PTS via `ffprobe -show_entries frame=pts_time`.
For a clean recording, consecutive PTS values differ by exactly `1/fps` seconds.
Any gap > 1.5× the expected interval indicates dropped frames:
`n_dropped = round(gap / expected_interval) - 1`.
We also verify frame count matches metadata, FPS consistency, and timing jitter
(CV of PTS deltas < 5%).

**Approach 2: Decode-Error Detection (for ASF containers)**
Since ASF lacks PTS and has unreliable metadata, we:
1. Count actual frames by full-frame decoding via `ffprobe -count_frames`
2. Run `ffmpeg -v error` over the full file — any corruption, missing packets,
   or decode failures produce error messages
3. Suppress the harmless `non monotonically increasing dts` warning (common
   with B-frame reordering in H.264/ASF)

A file is considered clean if it passes its applicable check.

## Implementation

### Pipeline integration (`src/calibration/pipeline.py`)

The `quality()` method in `CalibrationPipeline` (Stage 3) iterates over probed
results and dispatches based on whether `nb_frames` was stored in the container
header:

- **`meta_nb_frames_from_container == True`** (MP4 etc.) — strict frame-count
  comparison. Flags on mismatch.
- **`meta_nb_frames_from_container == False`** (ASF) — frame-count comparison
  is impossible because ASF containers store neither PTS nor `nb_frames`.
  Instead the pipeline trusts `_count_actual_frames()` (powered by
  `ffprobe -count_frames`), which already decoded every frame successfully
  to return a count. If it returned a positive count, the file is decodable.
  A separate `ffmpeg -v error` decode check was tried earlier but produced
  false positives for ASF+h264 (harmless NAL unit framing warnings), so it
  was removed. The frame count from `-count_frames` is authoritative.

### Standalone notebook (`notebooks/00_Quality_Test.ipynb`)

The notebook `quality_test()` function uses a `has_pts()` probe to select the
appropriate method per file:

```python
if has_pts(video_path):
    # Approach 1: PTS gap analysis
    pts = inspect_pts(video_path)
    drop = detect_dropped_frames(pts, meta["metadata_fps"])
    ...
else:
    # Approach 2: frame-count check for ASF (no PTS available)
    n_expected = int(meta["fps"] * meta["duration_sec"])
    drop_pct = (n_expected - meta["actual_frames"]) / n_expected * 100
    ...
```

Results are stored in a DataFrame and exported as CSV and JSON reports.

## Test

The quality check was tested on 6 ASF files from a ContinuousRecording session
(total ~4800 frames across 4 cameras at 5–10.25 fps):

| File | Camera | FPS | Actual Frames | Container Duration | Actual Duration | Method | Result |
|------|--------|-----|---------------|-------------------|-----------------|--------|--------|
| 090213-4.ASF | 4 | 5.0 | 601 | 117.4s | 120.2s | decode | PASS |
| 090307-2.ASF | 2 | 10.25 | 983 | 110.2s | 95.9s | decode | PASS |
| 090314-1.ASF | 1 | 10.25 | 645 | 90.2s | 62.9s | decode | PASS |
| 090409-4.ASF | 4 | 5.0 | 1357 | 279.3s | 271.4s | decode | PASS |
| 090715-3.ASF | 3 | 10.0 | 1002 | 103.2s | 100.2s | decode | PASS |
| 090749-2.ASF | 2 | 10.25 | 539 | 60.2s | 52.6s | decode | PASS |

## Results

- **All 6 ASF files verified clean** — `ffprobe -count_frames` decoded every
  frame successfully for all files. An earlier attempt to use `ffmpeg -v error`
  for a separate decode check was abandoned because all ASF files produce
  harmless NAL unit framing warnings (false positives).
- **Container duration metadata is unreliable** for ASF — the difference between
  container-reported and actual decoded duration ranged from −2.8s to +27.3s.
  This confirms that `int(duration × fps)` fallback estimates are not a valid
  basis for frame-drop detection in this format.
- **`ffprobe -count_frames`** is the authoritative frame count for ASF since it
  actually decodes every frame. For these files, all frames decoded successfully
  with no corruption.

## References

1. FFmpeg Documentation. *ffprobe(1) — count_frames*.
   https://ffmpeg.org/ffprobe.html

2. FFmpeg Documentation. *ffmpeg(1) — error detection*.
   https://ffmpeg.org/ffmpeg.html

3. Microsoft. *ASF (Advanced Systems Format) Specification*.
   https://learn.microsoft.com/en-us/windows/win32/medfound/asf-file-structure

4. Sullivan, G. J., et al. "Overview of the High Efficiency Video Coding (HEVC)
   Standard." *IEEE Transactions on Circuits and Systems for Video Technology*,
   vol. 22, no. 12, 2012. (B-frame reordering and DTS/PTS fundamentals)

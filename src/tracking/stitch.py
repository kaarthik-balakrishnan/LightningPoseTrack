"""
src/tracking/stitch.py
=======================
Assemble per-camera pose tracks into one global trajectory.

Timeline format (unified_timeline.csv from NB02b)
-------------------------------------------------
  Per-frame CSV — one row per decoded video frame:
    camera     int          (1-4)
    filename   str          e.g. "161308-1.ASF"
    frame_idx  int          0-indexed within that file
    time_rel   float        seconds since session start
    time_abs   float        wall-clock seconds since midnight

Pipeline
--------
1. Load unified_timeline.csv                 → per-frame wall-clock anchors
2. Load calibration_result.json (optional)   → fps per file
3. Load pose parquet files from NB04         → per-frame keypoint pixels
4. Merge timeline ↔ pose on (camera, filename, frame_idx=frame)
5. Apply homography                          → pixels → world cm
6. Fuse overlap windows                      → pick best camera per time-bin
7. Interpolate short gaps                    → bridge camera transitions
8. Savitzky-Golay smooth
9. Export                                    → parquet + CSV

Output columns
--------------
  abs_time_sec           wall-clock seconds (from timeline time_abs)
  time_rel               seconds since session start
  camera                 source camera
  filename               source video file
  source                 cam{N}_exclusive | cam{A}_cam{B}_overlap | interpolated
  gap_flag               True when previous row was > GAP_FLAG_SEC ago
  mean_likelihood        average keypoint confidence
  valid                  mean_likelihood >= CONFIDENCE_THRESHOLD
  centroid_wx / wy       hip world position (cm)
  {kp}_wx / {kp}_wy      world position per keypoint (cm)
  {kp}_likelihood        confidence per keypoint
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.calibration.homography import (
    load_all_homographies,
    load_camera_layout,
    pixel_array_to_world,
)

# ── tuneable constants ────────────────────────────────────────────────────────

KEYPOINT_NAMES = [
    "snout", "left_ear", "right_ear", "neck",
    "shoulders", "mid_back", "hip", "tail_base",
]

CONFIDENCE_THRESHOLD = 0.5      # min mean likelihood to use a detection
OVERLAP_BIN_SEC      = 0.10     # time-bin width for overlap fusion (s)
GAP_INTERP_LIMIT_SEC = 1.5      # max gap to fill by linear interpolation
GAP_FLAG_SEC         = 0.5      # gaps > this get gap_flag = True
SMOOTH_WINDOW        = 7        # Savitzky-Golay window (odd, >= 5)
SMOOTH_POLYORDER     = 2


# ── internal helpers ──────────────────────────────────────────────────────────

def _extract_cam_and_stem(pose_path: Path) -> tuple[int, str]:
    """
    Parse camera number and original video stem from a pose parquet path.
    NB04 names files as  {video_stem}_cam{N}_pose.parquet.
    Falls back to the numeric suffix pattern in the video stem itself.
    """
    import re
    name = pose_path.stem   # e.g. "161308-1_cam1_pose"
    # Try _cam{N}_pose suffix first
    m = re.search(r"_cam(\d+)_pose$", name)
    if m:
        cam  = int(m.group(1))
        stem = name[:m.start()]          # "161308-1"
        return cam, stem
    # Fall back: last hyphen/underscore-delimited integer in stem
    parts = re.split(r"[-_]+", name.replace("_pose", ""))
    for p in reversed(parts):
        try:
            n = int(p)
            if 1 <= n <= 4:
                # Reconstruct stem by stripping that trailing segment
                stem = name[:name.rfind(p) - 1].replace("_pose", "")
                return n, stem
        except ValueError:
            continue
    return 0, name.replace("_pose", "")


def _match_timeline_to_pose(
    timeline: pd.DataFrame,
    pose_path: Path,
) -> pd.DataFrame | None:
    """
    Filter the per-frame timeline to rows that belong to this pose parquet.
    Returns None if no match found.
    """
    cam, stem = _extract_cam_and_stem(pose_path)
    if cam == 0:
        return None

    # Try to match filename exactly (stem.ASF / stem.asf / stem.mp4 etc.)
    mask = (
        (timeline["camera"] == cam) &
        timeline["filename"].apply(lambda f: Path(f).stem == stem)
    )
    sub = timeline[mask]
    if sub.empty:
        # Looser: camera matches and stem is a prefix of filename
        mask2 = (
            (timeline["camera"] == cam) &
            timeline["filename"].apply(
                lambda f: Path(f).stem.startswith(stem) or stem.startswith(Path(f).stem)
            )
        )
        sub = timeline[mask2]

    return sub if not sub.empty else None


def _build_world_df(
    pose_path: Path,
    timeline_sub: pd.DataFrame,
    H: np.ndarray,
    camera: int,
) -> pd.DataFrame | None:
    """
    Merge pose parquet with its timeline slice, apply homography,
    return a world-coordinate DataFrame.
    """
    df_pose = pd.read_parquet(pose_path)

    # Rename frame_idx → frame for merge (NB04 uses "frame" column)
    tl = timeline_sub[["frame_idx", "time_abs", "time_rel"]].rename(
        columns={"frame_idx": "frame"}
    )

    merged = df_pose.merge(tl, on="frame", how="left")

    if "time_abs" not in merged.columns or merged["time_abs"].isna().all():
        return None

    merged["abs_time_sec"] = merged["time_abs"]
    merged["camera"]       = camera
    merged["filename"]     = Path(pose_path).name

    # Apply homography to each keypoint
    for kp in KEYPOINT_NAMES:
        xcol, ycol = f"{kp}_x", f"{kp}_y"
        if xcol not in merged.columns or ycol not in merged.columns:
            merged[f"{kp}_wx"] = np.nan
            merged[f"{kp}_wy"] = np.nan
            continue
        uv  = merged[[xcol, ycol]].values.astype(float)
        xy  = pixel_array_to_world(uv, H)
        merged[f"{kp}_wx"] = xy[:, 0]
        merged[f"{kp}_wy"] = xy[:, 1]

    # Confidence
    like_cols = [f"{kp}_likelihood" for kp in KEYPOINT_NAMES
                 if f"{kp}_likelihood" in merged.columns]
    merged["mean_likelihood"] = (
        merged[like_cols].mean(axis=1) if like_cols else np.nan
    )
    merged["valid"] = merged["mean_likelihood"] >= CONFIDENCE_THRESHOLD

    return merged


def _fuse_overlap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    t_start: float,
    t_end: float,
) -> pd.DataFrame:
    """Pick the more-confident detection in each time-bin for the overlap window."""
    bins = np.arange(t_start, t_end + OVERLAP_BIN_SEC, OVERLAP_BIN_SEC)

    def _best(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[(df["abs_time_sec"] >= t_start) & (df["abs_time_sec"] < t_end)].copy()
        if sub.empty:
            return sub
        sub["_bin"] = pd.cut(
            sub["abs_time_sec"], bins=bins, labels=bins[:-1], right=False
        )
        return (
            sub.sort_values("mean_likelihood", ascending=False)
               .groupby("_bin", observed=True).first()
               .reset_index(drop=True)
        )

    combined = pd.concat([_best(df_a), _best(df_b)], ignore_index=True)
    if combined.empty:
        return combined
    if "_bin" in combined.columns:
        combined = (
            combined.sort_values("mean_likelihood", ascending=False)
                    .groupby("_bin", observed=True).first()
                    .reset_index(drop=True)
                    .drop(columns=["_bin"], errors="ignore")
        )
    return combined.sort_values("abs_time_sec").reset_index(drop=True)


def _interpolate_gaps(df: pd.DataFrame, mean_fps: float) -> pd.DataFrame:
    max_gap = max(1, int(GAP_INTERP_LIMIT_SEC * mean_fps))
    cols = (
        [f"{kp}_wx" for kp in KEYPOINT_NAMES if f"{kp}_wx" in df.columns] +
        [f"{kp}_wy" for kp in KEYPOINT_NAMES if f"{kp}_wy" in df.columns]
    )
    for col in cols:
        df[col] = df[col].interpolate(
            method="linear", limit=max_gap, limit_direction="forward"
        )
    return df


def _smooth(df: pd.DataFrame) -> pd.DataFrame:
    w = SMOOTH_WINDOW
    if len(df) < w:
        return df
    for kp in KEYPOINT_NAMES:
        for ax in ("wx", "wy"):
            col = f"{kp}_{ax}"
            if col not in df.columns:
                continue
            vals = df[col].values.astype(float)
            ok   = ~np.isnan(vals)
            if ok.sum() < w:
                continue
            try:
                vals[ok] = savgol_filter(vals[ok], w, SMOOTH_POLYORDER)
                df[col]  = vals
            except Exception:
                pass
    return df


# ── public API ────────────────────────────────────────────────────────────────

def stitch_session(
    session_name:   str,
    pose_dir:       str | Path,
    homography_dir: str | Path,
    timeline_csv:   str | Path,
    output_dir:     str | Path,
    calib_json:     str | Path | None = None,
    smooth:         bool = True,
    verbose:        bool = True,
) -> pd.DataFrame:
    """
    Stitch all camera pose tracks for one session into a single global
    trajectory in world coordinates.

    Parameters
    ----------
    session_name   : folder name of the session, e.g. "260608.00000009"
    pose_dir       : Drive folder containing per-video pose parquet files
    homography_dir : Drive folder with H_cam{N}.npy + camera_layout.json
    timeline_csv   : unified_timeline.csv written by NB02b
                     Path format: {DRIVE_ROOT}/{session}/calibration_output/unified_timeline.csv
    output_dir     : where to write stitched parquet / CSV
    calib_json     : optional calibration_result.json (same folder as timeline_csv)
                     used to look up per-file fps; if absent, fps is inferred
                     from consecutive timeline rows
    smooth         : apply Savitzky-Golay smoothing

    Returns
    -------
    pd.DataFrame  (see module docstring for column list)
    """
    pose_dir       = Path(pose_dir)
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load homographies ─────────────────────────────────────────────────
    homographies = load_all_homographies(homography_dir)
    if not homographies:
        raise RuntimeError(
            f"No homography files found in {homography_dir}.\n"
            "Run 02c_Spatial_Calibration.ipynb first."
        )
    if verbose:
        print(f"Homographies loaded: cameras {sorted(homographies)}")

    # ── 2. Load camera layout ────────────────────────────────────────────────
    try:
        camera_layout = {e["camera"]: e for e in load_camera_layout(homography_dir)}
    except FileNotFoundError:
        camera_layout = {}
        warnings.warn("camera_layout.json not found — spatial overlap detection skipped.")

    # ── 3. Load per-frame timeline ───────────────────────────────────────────
    timeline = pd.read_csv(timeline_csv)
    required = {"camera", "filename", "frame_idx", "time_abs"}
    missing  = required - set(timeline.columns)
    if missing:
        raise ValueError(
            f"unified_timeline.csv is missing columns: {missing}\n"
            f"Found: {list(timeline.columns)}\n"
            "Make sure you are using the correct path "
            "({DRIVE_ROOT}/{session_name}/calibration_output/unified_timeline.csv)."
        )
    timeline["camera"] = timeline["camera"].astype(int)
    if verbose:
        print(f"Timeline: {len(timeline):,} frame-rows, "
              f"cameras {sorted(timeline['camera'].unique())}")

    # ── 4. Optional: load fps lookup from calibration_result.json ────────────
    fps_lookup: dict[tuple[int, str], float] = {}   # (camera, filename) → fps
    if calib_json is None:
        # Infer calib_json path from timeline_csv location
        calib_json = Path(timeline_csv).parent / "calibration_result.json"
    if Path(calib_json).exists():
        with open(calib_json) as f:
            cdata = json.load(f)
        for cam_str, entries in cdata.get("per_camera_timelines", {}).items():
            cam = int(cam_str)
            for entry in entries:
                fps_lookup[(cam, entry["filename"])] = float(entry["fps"])
        if verbose and fps_lookup:
            print(f"fps lookup loaded for {len(fps_lookup)} files")
    else:
        if verbose:
            print("calibration_result.json not found — fps inferred from timeline")

    # ── 5. Find all pose parquet files ───────────────────────────────────────
    pose_files = sorted(pose_dir.rglob("*.parquet"))
    if verbose:
        print(f"Pose parquet files found: {len(pose_files)}")

    # ── 6. Load, transform, timestamp ────────────────────────────────────────
    per_camera: dict[int, list[pd.DataFrame]] = {c: [] for c in range(1, 5)}

    for pose_path in pose_files:
        cam, stem = _extract_cam_and_stem(pose_path)
        if cam == 0 or cam not in homographies:
            if verbose and cam == 0:
                print(f"  {pose_path.name}: could not parse camera — skipped")
            continue

        tl_sub = _match_timeline_to_pose(timeline, pose_path)
        if tl_sub is None:
            if verbose:
                print(f"  Cam{cam} {pose_path.name}: no timeline match — skipped")
            continue

        try:
            df_w = _build_world_df(pose_path, tl_sub, homographies[cam], cam)
            if df_w is None or df_w.empty:
                if verbose:
                    print(f"  Cam{cam} {pose_path.name}: empty after merge — skipped")
                continue
            per_camera[cam].append(df_w)
            if verbose:
                pct = 100 * df_w["valid"].mean()
                print(f"  Cam{cam} {pose_path.name}: "
                      f"{len(df_w)} frames, {pct:.0f}% confident")
        except Exception as exc:
            if verbose:
                print(f"  Error on {pose_path.name}: {exc}")

    # Concatenate per camera
    cam_dfs: dict[int, pd.DataFrame] = {}
    for cam, parts in per_camera.items():
        if parts:
            cam_dfs[cam] = (
                pd.concat(parts, ignore_index=True)
                  .sort_values("abs_time_sec")
                  .reset_index(drop=True)
            )
    if not cam_dfs:
        raise RuntimeError("No pose data loaded after world-transform.")

    cameras = sorted(cam_dfs)
    if verbose:
        print(f"\nCameras with pose data: {cameras}")

    # ── 7. Determine temporal overlap windows ────────────────────────────────
    overlap_windows: list[tuple[int, int, float, float]] = []
    for i in range(len(cameras) - 1):
        ca, cb = cameras[i], cameras[i + 1]
        t_s = max(cam_dfs[ca]["abs_time_sec"].min(), cam_dfs[cb]["abs_time_sec"].min())
        t_e = min(cam_dfs[ca]["abs_time_sec"].max(), cam_dfs[cb]["abs_time_sec"].max())
        if t_e > t_s:
            overlap_windows.append((ca, cb, t_s, t_e))
            if verbose:
                print(f"  Overlap Cam{ca}↔Cam{cb}: {t_s:.1f}→{t_e:.1f}s "
                      f"({t_e - t_s:.1f}s)")

    # ── 8. Assemble segments ─────────────────────────────────────────────────
    overlap_spans: dict[int, list[tuple[float, float]]] = {c: [] for c in cameras}
    for ca, cb, t_s, t_e in overlap_windows:
        overlap_spans[ca].append((t_s, t_e))
        overlap_spans[cb].append((t_s, t_e))

    segments: list[pd.DataFrame] = []

    for cam in cameras:
        df = cam_dfs[cam]
        mask = np.ones(len(df), dtype=bool)
        for t_s, t_e in overlap_spans[cam]:
            mask &= ~((df["abs_time_sec"] >= t_s) & (df["abs_time_sec"] <= t_e))
        excl = df[mask & df["valid"]].copy()
        if len(excl):
            excl["source"] = f"cam{cam}_exclusive"
            segments.append(excl)

    for ca, cb, t_s, t_e in overlap_windows:
        fused = _fuse_overlap(cam_dfs[ca], cam_dfs[cb], t_s, t_e)
        if not fused.empty:
            fused = fused[fused["valid"]].copy()
        if not fused.empty:
            fused["source"] = f"cam{ca}_cam{cb}_overlap"
            segments.append(fused)

    if not segments:
        raise RuntimeError(
            "No valid trajectory segments — check confidence threshold "
            f"({CONFIDENCE_THRESHOLD}) or pose file matching."
        )

    trajectory = (
        pd.concat(segments, ignore_index=True)
          .sort_values("abs_time_sec")
          .reset_index(drop=True)
    )

    # ── 9. Fill gaps + smooth ────────────────────────────────────────────────
    # Estimate mean fps from timeline time deltas within files
    dt_vals = (
        timeline.sort_values(["camera", "filename", "frame_idx"])
                .groupby(["camera", "filename"])["time_abs"]
                .diff()
                .dropna()
    )
    dt_vals = dt_vals[dt_vals > 0]
    mean_fps = float(1.0 / dt_vals.mean()) if not dt_vals.empty else 10.0

    trajectory = _interpolate_gaps(trajectory, mean_fps)
    trajectory["source"] = trajectory["source"].fillna("interpolated")

    if smooth:
        trajectory = _smooth(trajectory)

    # ── 10. Metadata columns ─────────────────────────────────────────────────
    trajectory["session"] = session_name
    dt = trajectory["abs_time_sec"].diff().fillna(0)
    trajectory["gap_flag"] = dt > GAP_FLAG_SEC

    trajectory["centroid_wx"] = trajectory.get(
        "hip_wx", pd.Series(np.nan, index=trajectory.index)
    )
    trajectory["centroid_wy"] = trajectory.get(
        "hip_wy", pd.Series(np.nan, index=trajectory.index)
    )

    # ── 11. Save ─────────────────────────────────────────────────────────────
    out_parquet = output_dir / f"{session_name}_stitched_trajectory.parquet"
    out_csv     = output_dir / f"{session_name}_stitched_trajectory.csv"

    trajectory.to_parquet(str(out_parquet), index=False)
    trajectory.to_csv(str(out_csv), index=False)

    if verbose:
        dur  = trajectory["abs_time_sec"].max() - trajectory["abs_time_sec"].min()
        nval = int(trajectory["valid"].sum()) if "valid" in trajectory.columns else len(trajectory)
        ngap = int(trajectory["gap_flag"].sum())
        print(f"\n✓ Stitched trajectory saved")
        print(f"  Frames    : {len(trajectory)}")
        print(f"  Duration  : {dur:.1f}s  ({dur / 60:.1f} min)")
        print(f"  Confident : {nval}/{len(trajectory)} "
              f"({100 * nval / max(len(trajectory), 1):.0f}%)")
        print(f"  Gap flags : {ngap}")
        print(f"  Parquet   : {out_parquet}")

    return trajectory

"""
src/calibration/homography.py
==============================
Pixel → world-coordinate mapping for top-down multi-camera setups.

Each camera's 3×3 homography matrix H is computed once from 4 manually
selected corner correspondences and saved to Drive.  All downstream
modules call pixel_to_world() / pixel_array_to_world() rather than
touching the matrix directly.

World coordinate system (linear-chain layout)
----------------------------------------------
  X axis  : direction of travel  (Camera 1 → Camera 4)
  Y axis  : perpendicular to travel (camera "width" direction)
  Origin  : far corner of Camera 1's chamber (top-left as seen from above)
  Units   : centimetres (matching what the user measures with a tape)

Typical file layout on Drive
-----------------------------
  {DRIVE_ROOT}/calibration/
      H_cam1.npy
      H_cam2.npy
      H_cam3.npy
      H_cam4.npy
      camera_layout.json    ← world bounding box for each camera
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_homography(
    pixel_pts: np.ndarray,
    world_pts: np.ndarray,
) -> np.ndarray:
    """
    Compute a 3×3 homography from ≥4 point correspondences.

    Parameters
    ----------
    pixel_pts : (N, 2) float32   image (u, v) coordinates
    world_pts : (N, 2) float32   world (X, Y) coordinates in cm

    Returns
    -------
    H : (3, 3) float64
        Such that  [X, Y, 1]^T  ~  H @ [u, v, 1]^T
    """
    pixel_pts = np.float32(pixel_pts)
    world_pts = np.float32(world_pts)
    if len(pixel_pts) < 4:
        raise ValueError("Need at least 4 point pairs to compute a homography.")
    H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError(
            "cv2.findHomography returned None — points may be collinear or too few."
        )
    inliers = int(mask.sum()) if mask is not None else len(pixel_pts)
    print(f"  Homography: {inliers}/{len(pixel_pts)} inliers (RANSAC)")
    return H


def pixel_to_world(u: float, v: float, H: np.ndarray) -> tuple[float, float]:
    """Transform one pixel (u, v) to world (X, Y)."""
    pt = np.float32([[[u, v]]])
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def world_to_pixel(X: float, Y: float, H: np.ndarray) -> tuple[float, float]:
    """Inverse: world (X, Y) → pixel (u, v)."""
    H_inv = np.linalg.inv(H)
    pt = np.float32([[[X, Y]]])
    out = cv2.perspectiveTransform(pt, H_inv)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def pixel_array_to_world(uv: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Vectorised pixel → world transform; NaN rows pass through as NaN.

    Parameters
    ----------
    uv : (N, 2) float array of (u, v) pixels (may contain NaN)
    H  : (3, 3) homography matrix

    Returns
    -------
    xy : (N, 2) array of (X, Y) world coordinates
    """
    uv = np.asarray(uv, dtype=float)
    valid = ~np.any(np.isnan(uv), axis=1)
    result = np.full_like(uv, np.nan)
    if valid.any():
        pts = uv[valid].astype(np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H)
        result[valid] = transformed.reshape(-1, 2)
    return result


def reprojection_error(
    pixel_pts: np.ndarray,
    world_pts: np.ndarray,
    H: np.ndarray,
) -> float:
    """Mean pixel reprojection error — lower is better (< 5 px is good)."""
    predicted = pixel_array_to_world(pixel_pts, H)
    err = np.sqrt(((predicted - world_pts) ** 2).sum(axis=1))
    return float(err.mean())


# ── I/O ──────────────────────────────────────────────────────────────────────

def save_homography(H: np.ndarray, camera: int, output_dir: str | Path) -> Path:
    """Save a camera's homography matrix as H_cam{N}.npy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"H_cam{camera}.npy"
    np.save(str(path), H)
    return path


def load_homography(camera: int, homography_dir: str | Path) -> np.ndarray:
    """Load H_cam{N}.npy; raises FileNotFoundError with a helpful message."""
    path = Path(homography_dir) / f"H_cam{camera}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Homography for camera {camera} not found at {path}.\n"
            "Run notebook 02c_Spatial_Calibration.ipynb first."
        )
    return np.load(str(path))


def load_all_homographies(
    homography_dir: str | Path,
    cameras: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Load all available H_cam{N}.npy files; skips missing cameras silently."""
    homography_dir = Path(homography_dir)
    cameras = cameras or [1, 2, 3, 4]
    result: dict[int, np.ndarray] = {}
    for cam in cameras:
        p = homography_dir / f"H_cam{cam}.npy"
        if p.exists():
            result[cam] = np.load(str(p))
    return result


# ── World-space layout helpers ────────────────────────────────────────────────

def build_world_layout(
    chamber_dims: dict[int, dict],
    overlaps_cm: dict[tuple[int, int], float],
    camera_order: list[int] | None = None,      # ← new parameter
) -> list[dict]:
    """
    camera_order : physical left-to-right sequence of camera IDs.
                   Defaults to ascending sort if not provided.
                   Example for a 4→3→2→1 chain: [4, 3, 2, 1]
    """
    cameras  = camera_order if camera_order is not None else sorted(chamber_dims.keys())
    layout   = []
    x_cursor = 0.0

    for i, cam in enumerate(cameras):
        dims = chamber_dims[cam]
        w    = dims["width_cm"]
        h    = dims["height_cm"]

        x_min = x_cursor
        x_max = x_cursor + w

        layout.append({
            "camera":  cam,
            "x_min":   round(x_min, 4),
            "x_max":   round(x_max, 4),
            "y_min":   0.0,
            "y_max":   round(h, 4),
            "world_corners": [
                (x_min, 0.0),   # top-left
                (x_max, 0.0),   # top-right
                (x_max, h),     # bottom-right
                (x_min, h),     # bottom-left
            ],
        })

        if i < len(cameras) - 1:
            next_cam = cameras[i + 1]
            # Look up overlap in both directions
            ov = overlaps_cm.get(
                (cam, next_cam),
                overlaps_cm.get((next_cam, cam), 0.0)
            )
            x_cursor += w - ov

    return layout


def save_camera_layout(layout: list[dict], output_dir: str | Path) -> Path:
    """Persist the world layout to camera_layout.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "camera_layout.json"
    with open(path, "w") as f:
        json.dump(layout, f, indent=2)
    return path


def load_camera_layout(homography_dir: str | Path) -> list[dict]:
    """Load camera_layout.json; raises FileNotFoundError with guidance."""
    path = Path(homography_dir) / "camera_layout.json"
    if not path.exists():
        raise FileNotFoundError(
            f"camera_layout.json not found at {path}.\n"
            "Run notebook 02c_Spatial_Calibration.ipynb first."
        )
    with open(path) as f:
        data = json.load(f)
    # Restore tuples for world_corners (JSON stores as lists)
    for entry in data:
        if "world_corners" in entry:
            entry["world_corners"] = [tuple(c) for c in entry["world_corners"]]
    return data


def compute_overlap_windows(layout: list[dict]) -> list[dict]:
    """
    Identify world-X overlap ranges between adjacent cameras.

    Returns list of {cam_a, cam_b, x_start, x_end} dicts.
    """
    windows = []
    for i in range(len(layout) - 1):
        a = layout[i]
        b = layout[i + 1]
        x_start = max(a["x_min"], b["x_min"])
        x_end   = min(a["x_max"], b["x_max"])
        if x_end > x_start:
            windows.append({
                "cam_a":   a["camera"],
                "cam_b":   b["camera"],
                "x_start": round(x_start, 4),
                "x_end":   round(x_end, 4),
                "width_cm": round(x_end - x_start, 4),
            })
    return windows

"""
src/export/bvh_writer.py
=========================
Export a stitched trajectory to BVH (BioVision Hierarchy) format.

Coordinate convention
---------------------
BVH uses a Y-up right-handed coordinate system.  Our top-down world
coordinates map as follows:

    BVH Xposition = world_X   (direction of travel)
    BVH Yposition = 0         (height — unobserved in top-down view)
    BVH Zposition = world_Y   (lateral direction)

Skeleton hierarchy
------------------
    ROOT hips
    ├── JOINT spine           (= mid_back keypoint)
    │   └── JOINT chest       (= shoulders keypoint)
    │       └── JOINT neck    (= neck keypoint)
    │           ├── JOINT head  (= snout direction; End Site = snout tip)
    │           ├── JOINT left_ear   (End Site)
    │           └── JOINT right_ear  (End Site)
    └── JOINT tail            (= tail_base keypoint, End Site)

Rotation encoding
-----------------
Only yaw (Y-axis rotation) is written — pitch and roll are unobservable
from a single top-down camera.  Every joint's frame data is:

    Zrotation  Xrotation  Yrotation
       0           0       <angle>

The angle at each joint is the *relative* turn from the parent segment
direction to the child segment direction (in degrees, right-hand rule
around +Y axis).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ── Keypoint names matching the stitched-trajectory DataFrame ─────────────────

KP = [
    "snout", "left_ear", "right_ear", "neck",
    "shoulders", "mid_back", "hip", "tail_base",
]

# Default T-pose bone offsets along the animal's X-axis (in world units = cm).
# These are refined at runtime from the actual trajectory data.
_DEFAULT_LENGTHS = {
    "hip_to_spine":    12.0,   # hips → mid_back
    "spine_to_chest":  12.0,   # mid_back → shoulders
    "chest_to_neck":    8.0,   # shoulders → neck
    "neck_to_snout":   10.0,   # neck → snout tip
    "neck_to_ear":      7.0,   # neck → ear (along axis)
    "ear_lateral":      5.0,   # ear offset perpendicular to axis
    "hip_to_tail":     10.0,   # hips → tail_base (backwards)
}


# ── bone-length estimation ────────────────────────────────────────────────────

def _median_dist(df: pd.DataFrame, kp_a: str, kp_b: str) -> float:
    """Median Euclidean world-distance between two keypoints."""
    ax = df.get(f"{kp_a}_wx", pd.Series(dtype=float)).values
    ay = df.get(f"{kp_a}_wy", pd.Series(dtype=float)).values
    bx = df.get(f"{kp_b}_wx", pd.Series(dtype=float)).values
    by = df.get(f"{kp_b}_wy", pd.Series(dtype=float)).values
    d  = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    ok = d[~np.isnan(d)]
    return float(np.median(ok)) if len(ok) > 0 else 10.0


def estimate_bone_lengths(df: pd.DataFrame) -> dict[str, float]:
    """
    Estimate T-pose segment lengths from the median inter-keypoint distance
    in the stitched trajectory.  Falls back to _DEFAULT_LENGTHS where data
    is missing.
    """
    bl = dict(_DEFAULT_LENGTHS)

    def _safe(kp_a: str, kp_b: str, key: str) -> None:
        v = _median_dist(df, kp_a, kp_b)
        if v > 0.5:   # ignore suspiciously tiny estimates
            bl[key] = round(v, 2)

    _safe("hip",       "mid_back",   "hip_to_spine")
    _safe("mid_back",  "shoulders",  "spine_to_chest")
    _safe("shoulders", "neck",       "chest_to_neck")
    _safe("neck",      "snout",      "neck_to_snout")
    _safe("neck",      "left_ear",   "neck_to_ear")
    _safe("neck",      "right_ear",  "neck_to_ear")      # take the larger of both ears
    _safe("hip",       "tail_base",  "hip_to_tail")

    # Lateral ear offset ≈ perpendicular component of neck→ear vector
    bl["ear_lateral"] = round(bl["neck_to_ear"] * 0.6, 2)

    return bl


# ── BVH HIERARCHY section ─────────────────────────────────────────────────────

def _hierarchy(bl: dict[str, float]) -> str:
    t = "\t"
    lines = [
        "HIERARCHY",
        "ROOT hips",
        "{",
        f"{t}OFFSET 0.00 0.00 0.00",
        f"{t}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
        "",
        f"{t}JOINT spine",
        f"{t}{{",
        f"{t}{t}OFFSET {bl['hip_to_spine']:.4f} 0.0000 0.0000",
        f"{t}{t}CHANNELS 3 Zrotation Xrotation Yrotation",
        "",
        f"{t}{t}JOINT chest",
        f"{t}{t}{{",
        f"{t}{t}{t}OFFSET {bl['spine_to_chest']:.4f} 0.0000 0.0000",
        f"{t}{t}{t}CHANNELS 3 Zrotation Xrotation Yrotation",
        "",
        f"{t}{t}{t}JOINT neck",
        f"{t}{t}{t}{{",
        f"{t}{t}{t}{t}OFFSET {bl['chest_to_neck']:.4f} 0.0000 0.0000",
        f"{t}{t}{t}{t}CHANNELS 3 Zrotation Xrotation Yrotation",
        "",
        f"{t}{t}{t}{t}JOINT head",
        f"{t}{t}{t}{t}{{",
        f"{t}{t}{t}{t}{t}OFFSET {bl['neck_to_snout']:.4f} 0.0000 0.0000",
        f"{t}{t}{t}{t}{t}CHANNELS 3 Zrotation Xrotation Yrotation",
        f"{t}{t}{t}{t}{t}End Site",
        f"{t}{t}{t}{t}{t}{{",
        f"{t}{t}{t}{t}{t}{t}OFFSET 0.0000 0.0000 0.0000",
        f"{t}{t}{t}{t}{t}}}",
        f"{t}{t}{t}{t}}}",
        "",
        f"{t}{t}{t}{t}JOINT left_ear",
        f"{t}{t}{t}{t}{{",
        f"{t}{t}{t}{t}{t}OFFSET {bl['neck_to_ear']:.4f} 0.0000 {bl['ear_lateral']:.4f}",
        f"{t}{t}{t}{t}{t}CHANNELS 3 Zrotation Xrotation Yrotation",
        f"{t}{t}{t}{t}{t}End Site",
        f"{t}{t}{t}{t}{t}{{",
        f"{t}{t}{t}{t}{t}{t}OFFSET 0.0000 0.0000 0.0000",
        f"{t}{t}{t}{t}{t}}}",
        f"{t}{t}{t}{t}}}",
        "",
        f"{t}{t}{t}{t}JOINT right_ear",
        f"{t}{t}{t}{t}{{",
        f"{t}{t}{t}{t}{t}OFFSET {bl['neck_to_ear']:.4f} 0.0000 -{bl['ear_lateral']:.4f}",
        f"{t}{t}{t}{t}{t}CHANNELS 3 Zrotation Xrotation Yrotation",
        f"{t}{t}{t}{t}{t}End Site",
        f"{t}{t}{t}{t}{t}{{",
        f"{t}{t}{t}{t}{t}{t}OFFSET 0.0000 0.0000 0.0000",
        f"{t}{t}{t}{t}{t}}}",
        f"{t}{t}{t}{t}}}",
        f"{t}{t}{t}}}",
        f"{t}{t}}}",
        f"{t}}}",
        "",
        f"{t}JOINT tail",
        f"{t}{{",
        f"{t}{t}OFFSET -{bl['hip_to_tail']:.4f} 0.0000 0.0000",
        f"{t}{t}CHANNELS 3 Zrotation Xrotation Yrotation",
        f"{t}{t}End Site",
        f"{t}{t}{{",
        f"{t}{t}{t}OFFSET -{bl['hip_to_tail']:.4f} 0.0000 0.0000",
        f"{t}{t}}}",
        f"{t}}}",
        "}",
    ]
    return "\n".join(lines)


# ── per-frame rotation computation ───────────────────────────────────────────

def _wrap(angle: float) -> float:
    """Wrap to (−180, 180]."""
    return ((angle + 180.0) % 360.0) - 180.0


def _dir_deg(dx: float, dz: float, fallback: float = 0.0) -> float:
    """
    Y-axis heading from (+X direction) given a world displacement (dx, dz).
    Returns degrees in the right-hand-rule convention used by BVH.
    """
    if math.isnan(dx) or math.isnan(dz):
        return fallback
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return fallback
    # atan2 in BVH XZ plane: rotate CCW from +X = positive Y rotation
    return math.degrees(math.atan2(-dz, dx))   # −dz because BVH Z points backward


def _get(row: pd.Series, kp: str) -> tuple[float, float]:
    """Return (wx, wy) for a keypoint; NaN if missing."""
    return (
        float(row.get(f"{kp}_wx", math.nan)),
        float(row.get(f"{kp}_wy", math.nan)),
    )


def _frame_channels(row: pd.Series, prev_root_heading: float) -> list[float]:
    """
    Compute all BVH channel values for one trajectory frame.

    Channel order (24 values total):
      hips      : Xpos Ypos Zpos Zrot Xrot Yrot   (6)
      spine     : Zrot Xrot Yrot                    (3)
      chest     : Zrot Xrot Yrot                    (3)
      neck      : Zrot Xrot Yrot                    (3)
      head      : Zrot Xrot Yrot                    (3)
      left_ear  : Zrot Xrot Yrot                    (3)
      right_ear : Zrot Xrot Yrot                    (3)
      tail      : Zrot Xrot Yrot                    (3)
    """
    nan = math.nan

    hx, hy   = _get(row, "hip")
    mbx, mby = _get(row, "mid_back")
    shx, shy = _get(row, "shoulders")
    nx, ny   = _get(row, "neck")
    sx, sy   = _get(row, "snout")
    tx, ty   = _get(row, "tail_base")

    # Root position (BVH Y-up: X=world_x, Y=0, Z=world_y)
    bvh_x = hx if not math.isnan(hx) else 0.0
    bvh_z = hy if not math.isnan(hy) else 0.0   # world Y → BVH Z

    # Root heading: direction hip → mid_back in world XY plane
    root_hdg = _dir_deg(mbx - hx, mby - hy, prev_root_heading)

    # Spine (mid_back relative to hips direction)
    spine_abs = _dir_deg(mbx - hx, mby - hy, root_hdg)
    spine_rel = _wrap(spine_abs - root_hdg)

    # Chest (shoulders relative to spine direction)
    chest_abs = _dir_deg(shx - mbx, shy - mby, spine_abs)
    chest_rel = _wrap(chest_abs - spine_abs)

    # Neck (neck relative to chest direction)
    neck_abs = _dir_deg(nx - shx, ny - shy, chest_abs)
    neck_rel = _wrap(neck_abs - chest_abs)

    # Head (snout relative to neck direction)
    head_abs = _dir_deg(sx - nx, sy - ny, neck_abs)
    head_rel = _wrap(head_abs - neck_abs)

    # Ears: we keep them at their T-pose offset (rotation = 0)
    lear_rel  = 0.0
    rear_rel  = 0.0

    # Tail (tail_base relative to hips, pointing backward in rest pose)
    tail_abs = _dir_deg(tx - hx, ty - hy, root_hdg + 180.0)
    tail_rel = _wrap(tail_abs - (root_hdg + 180.0))

    return [
        # hips: Xpos Ypos Zpos Zrot Xrot Yrot
        bvh_x, 0.0, bvh_z, 0.0, 0.0, root_hdg,
        # spine
        0.0, 0.0, spine_rel,
        # chest
        0.0, 0.0, chest_rel,
        # neck
        0.0, 0.0, neck_rel,
        # head
        0.0, 0.0, head_rel,
        # left_ear
        0.0, 0.0, lear_rel,
        # right_ear
        0.0, 0.0, rear_rel,
        # tail
        0.0, 0.0, tail_rel,
    ]


# ── public API ────────────────────────────────────────────────────────────────

def write_bvh(
    trajectory:  pd.DataFrame,
    output_path: str | Path,
    fps:         float = 10.0,
    verbose:     bool  = True,
) -> Path:
    """
    Convert a stitched trajectory DataFrame to a .bvh file.

    Parameters
    ----------
    trajectory  : output of stitch.stitch_session()
    output_path : destination file path (will be created)
    fps         : playback frame rate written to MOTION header
    verbose     : print progress info

    Returns
    -------
    Path of the written .bvh file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Estimate segment lengths from data
    bl = estimate_bone_lengths(trajectory)
    if verbose:
        print("Bone lengths estimated from data (cm):")
        for k, v in bl.items():
            print(f"  {k}: {v:.1f}")

    # HIERARCHY
    hier = _hierarchy(bl)

    # MOTION
    n_frames   = len(trajectory)
    frame_time = 1.0 / fps

    motion_header = ["MOTION", f"Frames: {n_frames}", f"Frame Time: {frame_time:.8f}"]
    frame_lines   = []

    prev_hdg = 0.0
    for _, row in trajectory.iterrows():
        vals = _frame_channels(row, prev_hdg)
        # Update prev_hdg (index 5 = root Yrotation)
        if not math.isnan(vals[5]):
            prev_hdg = vals[5]
        frame_lines.append(
            " ".join(f"{v:.6f}" if not math.isnan(v) else "0.000000" for v in vals)
        )

    bvh_text = hier + "\n" + "\n".join(motion_header + frame_lines) + "\n"
    output_path.write_text(bvh_text, encoding="utf-8")

    if verbose:
        sz_kb = output_path.stat().st_size / 1024
        dur   = n_frames / fps
        print(f"\n✓ BVH written: {output_path}")
        print(f"  Joints : 8 (hips → head → ears + tail)")
        print(f"  Frames : {n_frames}  @  {fps:.2f} fps  ({dur:.1f}s)")
        print(f"  Size   : {sz_kb:.0f} KB")

    return output_path

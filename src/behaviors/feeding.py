import pandas as pd
import numpy as np


FEEDER_DISTANCE_THRESHOLD = 50.0
HEADING_ALIGNMENT_TOLERANCE = np.pi / 4
MIN_FEEDING_DURATION_SEC = 2.0


def detect_feeding_events(
    df: pd.DataFrame,
    feeder_x: float,
    feeder_y: float,
    fps: float,
    distance_threshold: float = FEEDER_DISTANCE_THRESHOLD,
    angle_tolerance: float = HEADING_ALIGNMENT_TOLERANCE,
    min_duration_sec: float = MIN_FEEDING_DURATION_SEC,
) -> pd.DataFrame:
    snout_x = df.get("snout_x")
    snout_y = df.get("snout_y")
    heading = df.get("heading_rad")
    if snout_x is None or snout_y is None or heading is None:
        return pd.DataFrame(columns=["start_frame", "end_frame", "start_time", "end_time", "duration_sec"])
    dist = np.sqrt((snout_x - feeder_x) ** 2 + (snout_y - feeder_y) ** 2)
    angle_to_feeder = np.arctan2(feeder_y - snout_y, feeder_x - snout_x)
    angle_diff = np.abs((heading - angle_to_feeder + np.pi) % (2 * np.pi) - np.pi)
    is_feeding = (dist < distance_threshold) & (angle_diff < angle_tolerance)
    changes = np.diff(is_feeding.astype(int), prepend=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(starts) == 0:
        return pd.DataFrame(columns=["start_frame", "end_frame", "start_time", "end_time", "duration_sec"])
    if len(ends) < len(starts):
        ends = np.append(ends, len(is_feeding) - 1)
    events = []
    for s, e in zip(starts, ends):
        duration = (e - s) / fps
        if duration >= min_duration_sec:
            events.append(
                {
                    "start_frame": int(s),
                    "end_frame": int(e),
                    "start_time": round(s / fps, 2),
                    "end_time": round(e / fps, 2),
                    "duration_sec": round(duration, 2),
                }
            )
    return pd.DataFrame(events)


def compute_feeding_summary(feeding_events: pd.DataFrame, total_duration_sec: float) -> dict:
    if feeding_events.empty:
        return {
            "total_feeding_events": 0,
            "total_feeding_duration_sec": 0.0,
            "feeding_pct": 0.0,
            "mean_bout_duration_sec": 0.0,
        }
    total = feeding_events["duration_sec"].sum()
    return {
        "total_feeding_events": len(feeding_events),
        "total_feeding_duration_sec": round(total, 2),
        "feeding_pct": round(total / total_duration_sec * 100, 2) if total_duration_sec > 0 else 0.0,
        "mean_bout_duration_sec": round(feeding_events["duration_sec"].mean(), 2),
    }

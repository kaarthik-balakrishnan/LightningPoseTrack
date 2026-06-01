import pandas as pd
import numpy as np


def compute_heading(df: pd.DataFrame) -> pd.Series:
    snout_x = df.get("snout_x")
    snout_y = df.get("snout_y")
    hip_x = df.get("hip_x")
    hip_y = df.get("hip_y")
    if any(c is None for c in [snout_x, snout_y, hip_x, hip_y]):
        return pd.Series(index=df.index, dtype=float)
    dx = snout_x - hip_x
    dy = snout_y - hip_y
    return np.arctan2(dy, dx)


def compute_angular_velocity(heading: pd.Series, fps: float) -> pd.Series:
    dt = 1.0 / fps
    diff = heading.diff()
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff / dt


def extract_orientation(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["heading_rad"] = compute_heading(df)
    features["heading_deg"] = np.degrees(features["heading_rad"])
    features["angular_velocity"] = compute_angular_velocity(features["heading_rad"], fps)
    features["turning_rate"] = features["angular_velocity"].abs()
    features["turning_rate_smooth"] = features["turning_rate"].rolling(window=5, center=True, min_periods=1).mean()
    return features

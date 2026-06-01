import pandas as pd
import numpy as np


KEYPOINT_NAMES = [
    "snout", "left_ear", "right_ear", "neck",
    "shoulders", "mid_back", "hip", "tail_base",
]


def compute_speed(series: pd.Series, fps: float) -> pd.Series:
    dt = 1.0 / fps
    return series.diff() / dt


def compute_acceleration(series: pd.Series, fps: float) -> pd.Series:
    dt = 1.0 / fps
    return series.diff().diff() / (dt ** 2)


def extract_kinematics(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for kp in KEYPOINT_NAMES:
        x_col = f"{kp}_x"
        y_col = f"{kp}_y"
        if x_col not in df.columns or y_col not in df.columns:
            continue
        pos = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2)
        vel = np.sqrt(compute_speed(df[x_col], fps) ** 2 + compute_speed(df[y_col], fps) ** 2)
        accel = np.sqrt(
            compute_acceleration(df[x_col], fps) ** 2
            + compute_acceleration(df[y_col], fps) ** 2
        )
        features[f"{kp}_speed"] = vel
        features[f"{kp}_acceleration"] = accel
    features["centroid_x"] = df[[c for c in df.columns if c.endswith("_x")]].mean(axis=1)
    features["centroid_y"] = df[[c for c in df.columns if c.endswith("_y")]].mean(axis=1)
    features["centroid_speed"] = np.sqrt(
        compute_speed(features["centroid_x"], fps) ** 2
        + compute_speed(features["centroid_y"], fps) ** 2
    )
    features["centroid_acceleration"] = np.sqrt(
        compute_acceleration(features["centroid_x"], fps) ** 2
        + compute_acceleration(features["centroid_y"], fps) ** 2
    )
    return features

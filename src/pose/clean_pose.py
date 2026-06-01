import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


KEYPOINT_NAMES = [
    "snout", "left_ear", "right_ear", "neck",
    "shoulders", "mid_back", "hip", "tail_base",
]


def clean_pose_df(
    df: pd.DataFrame,
    likelihood_col: str = "likelihood",
    likelihood_threshold: float = 0.5,
    window_length: int = 7,
    polyorder: int = 2,
    max_consecutive_gap: int = 5,
) -> pd.DataFrame:
    result = df.copy()
    for kp in KEYPOINT_NAMES:
        x_col = f"{kp}_x"
        y_col = f"{kp}_y"
        lh_col = f"{kp}_{likelihood_col}"
        if x_col not in result.columns or y_col not in result.columns:
            continue
        low_conf = result[lh_col] < likelihood_threshold
        result.loc[low_conf, x_col] = np.nan
        result.loc[low_conf, y_col] = np.nan
        result[x_col] = interpolate_gaps(result[x_col], max_gap=max_consecutive_gap)
        result[y_col] = interpolate_gaps(result[y_col], max_gap=max_consecutive_gap)
        if result[x_col].notna().sum() > window_length:
            result[x_col] = savgol_filter(
                result[x_col].ffill().bfill(),
                window_length=min(window_length, len(result) | 1),
                polyorder=polyorder,
            )
        if result[y_col].notna().sum() > window_length:
            result[y_col] = savgol_filter(
                result[y_col].ffill().bfill(),
                window_length=min(window_length, len(result) | 1),
                polyorder=polyorder,
            )
    return result


def interpolate_gaps(series: pd.Series, max_gap: int = 5) -> pd.Series:
    result = series.copy()
    is_nan = result.isna()
    if is_nan.sum() == 0:
        return result
    gaps = is_nan.astype(int).groupby((~is_nan).cumsum()).sum()
    long_gaps = gaps[gaps > max_gap].index
    for g in long_gaps:
        mask = is_nan & (g == (~is_nan).cumsum())
        result[mask] = 0.0
    result = result.interpolate(method="linear", limit=max_gap)
    result = result.ffill().bfill()
    return result

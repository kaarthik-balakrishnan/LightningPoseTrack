import pandas as pd
import numpy as np


def distance_to_point(x: pd.Series, y: pd.Series, px: float, py: float) -> pd.Series:
    return np.sqrt((x - px) ** 2 + (y - py) ** 2)


def zone_occupancy(x: pd.Series, y: pd.Series, x_min: float, x_max: float, y_min: float, y_max: float) -> pd.Series:
    return ((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)).astype(int)


def distance_to_walls(x: pd.Series, y: pd.Series, frame_width: float, frame_height: float) -> pd.DataFrame:
    result = pd.DataFrame(index=x.index)
    result["dist_left_wall"] = x
    result["dist_right_wall"] = frame_width - x
    result["dist_top_wall"] = y
    result["dist_bottom_wall"] = frame_height - y
    result["dist_nearest_wall"] = result[["dist_left_wall", "dist_right_wall", "dist_top_wall", "dist_bottom_wall"]].min(axis=1)
    return result


def extract_spatial(
    df: pd.DataFrame,
    feeder_position: tuple[float, float] | None = None,
    frame_size: tuple[float, float] | None = None,
    zones: list[dict] | None = None,
) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    centroid_x = df.get("centroid_x", df.get("snout_x"))
    centroid_y = df.get("centroid_y", df.get("snout_y"))
    snout_x = df.get("snout_x")
    snout_y = df.get("snout_y")
    if centroid_x is not None and centroid_y is not None:
        if feeder_position is not None:
            fx, fy = feeder_position
            features["dist_to_feeder"] = distance_to_point(centroid_x, centroid_y, fx, fy)
        if snout_x is not None and snout_y is not None and feeder_position is not None:
            fx, fy = feeder_position
            features["snout_dist_to_feeder"] = distance_to_point(snout_x, snout_y, fx, fy)
        if frame_size is not None:
            wall_dist = distance_to_walls(centroid_x, centroid_y, frame_size[0], frame_size[1])
            features = pd.concat([features, wall_dist], axis=1)
        if zones:
            for i, zone in enumerate(zones):
                occ = zone_occupancy(
                    centroid_x, centroid_y,
                    zone["x_min"], zone["x_max"],
                    zone["y_min"], zone["y_max"],
                )
                features[f"zone_{i}"] = occ
    return features

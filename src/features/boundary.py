import json
from pathlib import Path

import numpy as np


def inside_polygon(point, polygon):
    """Ray casting: returns True if (x, y) is inside the polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def filter_keypoints_by_boundary(
    df,
    polygon,
    keypoint_names,
    confidence_col_suffix="_likelihood",
    x_suffix="_x",
    y_suffix="_y",
    confidence_threshold=0.5,
):
    """Zero out keypoints outside the boundary polygon or below confidence."""
    out = df.copy()
    for kp in keypoint_names:
        cx = out[f"{kp}{x_suffix}"]
        cy = out[f"{kp}{y_suffix}"]
        conf = out[f"{kp}{confidence_col_suffix}"]
        inside = np.array(
            [inside_polygon((cx.iloc[i], cy.iloc[i]), polygon) for i in range(len(out))]
        )
        mask = ~inside | (conf < confidence_threshold)
        out.loc[mask, f"{kp}{x_suffix}"] = 0.0
        out.loc[mask, f"{kp}{y_suffix}"] = 0.0
        out.loc[mask, f"{kp}{confidence_col_suffix}"] = 0.0
    return out


def load_boundary_config(path="boundaries.json") -> dict:
    with open(path) as f:
        return json.load(f)


def save_boundary_config(config, path="boundaries.json"):
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def sample_boundary_config():
    """Return a template with placeholder polygons for cameras 1-4."""
    return {
        "cam1": {"polygon": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]},
        "cam2": {"polygon": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]},
        "cam3": {"polygon": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]},
        "cam4": {"polygon": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]},
    }

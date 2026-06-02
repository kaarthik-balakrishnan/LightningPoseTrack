import json
import csv
from pathlib import Path
import pandas as pd


KEYPOINT_NAMES = [
    "snout", "left_ear", "right_ear", "neck",
    "shoulders", "mid_back", "hip", "tail_base",
]


def convert_labelme_to_lp_csv(
    labelme_dir: str | Path,
    output_csv: str | Path,
    scorer: str = "LP",
) -> str:
    labelme_dir = Path(labelme_dir)
    records = []
    json_files = sorted(labelme_dir.rglob("*.json"))

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)
        image_path = data.get("imagePath", "")
        image_name = Path(image_path).name

        kp_map = {}
        for shape in data.get("shapes", []):
            label = shape.get("label", "").lower().replace(" ", "_")
            points = shape.get("points", [])
            if label in KEYPOINT_NAMES and len(points) > 0:
                pt = points[0]
                kp_map[label] = (pt[0], pt[1], 1.0)

        row = {"image": image_name}
        for kp in KEYPOINT_NAMES:
            if kp in kp_map:
                row[f"{kp}_x"] = kp_map[kp][0]
                row[f"{kp}_y"] = kp_map[kp][1]
            else:
                row[f"{kp}_x"] = 0.0
                row[f"{kp}_y"] = 0.0
        records.append(row)

    df = pd.DataFrame(records)
    df = df.set_index("image")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header_1 = [scorer] * (1 + len(KEYPOINT_NAMES) * 2)
        header_2 = ["image"] + [k for kp in KEYPOINT_NAMES for k in (kp, kp)]
        header_3 = [""] + ["x", "y"] * len(KEYPOINT_NAMES)
        writer.writerow(header_1)
        writer.writerow(header_2)
        writer.writerow(header_3)
        for img_name, row in df.iterrows():
            vals = [img_name]
            for kp in KEYPOINT_NAMES:
                vals.extend([row[f"{kp}_x"], row[f"{kp}_y"]])
            writer.writerow(vals)

    return str(output_csv)


def validate_labels(csv_path: str | Path) -> dict:
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    n_images = len(df)
    kp_set = set()
    for col in df.columns:
        if col[1] != "image":
            kp_set.add(col[1])
    labeled = len(kp_set & set(KEYPOINT_NAMES))
    missing = set(KEYPOINT_NAMES) - kp_set
    return {
        "total_images": n_images,
        "keypoints_labeled": labeled,
        "keypoints_missing": sorted(missing) if missing else None,
        "ready": len(missing) == 0,
    }

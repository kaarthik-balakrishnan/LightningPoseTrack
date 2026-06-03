#!/usr/bin/env python3
"""
Remove duplicate keypoint labels from LabelMe JSON files.

Usage:
    python src/pose/clean_json_labels.py /path/to/labeled_frames

Scans all .json files in the given directory (recursive).
For any file where a keypoint name appears more than once in "shapes",
keeps only the first occurrence and removes the rest.
"""

import json
import sys
from pathlib import Path


KEYPOINT_NAMES = {
    "snout", "left_ear", "right_ear", "neck",
    "shoulders", "mid_back", "hip", "tail_base",
}


def clean_json(filepath: Path, dry_run: bool = False) -> int:
    with open(filepath) as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    seen = {}
    cleaned = []
    removed = 0

    for shape in shapes:
        label = shape.get("label", "").lower().replace(" ", "_")
        if label in KEYPOINT_NAMES:
            if label in seen:
                removed += 1
                continue
            seen[label] = True
        cleaned.append(shape)

    if removed == 0:
        return 0

    if dry_run:
        print(f"  [DRY RUN] {filepath.name}: would remove {removed} duplicate(s): ", end="")
        dupes = []
        seen2 = set()
        for shape in shapes:
            label = shape.get("label", "").lower().replace(" ", "_")
            if label in KEYPOINT_NAMES:
                if label in seen2:
                    dupes.append(label)
                seen2.add(label)
        print(", ".join(dupes))
        return removed

    data["shapes"] = cleaned
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Fixed {filepath.name}: removed {removed} duplicate(s)")
    return removed


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        print(f"No .json files found in {root}")
        return

    print(f"Scanning {len(json_files)} JSON files in {root} ...")
    total_fixed = 0
    total_removed = 0

    for fp in json_files:
        n = clean_json(fp, dry_run=dry_run)
        if n > 0:
            total_fixed += 1
            total_removed += n

    if total_fixed == 0:
        print("All files clean — no duplicate labels found.")
    else:
        print(f"\nSummary: {total_fixed} file(s) had duplicates, {total_removed} duplicate label(s) removed.")
        if dry_run:
            print("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()

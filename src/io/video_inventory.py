import os
import re
import cv2
import pandas as pd
from pathlib import Path


def parse_camera_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    parts = re.split(r"[-_]+", stem)
    for part in reversed(parts):
        try:
            num = int(part)
            if 1 <= num <= 4:
                return num
        except ValueError:
            continue
    return 0


def scan_videos(
    root_dir: str | Path, verbose: bool = True
) -> pd.DataFrame:
    root = Path(root_dir)
    records = []
    video_extensions = {".asf", ".mp4", ".avi", ".mov", ".mkv"}
    found_files = 0
    opened_ok = 0
    opened_fail = 0
    no_ext = 0
    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        for filename in sorted(filenames):
            filepath = dirpath / filename
            if filepath.suffix.lower() not in video_extensions:
                no_ext += 1
                if verbose:
                    print(f"  Skipped (not video): {filename}")
                continue
            found_files += 1
            if verbose:
                print(f"  {filename}", end="")
            session = dirpath.name
            camera = parse_camera_from_filename(filename)
            cap = cv2.VideoCapture(str(filepath))
            if not cap.isOpened():
                opened_fail += 1
                if verbose:
                    print(" — FAILED to open")
                continue
            opened_ok += 1
            if verbose:
                print(" — OK")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0
            cap.release()
            records.append(
                {
                    "filename": filename,
                    "path": str(filepath.relative_to(root)),
                    "session": session,
                    "camera": camera,
                    "fps": round(fps, 2),
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "duration_sec": duration_sec,
                    "duration_min": round(duration_sec / 60, 2),
                }
            )
    if verbose:
        print(
            f"\nSummary: {found_files} video files, "
            f"{no_ext} non-video files, "
            f"{opened_ok} opened OK, {opened_fail} failed to open"
        )
    return pd.DataFrame(records)

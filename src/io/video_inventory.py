import cv2
import pandas as pd
from pathlib import Path


def parse_camera_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    parts = stem.split("_")
    for part in reversed(parts):
        try:
            num = int(part)
            if 1 <= num <= 4:
                return num
        except ValueError:
            continue
    return 0


def scan_videos(root_dir: str | Path) -> pd.DataFrame:
    root = Path(root_dir)
    records = []
    video_extensions = {".asf", ".mp4", ".avi", ".mov", ".mkv"}
    for video_path in sorted(root.rglob("*")):
        if video_path.suffix.lower() not in video_extensions:
            continue
        session = video_path.parent.name
        camera = parse_camera_from_filename(video_path.name)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0
        cap.release()
        records.append(
            {
                "filename": video_path.name,
                "path": str(video_path.relative_to(root)),
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
    return pd.DataFrame(records)

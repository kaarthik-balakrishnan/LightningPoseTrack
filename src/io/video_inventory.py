import os
import re
import json
import subprocess
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


def probe_video_metadata(video_path: str | Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    return json.loads(result.stdout)


def extract_recording_time(info: dict) -> str | None:
    fmt = info.get("format", {})
    tags = fmt.get("tags", {})
    for key in ("creation_time", "date", "DATE"):
        val = tags.get(key)
        if val:
            return str(val)
    for s in info.get("streams", []):
        stags = s.get("tags", {})
        for key in ("creation_time", "date", "DATE"):
            val = stags.get(key)
            if val:
                return str(val)
    return None


def extract_burned_timestamp(
    video_path: str | Path, frame_idx: int = 0
) -> str | None:
    try:
        from pytesseract import image_to_string
        import cv2
        import numpy as np
    except ImportError:
        return None
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bottom_path = Path(tmp) / "bottom.png"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vf", f"select='eq(n,{frame_idx})',crop=1920:200:0:880",
             "-vframes", "1", str(bottom_path)],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            return None
        gray = cv2.imread(str(bottom_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        texts = []
        for psm in (6, 7, 3):
            try:
                t = image_to_string(scaled, config=f"--psm {psm} --oem 3").strip()
                if t:
                    texts.append(t)
            except Exception:
                continue
        full = " | ".join(texts)
        for pat in (
            r"(\d{4})[/.-](\d{2})[/.-](\d{2})\s+(\d{2})[:.](\d{2})[:.](\d{2})",
            r"(\d{4})[/.-](\d{2})[/.-](\d{2})",
            r"(\d{2})[:.](\d{2})[:.](\d{2})",
        ):
            m = re.search(pat, full)
            if m:
                return m.group(0)
    return None


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
    if verbose:
        print(f"  Root path: {root}")
        print(f"  Path exists: {root.exists()}")
        if root.exists():
            entries = list(root.iterdir())[:20]
            print(f"  First 20 entries: {[e.name for e in entries]}")
        else:
            print(f"  Parent exists: {root.parent.exists()}")
            if root.parent.exists():
                print(f"  Parent contents: {[e.name for e in root.parent.iterdir()][:20]}")
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
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json",
                     "-show_format", "-show_streams", str(filepath)],
                    capture_output=True, text=True, timeout=30,
                )
                info = json.loads(result.stdout)
                video_stream = None
                for s in info.get("streams", []):
                    if s.get("codec_type") == "video":
                        video_stream = s
                        break
                if video_stream is None:
                    raise ValueError("no video stream")
                width = int(video_stream.get("width", 0))
                height = int(video_stream.get("height", 0))
                r_frame_rate = video_stream.get("r_frame_rate", "0/1")
                num, den = r_frame_rate.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 0.0
                nb_frames = video_stream.get("nb_frames")
                if nb_frames is None:
                    duration = float(info.get("format", {}).get("duration", 0))
                    nb_frames = int(duration * fps) if fps > 0 else 0
                else:
                    nb_frames = int(nb_frames)
                recording_time = extract_recording_time(info)
                if not recording_time:
                    recording_time = extract_burned_timestamp(filepath)
                opened_ok += 1
                if verbose:
                    codec = video_stream.get("codec_name", "?")
                    ts = f", time={recording_time}" if recording_time else ""
                    print(f" — OK ({codec}, {width}x{height}, {fps:.1f} fps, {nb_frames}f{ts})")
            except Exception as e:
                opened_fail += 1
                if verbose:
                    print(f" — FAILED ({e})")
                continue
            duration_sec = round(nb_frames / fps, 2) if fps > 0 else 0.0
            records.append(
                {
                    "filename": filename,
                    "path": str(filepath.relative_to(root)),
                    "session": session,
                    "camera": camera,
                    "fps": round(fps, 2),
                    "frame_count": nb_frames,
                    "width": width,
                    "height": height,
                    "duration_sec": duration_sec,
                    "duration_min": round(duration_sec / 60, 2),
                    "recording_time": recording_time or "",
                }
            )
    if verbose:
        print(
            f"\nSummary: {found_files} video files, "
            f"{no_ext} non-video files, "
            f"{opened_ok} opened OK, {opened_fail} failed to open"
        )
    return pd.DataFrame(records)

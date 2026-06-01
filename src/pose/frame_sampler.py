import json
import random
import subprocess
import cv2
import numpy as np
from pathlib import Path


def probe(video_path: str | Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    info = json.loads(result.stdout)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            w = int(s.get("width", 0))
            h = int(s.get("height", 0))
            rfr = s.get("r_frame_rate", "0/1")
            num, den = rfr.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 0.0
            nf = s.get("nb_frames")
            if nf is None:
                dur = float(info.get("format", {}).get("duration", 0))
                nf = int(dur * fps) if fps > 0 else 0
            else:
                nf = int(nf)
            return {"width": w, "height": h, "fps": fps, "frame_count": nf,
                    "codec": s.get("codec_name", "?")}
    raise ValueError("no video stream")


def stream_frames(video_path: str | Path, width: int, height: int):
    frame_size = width * height * 3
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    idx = 0
    try:
        while True:
            chunk = proc.stdout.read(frame_size)
            if len(chunk) < frame_size:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
            yield idx, frame
            idx += 1
    finally:
        proc.stdout.close()
        proc.wait()


def extract_frames(video_path: str | Path, width: int, height: int,
                   indices: list[int]) -> list[tuple[int, np.ndarray]]:
    if not indices:
        return []
    sel = "+".join(f"eq(n,{i})" for i in indices)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='{sel}'", "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    fs = width * height * 3
    frames = []
    for i in range(0, len(result.stdout), fs):
        chunk = result.stdout[i:i + fs]
        if len(chunk) == fs:
            frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3))
    return list(zip(indices[:len(frames)], frames))


def sample_uniform(video_path: str | Path, n_frames: int):
    try:
        info = probe(video_path)
    except Exception:
        return []
    total = info["frame_count"]
    if total == 0:
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
    try:
        return extract_frames(video_path, info["width"], info["height"], indices)
    except Exception:
        return []


def sample_random(video_path: str | Path, n_frames: int, seed: int = 42):
    random.seed(seed)
    try:
        info = probe(video_path)
    except Exception:
        return []
    total = info["frame_count"]
    if total == 0:
        return []
    indices = sorted(random.sample(range(total), min(n_frames, total)))
    try:
        return extract_frames(video_path, info["width"], info["height"], indices)
    except Exception:
        return []


def sample_motion_based(
    video_path: str | Path, n_frames: int, motion_threshold: float = 15.0
):
    raise NotImplementedError("Use sample_with_background_subtraction instead")


def compute_background(video_path: str | Path, sample_every_n: int = 100) -> np.ndarray | None:
    try:
        info = probe(video_path)
    except Exception:
        return None
    total = info["frame_count"]
    if total == 0:
        return None
    w, h = info["width"], info["height"]
    accum = []
    for idx, frame_bgr in stream_frames(video_path, w, h):
        if idx % sample_every_n == 0:
            accum.append(frame_bgr.astype(np.float32))
    if not accum:
        return None
    return np.median(np.stack(accum, axis=0), axis=0).astype(np.uint8)


def detect_animal(
    frame: np.ndarray,
    background: np.ndarray,
    threshold: int = 30,
    min_pixels: int = 500,
) -> tuple[bool, int]:
    diff = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    pixel_count = int(np.sum(thresh > 0))
    return pixel_count > min_pixels, pixel_count


def sample_with_background_subtraction(
    video_path: str | Path,
    n_frames: int,
    background: np.ndarray | None = None,
    threshold: int = 30,
    min_pixels: int = 500,
    sample_every_n: int = 100,
) -> list[tuple[int, np.ndarray]]:
    if background is None:
        background = compute_background(video_path, sample_every_n)
    if background is None:
        return []
    try:
        info = probe(video_path)
    except Exception:
        return []
    total = info["frame_count"]
    if total == 0:
        return []
    w, h = info["width"], info["height"]
    candidates = []
    for idx, frame_bgr in stream_frames(video_path, w, h):
        present, _ = detect_animal(frame_bgr, background, threshold, min_pixels)
        if present:
            candidates.append(idx)
    if not candidates:
        return []
    selected = sorted(random.sample(candidates, min(n_frames, len(candidates))))
    try:
        return extract_frames(video_path, w, h, selected)
    except Exception:
        return []


def save_frame(output_dir: str | Path, session: str, camera: int,
               video_stem: str, frame_idx: int, frame: np.ndarray):
    out_dir = Path(output_dir) / session
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{video_stem}_cam{camera}_frame{frame_idx:06d}.jpg"
    cv2.imwrite(str(out_dir / filename), frame)
    return filename

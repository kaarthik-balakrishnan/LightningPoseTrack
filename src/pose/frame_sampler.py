import cv2
import imageio
import random
import numpy as np
from pathlib import Path


def open_video(video_path: str | Path):
    reader = imageio.get_reader(str(video_path), format="ffmpeg")
    return reader


def frame_count(reader) -> int:
    try:
        return reader.count_frames()
    except Exception:
        return 0


def read_frame(reader, idx: int) -> np.ndarray:
    frame_rgb = reader.get_data(idx)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr


def compute_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def sample_uniform(video_path: str | Path, n_frames: int):
    reader = open_video(video_path)
    total = frame_count(reader)
    if total == 0:
        reader.close()
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
    frames = []
    for idx in indices:
        try:
            frame = read_frame(reader, idx)
            frames.append((idx, frame))
        except Exception:
            continue
    reader.close()
    return frames


def sample_random(video_path: str | Path, n_frames: int, seed: int = 42):
    random.seed(seed)
    reader = open_video(video_path)
    total = frame_count(reader)
    if total == 0:
        reader.close()
        return []
    indices = sorted(random.sample(range(total), min(n_frames, total)))
    frames = []
    for idx in indices:
        try:
            frame = read_frame(reader, idx)
            frames.append((idx, frame))
        except Exception:
            continue
    reader.close()
    return frames


def sample_motion_based(
    video_path: str | Path, n_frames: int, motion_threshold: float = 15.0
):
    reader = open_video(video_path)
    total = frame_count(reader)
    if total == 0:
        reader.close()
        return []
    scores = []
    prev_frame_rgb = None
    frame_idx = 0
    for frame_rgb in reader:
        if prev_frame_rgb is None:
            prev_frame_rgb = frame_rgb
            scores.append((frame_idx, 0.0))
            frame_idx += 1
            continue
        prev_gray = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        score = compute_motion_score(prev_gray, curr_gray)
        scores.append((frame_idx, score))
        prev_frame_rgb = frame_rgb
        frame_idx += 1
    reader.close()
    high_motion = [i for i, s in scores if s > motion_threshold]
    if not high_motion:
        high_motion = [i for i, _ in scores]
    selected = sorted(random.sample(high_motion, min(n_frames, len(high_motion))))
    reader = open_video(video_path)
    frames = []
    for idx in selected:
        try:
            frame = read_frame(reader, idx)
            frames.append((idx, frame))
        except Exception:
            continue
    reader.close()
    return frames


def compute_background(video_path: str | Path, sample_every_n: int = 100) -> np.ndarray | None:
    reader = open_video(video_path)
    total = frame_count(reader)
    if total == 0:
        reader.close()
        return None
    bg_frames = []
    frame_idx = 0
    for frame_rgb in reader:
        if frame_idx % sample_every_n == 0:
            bg_frames.append(frame_rgb.astype(np.float32))
        frame_idx += 1
    reader.close()
    if not bg_frames:
        return None
    background_rgb = np.median(np.stack(bg_frames, axis=0), axis=0).astype(np.uint8)
    background_bgr = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2BGR)
    return background_bgr


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
    reader = open_video(video_path)
    total = frame_count(reader)
    if total == 0:
        reader.close()
        return []
    candidates = []
    frame_idx = 0
    for frame_rgb in reader:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        present, _ = detect_animal(frame_bgr, background, threshold, min_pixels)
        if present:
            candidates.append(frame_idx)
        frame_idx += 1
    reader.close()
    if not candidates:
        return []
    selected = sorted(random.sample(candidates, min(n_frames, len(candidates))))
    reader = open_video(video_path)
    frames = []
    for idx in selected:
        try:
            frame = read_frame(reader, idx)
            frames.append((idx, frame))
        except Exception:
            continue
    reader.close()
    return frames


def save_frame(output_dir: str | Path, session: str, camera: int, video_stem: str, frame_idx: int, frame: np.ndarray):
    out_dir = Path(output_dir) / session
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{video_stem}_cam{camera}_frame{frame_idx:06d}.jpg"
    cv2.imwrite(str(out_dir / filename), frame)
    return filename

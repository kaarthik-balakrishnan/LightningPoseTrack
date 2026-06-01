import cv2
import random
import numpy as np
from pathlib import Path


def compute_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def sample_uniform(video_path: str | Path, n_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def sample_random(video_path: str | Path, n_frames: int, seed: int = 42):
    random.seed(seed)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = sorted(random.sample(range(total), min(n_frames, total)))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def sample_motion_based(
    video_path: str | Path, n_frames: int, motion_threshold: float = 15.0
):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    scores = []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 0
    scores.append((frame_idx, 0.0))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = compute_motion_score(prev_gray, gray)
        scores.append((frame_idx, score))
        prev_gray = gray
    cap.release()
    high_motion = [i for i, s in scores if s > motion_threshold]
    if not high_motion:
        high_motion = [i for i, _ in scores]
    selected = sorted(random.sample(high_motion, min(n_frames, len(high_motion))))
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def compute_background(video_path: str | Path, sample_every_n: int = 100) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n == 0:
            frames.append(frame.astype(np.float32))
        frame_idx += 1
    cap.release()
    if not frames:
        return None
    background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return background


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
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    candidates = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        present, _ = detect_animal(frame, background, threshold, min_pixels)
        if present:
            candidates.append(frame_idx)
        frame_idx += 1
    cap.release()
    if not candidates:
        return []
    selected = sorted(random.sample(candidates, min(n_frames, len(candidates))))
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def save_frame(output_dir: str | Path, session: str, camera: int, video_stem: str, frame_idx: int, frame: np.ndarray):
    out_dir = Path(output_dir) / session
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{video_stem}_cam{camera}_frame{frame_idx:06d}.jpg"
    cv2.imwrite(str(out_dir / filename), frame)
    return filename

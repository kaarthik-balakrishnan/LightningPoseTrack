"""
Cross-Camera Timing Calibration Pipeline
=========================================

Aligns video files from multiple cameras to a unified timeline.

Pipeline stages:
  1. scan()        — Discover video files, parse metadata
  2. probe()       — Count actual frames, compute accurate durations
  3. quality()     — Detect dropped frames, flag problematic files
  4. parse_timestamps() — Extract wall-clock times from filenames
  5. build_timelines()  — Build per-camera continuous timelines
  6. compute_motion()   — Compute motion energy signals for overlap windows
  7. synchronize()      — Cross-correlate motion to find camera offsets
  8. export()           — Save unified timeline

Usage:
    from src.calibration.pipeline import CalibrationPipeline

    pipe = CalibrationPipeline("/path/to/video/folder")
    pipe.run()
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


class CalibrationPipeline:
    """Orchestrates the cross-camera calibration pipeline."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.video_extensions = {".asf", ".mp4", ".avi", ".mov", ".mkv"}
        self.files: list[Path] = []
        self.results: list[dict] = []
        self.timelines: dict[int, list[dict]] = {}
        self.offsets: dict[str, float] = {}
        self.unified: list[dict] = []

        self._stage = 0
        self._total_stages = 8

    # ------------------------------------------------------------------
    # Stage 1: Scan
    # ------------------------------------------------------------------
    def scan(self) -> list[Path]:
        self._print_stage(1, "Scan", "Finding video files")
        self.files = sorted(
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in self.video_extensions
        )
        print(f"  Found {len(self.files)} video files in {self.root_dir}")
        for f in self.files:
            print(f"    {f.name}")
        print()
        return self.files

    # ------------------------------------------------------------------
    # Stage 2: Probe
    # ------------------------------------------------------------------
    def probe(self) -> list[dict]:
        self._print_stage(2, "Probe", "Reading metadata and counting frames")
        self.results = []
        for i, f in enumerate(self.files):
            print(f"  [{i+1}/{len(self.files)}] {f.name} ...", end=" ", flush=True)
            try:
                info = self._probe_file(f)
                self.results.append(info)
                print(f"OK — {info['actual_frames']}f @ {info['fps']} fps, "
                      f"{info['actual_duration_sec']:.1f}s, "
                      f"{info['size_mb']:.0f} MB")
            except Exception as e:
                print(f"FAILED — {e}")
                self.results.append({
                    "filename": f.name, "error": str(e),
                })
        print(f"  Probed {len(self.results)} files ({sum(1 for r in self.results if 'error' not in r)} OK)\n")
        return self.results

    def _probe_file(self, path: Path) -> dict:
        stem = path.stem
        camera = self._parse_camera(stem)

        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        meta = json.loads(r.stdout)

        video_stream = None
        for s in meta.get("streams", []):
            if s.get("codec_type") == "video":
                video_stream = s
                break
        if video_stream is None:
            raise ValueError("no video stream found")

        rfr = video_stream.get("r_frame_rate", "0/1")
        num, den = rfr.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 0.0
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec = video_stream.get("codec_name", "?")
        size_bytes = int(meta.get("format", {}).get("size", 0))
        meta_nb_frames = video_stream.get("nb_frames")
        meta_duration = float(meta.get("format", {}).get("duration", 0))

        if meta_nb_frames is not None:
            meta_nb_frames = int(meta_nb_frames)
        else:
            meta_nb_frames = int(meta_duration * fps) if fps > 0 else 0

        actual_frames = self._count_actual_frames(path)

        return {
            "filename": path.name,
            "path": str(path),
            "camera": camera,
            "fps": round(fps, 4),
            "width": width,
            "height": height,
            "codec": codec,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1024 / 1024, 1),
            "meta_nb_frames": meta_nb_frames,
            "meta_duration_sec": round(meta_duration, 3),
            "actual_frames": actual_frames,
            "actual_duration_sec": round(actual_frames / fps, 3) if fps > 0 else 0.0,
            "r_frame_rate": rfr,
        }

    def _parse_camera(self, stem: str) -> int:
        parts = re.split(r"[-_]+", stem)
        for part in reversed(parts):
            try:
                num = int(part)
                if 1 <= num <= 4:
                    return num
            except ValueError:
                continue
        return 0

    def _count_actual_frames(self, path: Path) -> int:
        """Count actual frames by decoding the video stream header.

        Uses ffprobe -count_frames which reads and counts every frame.
        This is the only reliable method for ASF files (which lack PTS).
        """
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames",
             "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        out = r.stdout.strip()
        if out and out != "N/A":
            return int(out)
        raise ValueError(f"could not count frames: {out}")

    # ------------------------------------------------------------------
    # Stage 3: Quality
    # ------------------------------------------------------------------
    def quality(self) -> list[dict]:
        self._print_stage(3, "Quality", "Checking for dropped frames and metadata accuracy")
        flagged = []
        for r in self.results:
            if "error" in r:
                flagged.append(r)
                continue
            meta_frames = r["meta_nb_frames"]
            actual_frames = r["actual_frames"]
            frames_ok = actual_frames == meta_frames
            if not frames_ok:
                diff = actual_frames - meta_frames
                print(f"  {r['filename']}: FRAME COUNT MISMATCH — "
                      f"meta={meta_frames}, actual={actual_frames} ({diff:+d})")
                r["frame_mismatch"] = True
                r["frame_diff"] = diff
                flagged.append(r)
            else:
                print(f"  {r['filename']}: OK — {actual_frames} frames matches metadata")
                r["frame_mismatch"] = False
                r["frame_diff"] = 0
        print(f"  Checked {len(self.results)} files, "
              f"{len(flagged)} flagged\n")
        return flagged

    # ------------------------------------------------------------------
    # Stage 4: Parse timestamps from filenames
    # ------------------------------------------------------------------
    def parse_timestamps(self) -> list[dict]:
        self._print_stage(4, "Timestamps", "Extracting wall-clock times from filenames")
        for r in self.results:
            if "error" in r:
                r["start_time_sec"] = None
                continue
            stem = Path(r["filename"]).stem
            ts = self._parse_file_timestamp(stem)
            if ts:
                h, m, s = ts
                r["start_time_sec"] = h * 3600 + m * 60 + s
                print(f"  {r['filename']}: {h:02d}:{m:02d}:{s:02d} "
                      f"(t={r['start_time_sec']}s)")
            else:
                r["start_time_sec"] = None
                print(f"  {r['filename']}: NO TIMESTAMP PARSED")
        print()
        return self.results

    def _parse_file_timestamp(self, stem: str) -> tuple[int, int, int] | None:
        m = re.match(r"(\d{2})(\d{2})(\d{2})", stem)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        return None

    # ------------------------------------------------------------------
    # Stage 5: Build per-camera timelines
    # ------------------------------------------------------------------
    def build_timelines(self) -> dict[int, list[dict]]:
        self._print_stage(5, "Timelines", "Building continuous per-camera timelines")
        cameras = sorted(set(
            r["camera"] for r in self.results if "error" not in r
        ))
        self.timelines = {}
        for cam in cameras:
            cam_files = [r for r in self.results
                         if r.get("camera") == cam and "error" not in r
                         and r.get("start_time_sec") is not None]
            cam_files.sort(key=lambda x: x["start_time_sec"])

            if not cam_files:
                continue

            base = cam_files[0]["start_time_sec"]
            timeline = []
            for r in cam_files:
                rel_start = r["start_time_sec"] - base
                dur = r["actual_duration_sec"]
                timeline.append({
                    "filename": r["filename"],
                    "camera": cam,
                    "fps": r["fps"],
                    "frame_count": r["actual_frames"],
                    "start_time_abs": r["start_time_sec"],
                    "start_time_rel": round(rel_start, 3),
                    "end_time_rel": round(rel_start + dur, 3),
                    "duration_sec": round(dur, 3),
                })
            self.timelines[cam] = timeline
            total = timeline[-1]["end_time_rel"]
            print(f"  Camera {cam}: {len(timeline)} files, "
                  f"0s → {total:.0f}s ({(total/60):.1f} min)")

        print(f"  Built timelines for {len(self.timelines)} cameras\n")
        return self.timelines

    # ------------------------------------------------------------------
    # Stage 6: Compute motion signals
    # ------------------------------------------------------------------
    def compute_motion(self, max_frames: int = 500):
        """Compute motion energy signals for overlapping time windows.

        For each camera, decode frames and compute frame-to-frame absolute
        difference (summed over all pixels) as a motion energy signal.
        Only processes the first `max_frames` frames per file for speed.

        The motion signal is stored in each camera's timeline entries.
        """
        self._print_stage(6, "Motion", "Computing frame-difference motion signals")
        for cam, tl in self.timelines.items():
            for entry in tl:
                path = Path(self.root_dir, entry["filename"])
                if not path.exists():
                    path = Path(entry["filename"])
                n_frames = min(entry["frame_count"], max_frames)
                print(f"  Camera {cam} — {entry['filename']}: "
                      f"decoding {n_frames} frames ...", end=" ", flush=True)
                try:
                    signal = self._motion_energy(path, n_frames)
                    entry["motion_signal"] = signal.tolist()
                    entry["motion_frames"] = len(signal)
                    entry["motion_duration_sec"] = round(len(signal) / entry["fps"], 3)
                    print(f"{len(signal)} diffs ({entry['motion_duration_sec']}s)")
                except Exception as e:
                    print(f"FAILED — {e}")
                    entry["motion_signal"] = []
        print()

    def _motion_energy(self, path: Path, max_frames: int) -> np.ndarray:
        """Decode video frames via ffmpeg pipe and compute frame-difference energy.

        Returns a 1D array of sum-of-absolute-differences between consecutive frames
        (downscaled 4x for speed).
        """
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", "480x270",  # 4x downscale from 1920x1080
            "-vframes", str(max_frames),
            "-",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        frame_size = 480 * 270  # 1 byte per pixel for gray
        raw = proc.stdout.read(frame_size * max_frames)
        proc.wait()

        n_read = len(raw) // frame_size
        if n_read < 2:
            return np.array([], dtype=np.float32)

        frames = np.frombuffer(raw, dtype=np.uint8).reshape(n_read, 270, 480)
        diffs = np.sum(np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)),
                       axis=(1, 2))
        return diffs

    # ------------------------------------------------------------------
    # Stage 7: Synchronize cameras
    # ------------------------------------------------------------------
    def synchronize(self):
        """Cross-correlate motion signals between camera pairs to find offsets.

        For each pair of cameras that have overlapping time windows with motion
        signals, compute the cross-correlation to find the temporal offset that
        best aligns them.
        """
        self._print_stage(7, "Sync", "Cross-correlating motion signals between cameras")
        if len(self.timelines) < 2:
            print("  Need at least 2 cameras to synchronize\n")
            return

        cameras = sorted(self.timelines.keys())
        self.offsets = {}

        for i, cam_a in enumerate(cameras):
            for cam_b in cameras[i + 1:]:
                offset = self._align_camera_pair(cam_a, cam_b)
                pair_key = f"{cam_a}_to_{cam_b}"
                if offset is not None:
                    self.offsets[pair_key] = round(offset, 3)
                    print(f"  Camera {cam_b} offset relative to Camera {cam_a}: "
                          f"{offset:.3f}s")
                else:
                    self.offsets[pair_key] = None
                    print(f"  Camera {cam_b} relative to Camera {cam_a}: "
                          f"NO OVERLAP / INSUFFICIENT DATA")
        print()

    def _align_camera_pair(self, cam_a: int, cam_b: int) -> float | None:
        """Find offset of cam_b relative to cam_a (cam_a's timeline is reference).

        Returns offset in seconds (positive means cam_b starts later than cam_a
        according to the cross-correlation).
        """
        sig_a, times_a = self._extract_overlap_signal(cam_a)
        sig_b, times_b = self._extract_overlap_signal(cam_b)

        if len(sig_a) < 10 or len(sig_b) < 10:
            return None

        # Resample to common time base at 10 Hz (0.1s intervals)
        common_t = np.arange(
            max(times_a[0], times_b[0]),
            min(times_a[-1], times_b[-1]),
            0.1,
        )
        if len(common_t) < 10:
            return None

        sig_a_r = np.interp(common_t, times_a, sig_a)
        sig_b_r = np.interp(common_t, times_b, sig_b)

        # Normalize for cross-correlation
        sig_a_r = (sig_a_r - np.mean(sig_a_r)) / (np.std(sig_a_r) + 1e-8)
        sig_b_r = (sig_b_r - np.mean(sig_b_r)) / (np.std(sig_b_r) + 1e-8)

        # Cross-correlate
        corr = np.correlate(sig_a_r, sig_b_r, mode="same")
        lag = np.argmax(corr) - len(corr) // 2
        offset_sec = lag * 0.1  # lag in samples × 0.1s per sample

        return offset_sec

    def _extract_overlap_signal(
        self, camera: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract motion signal with absolute timestamps for a camera."""
        tl = self.timelines.get(camera, [])
        all_signal = []
        all_times = []
        for entry in tl:
            sig = entry.get("motion_signal", [])
            if not sig:
                continue
            fps = entry["fps"]
            abs_start = entry.get("start_time_abs", 0)
            times = abs_start + np.arange(len(sig)) / fps
            all_signal.extend(sig)
            all_times.extend(times.tolist())
        return np.array(all_signal), np.array(all_times)

    # ------------------------------------------------------------------
    # Stage 8: Export
    # ------------------------------------------------------------------
    def export(self, output_dir: str | Path | None = None) -> dict:
        """Export unified timeline and calibration results."""
        self._print_stage(8, "Export", "Saving unified timeline and calibration data")

        if output_dir is None:
            output_dir = self.root_dir / "calibration_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Build per-camera frame index ---
        frame_index = []
        for cam, tl in self.timelines.items():
            abs_start = tl[0]["start_time_abs"] if tl else 0
            for entry in tl:
                fps = entry["fps"]
                n = entry["frame_count"]
                rel_offset = entry["start_time_rel"]
                for i in range(n):
                    frame_index.append({
                        "camera": cam,
                        "filename": entry["filename"],
                        "frame_idx": i,
                        "time_rel": round(rel_offset + i / fps, 4),
                        "time_abs": round(abs_start + rel_offset + i / fps, 4),
                    })

        unified = {
            "session": self.root_dir.name,
            "root_dir": str(self.root_dir),
            "n_cameras": len(self.timelines),
            "n_files": len(self.results),
            "n_frames": sum(r.get("actual_frames", 0) for r in self.results),
            "calibration_offsets": self.offsets,
            "per_camera_timelines": {
                str(cam): entry
                for cam, entry in self.timelines.items()
            },
            "frame_index": frame_index,
        }

        csv_path = output_dir / "unified_timeline.csv"
        json_path = output_dir / "calibration_result.json"
        summary_path = output_dir / "summary.txt"

        # CSV (frame index)
        if frame_index:
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=frame_index[0].keys())
                w.writeheader()
                w.writerows(frame_index)
            print(f"  Frame index: {csv_path} ({len(frame_index)} rows)")

        # JSON (full result)
        with open(json_path, "w") as f:
            json.dump(unified, f, indent=2, default=str)
        print(f"  Full result: {json_path}")

        # Summary text
        with open(summary_path, "w") as f:
            f.write(self._format_summary())
        print(f"  Summary: {summary_path}")

        print()
        return unified

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _format_summary(self) -> str:
        lines = ["=" * 60]
        lines.append(f"CROSS-CAMERA CALIBRATION REPORT")
        lines.append(f"Session: {self.root_dir.name}")
        lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Files scanned: {len(self.files)}")
        lines.append(f"Files probed: {len(self.results)}")
        ok = sum(1 for r in self.results if "error" not in r)
        err = sum(1 for r in self.results if "error" in r)
        lines.append(f"  OK: {ok}, Failed: {err}")
        lines.append("")

        total_frames = sum(r.get("actual_frames", 0) for r in self.results)
        total_dur = sum(r.get("actual_duration_sec", 0) for r in self.results)
        lines.append(f"Total frames: {total_frames}")
        lines.append(f"Total duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")
        lines.append("")

        lines.append("Cameras:")
        for r in self.results:
            if "error" not in r:
                lines.append(
                    f"  Cam {r['camera']} — {r['filename']}: "
                    f"{r['actual_frames']}f @ {r['fps']} fps, "
                    f"{r['actual_duration_sec']:.1f}s"
                )
        lines.append("")

        lines.append("Timelines:")
        for cam, tl in sorted(self.timelines.items()):
            dur = tl[-1]["end_time_rel"]
            lines.append(f"  Camera {cam}: {len(tl)} files, {dur:.1f}s total")
            for entry in tl:
                lines.append(
                    f"    {entry['filename']}: "
                    f"{entry['start_time_rel']:.1f}s → {entry['end_time_rel']:.1f}s "
                    f"({entry['frame_count']}f)"
                )
        lines.append("")

        lines.append("Calibration Offsets:")
        for pair, offset in sorted(self.offsets.items()):
            status = f"{offset:.3f}s" if offset is not None else "NO DATA"
            lines.append(f"  {pair}: {status}")
        lines.append("")

        n_entries = len(self.unified) if self.unified else 0
        if n_entries > 0:
            lines.append(f"Frame index entries: {n_entries}")
        else:
            lines.append("Frame index entries: (run export() to generate)")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _print_stage(self, stage: int, name: str, desc: str):
        print(f"\n{'─' * 60}")
        print(f"Stage {stage}/{self._total_stages}: {name} — {desc}")
        print(f"{'─' * 60}")

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------
    def run(self, compute_motion: bool = True, export: bool = True):
        """Run the full pipeline end-to-end."""
        t0 = time.time()
        print(f"Cross-Camera Calibration Pipeline")
        print(f"Root: {self.root_dir}")
        print(f"Files to process: {len(self.files)}")
        print()

        self.scan()
        self.probe()
        self.quality()
        self.parse_timestamps()
        self.build_timelines()
        if compute_motion:
            self.compute_motion()
            self.synchronize()
        if export:
            result = self.export()
        elapsed = time.time() - t0
        print(f"{'─' * 60}")
        print(f"Pipeline complete in {elapsed:.1f}s")

        return self.results, self.timelines, self.offsets


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Cross-camera timing calibration pipeline")
    parser.add_argument("root_dir", type=str,
                        help="Directory containing video files")
    parser.add_argument("--no-motion", action="store_true",
                        help="Skip motion computation and sync")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip export")
    args = parser.parse_args()

    pipe = CalibrationPipeline(args.root_dir)
    pipe.run(
        compute_motion=not args.no_motion,
        export=not args.no_export,
    )


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


PLOT_STYLE = {
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
}


def plot_trajectory(ax, df: pd.DataFrame, session: str, camera: int):
    centroid_x = df.get("centroid_x")
    centroid_y = df.get("centroid_y")
    if centroid_x is not None and centroid_y is not None:
        ax.plot(centroid_x, centroid_y, linewidth=0.5, alpha=0.7, color="blue")
        ax.scatter(centroid_x.iloc[0], centroid_y.iloc[0], c="green", s=50, label="Start")
        ax.scatter(centroid_x.iloc[-1], centroid_y.iloc[-1], c="red", s=50, label="End")
    ax.set_title(f"{session} - Camera {camera}: Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_aspect("equal")


def plot_speed_distribution(ax, df: pd.DataFrame, session: str, camera: int):
    speed = df.get("centroid_speed")
    if speed is not None:
        ax.hist(speed.dropna(), bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_title(f"{session} - Camera {camera}: Speed Distribution")
    ax.set_xlabel("Speed (px/s)")
    ax.set_ylabel("Frequency")


def plot_heading(ax, df: pd.DataFrame, session: str, camera: int):
    heading = df.get("heading_deg")
    if heading is not None:
        ax.plot(heading, linewidth=0.5, color="purple", alpha=0.7)
    ax.set_title(f"{session} - Camera {camera}: Heading")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Heading (deg)")
    ax.set_ylim(-180, 180)


def plot_feeding_timeline(ax, feeding_events: pd.DataFrame, total_frames: int, session: str, camera: int):
    if feeding_events.empty:
        ax.text(0.5, 0.5, "No feeding events", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{session} - Camera {camera}: Feeding Timeline")
        return
    for _, event in feeding_events.iterrows():
        ax.barh(0, event["end_frame"] - event["start_frame"],
                left=event["start_frame"], height=0.6, color="orange", edgecolor="black")
    ax.set_xlim(0, total_frames)
    ax.set_title(f"{session} - Camera {camera}: Feeding Events")
    ax.set_xlabel("Frame")
    ax.set_yticks([])


def generate_pdf_report(
    pose_df: pd.DataFrame | None,
    feeding_events: pd.DataFrame | None,
    features_df: pd.DataFrame | None,
    output_path: str | Path,
    session: str = "",
    camera: int = 0,
    fps: float = 30.0,
):
    plt.rcParams.update(PLOT_STYLE)
    with PdfPages(output_path) as pdf:
        total_frames = len(pose_df) if pose_df is not None else 0
        duration_min = round(total_frames / fps / 60, 2) if fps > 0 else 0
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        text = f"Pig Behavior Report\nSession: {session}\nCamera: {camera}\nDuration: {duration_min} min\nFrames: {total_frames}\nFPS: {fps}"
        if feeding_events is not None and not feeding_events.empty:
            text += f"\nFeeding Events: {len(feeding_events)}"
            text += f"\nTotal Feeding: {feeding_events['duration_sec'].sum():.1f}s"
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=16, transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)
        if pose_df is not None:
            fig, ax = plt.subplots()
            plot_trajectory(ax, pose_df, session, camera)
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots()
            plot_speed_distribution(ax, pose_df, session, camera)
            pdf.savefig(fig)
            plt.close(fig)
            if "heading_deg" in pose_df.columns:
                fig, ax = plt.subplots()
                plot_heading(ax, pose_df, session, camera)
                pdf.savefig(fig)
                plt.close(fig)
        if feeding_events is not None:
            fig, ax = plt.subplots(figsize=(12, 2))
            plot_feeding_timeline(ax, feeding_events, total_frames, session, camera)
            pdf.savefig(fig)
            plt.close(fig)
        if features_df is not None:
            for col in features_df.columns:
                if features_df[col].dtype.kind in "fc":
                    fig, ax = plt.subplots()
                    ax.plot(features_df[col], linewidth=0.5, alpha=0.7)
                    ax.set_title(col)
                    ax.set_xlabel("Frame")
                    pdf.savefig(fig)
                    plt.close(fig)


def generate_html_report(
    pose_df: pd.DataFrame | None,
    feeding_events: pd.DataFrame | None,
    features_df: pd.DataFrame | None,
    output_path: str | Path,
    session: str = "",
    camera: int = 0,
    fps: float = 30.0,
):
    plt.rcParams.update(PLOT_STYLE)
    total_frames = len(pose_df) if pose_df is not None else 0
    duration_min = round(total_frames / fps / 60, 2) if fps > 0 else 0
    image_paths = []
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        text = f"Session: {session} | Camera: {camera} | Duration: {duration_min} min | Frames: {total_frames} | FPS: {fps}"
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, transform=ax.transAxes)
        p = Path(tmpdir) / "cover.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        image_paths.append(p)
        if pose_df is not None:
            fig, ax = plt.subplots()
            plot_trajectory(ax, pose_df, session, camera)
            p = Path(tmpdir) / "trajectory.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            image_paths.append(p)
            fig, ax = plt.subplots()
            plot_speed_distribution(ax, pose_df, session, camera)
            p = Path(tmpdir) / "speed.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            image_paths.append(p)
        if feeding_events is not None and not feeding_events.empty:
            fig, ax = plt.subplots()
            feeding_events["duration_sec"].hist(ax=ax, bins=20, color="orange", edgecolor="black")
            ax.set_xlabel("Duration (s)")
            ax.set_ylabel("Count")
            ax.set_title("Feeding Bout Durations")
            p = Path(tmpdir) / "feeding_durations.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            image_paths.append(p)
        html = "<html><head><style>"
        html += "body{font-family:Arial,sans-serif;margin:40px;background:#f5f5f5}"
        html += "h1{color:#333}.section{background:white;padding:20px;margin:20px 0;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}"
        html += "img{max-width:100%;height:auto;display:block;margin:10px 0}"
        html += "</style></head><body>"
        html += f"<h1>Pig Behavior Report</h1>"
        html += f"<div class='section'><h2>Summary</h2>"
        html += f"<p><b>Session:</b> {session} | <b>Camera:</b> {camera}</p>"
        html += f"<p><b>Duration:</b> {duration_min} min | <b>Frames:</b> {total_frames} | <b>FPS:</b> {fps}</p>"
        if feeding_events is not None and not feeding_events.empty:
            html += f"<p><b>Feeding Events:</b> {len(feeding_events)} | <b>Total Feeding:</b> {feeding_events['duration_sec'].sum():.1f}s</p>"
        html += "</div>"
        for img_path in image_paths:
            import base64
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            html += f"<div class='section'><img src='data:image/png;base64,{b64}'/></div>"
        html += "</body></html>"
        with open(output_path, "w") as f:
            f.write(html)

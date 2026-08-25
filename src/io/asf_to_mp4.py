import subprocess
import sys
from pathlib import Path


def convert_asf_to_mp4(
    folder: str | Path,
    recursive: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """Convert all .asf files under *folder* to .mp4 using ffmpeg.

    Skips any file whose .mp4 counterpart already exists (unless *overwrite*).

    Parameters
    ----------
    folder : str | Path
        Directory containing .asf files.
    recursive : bool
        If True, walk into subdirectories.
    overwrite : bool
        If True, re-convert even when the .mp4 already exists.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    list[Path]
        Paths to the newly created (or overwritten) .mp4 files.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    asf_files = sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".asf"
        and not p.name.startswith("._")
    )
    if not asf_files:
        if verbose:
            print(f"No .asf files found in {folder}")
        return []

    converted: list[Path] = []
    for asf_path in asf_files:
        mp4_path = asf_path.with_suffix(".mp4")
        if mp4_path.exists() and not overwrite:
            if verbose:
                print(f"  [skip] {asf_path.name} (mp4 exists)")
            continue
        if verbose:
            print(f"  [convert] {asf_path.name} -> {mp4_path.name} ...", end=" ", flush=True)
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(asf_path),
                "-c:v", "libx264", "-preset", "fast",
                "-an",
                str(mp4_path),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            if verbose:
                print(f"FAILED\n    {result.stderr.strip().splitlines()[-1]}")
            continue
        if verbose:
            print("OK")
        converted.append(mp4_path)

    if verbose:
        print(f"\nDone: {len(converted)}/{len(asf_files)} converted")
    return converted


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert .asf files to .mp4")
    parser.add_argument("folder", help="Directory containing .asf files")
    parser.add_argument("--no-recursive", action="store_true", help="Don't scan subdirectories")
    parser.add_argument("--overwrite", action="store_true", help="Re-convert even if .mp4 exists")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    converted = convert_asf_to_mp4(
        folder=args.folder,
        recursive=not args.no_recursive,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )
    sys.exit(0 if converted or not args.quiet else 1)


if __name__ == "__main__":
    main()

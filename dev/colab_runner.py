"""
Colab Runner — execute a Jupyter notebook headlessly on Colab,
capture errors, and enable the opencode fix loop.

Usage:
    python dev/colab_runner.py [--drive-root PATH] notebooks/XX.ipynb

    --drive-root  Override DRIVE_ROOT in the notebook config before running.

On success: exit 0
On error:   saves traceback to /tmp/colab_error.txt, prints details, exit 1

The fix loop:
    1. Run:        python dev/colab_runner.py notebooks/XX.ipynb
    2. Error?      cat /tmp/colab_error.txt  → paste to opencode
    3. opencode    fixes code → pushes to GitHub
    4. Re-run:     !git pull && python dev/colab_runner.py notebooks/XX.ipynb
"""

import os
import sys
import json
import re
import subprocess
import tempfile
import shutil
from pathlib import Path


def inject_drive_root(nb: dict, drive_root: str) -> bool:
    """Replace DRIVE_ROOT in the notebook's config cell in-place.
    Returns True if a replacement was made."""
    if not drive_root:
        return False
    target_line = f'DRIVE_ROOT = "{drive_root}"'
    modified = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            src = [src]
        for i, line in enumerate(src):
            stripped = line.strip()
            if stripped.startswith("DRIVE_ROOT = "):
                # Preserve existing indentation and any comment after
                indent = line[:len(line) - len(line.lstrip())]
                comment = ""
                if "#" in line:
                    comment = "  " + line.split("#", 1)[1].rstrip("\n")
                src[i] = f'{indent}{target_line}{comment}\n'
                modified = True
    return modified


def run_notebook(notebook_path: str, timeout: int = 3600, drive_root: str = None) -> dict:
    """Execute a notebook via nbconvert and return results.

    Returns dict with keys: success, output, error, notebook_path
    """
    nb_path = Path(notebook_path)
    if not nb_path.exists():
        return {
            "success": False,
            "output": "",
            "error": f"Notebook not found: {nb_path}",
            "notebook_path": str(nb_path),
        }

    # Work in a temp dir to avoid clobbering the original
    tmpdir = Path(tempfile.mkdtemp())
    tmp_nb = tmpdir / nb_path.name
    shutil.copy2(nb_path, tmp_nb)

    # Inject DRIVE_ROOT into the temp copy if provided
    if drive_root:
        with open(tmp_nb) as f:
            nb_data = json.load(f)
        if inject_drive_root(nb_data, drive_root):
            with open(tmp_nb, "w") as f:
                json.dump(nb_data, f, indent=1, ensure_ascii=False)
            print(f"  Injected DRIVE_ROOT={drive_root}")

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
        "--execute", str(tmp_nb),
        "--ExecutePreprocessor.timeout", str(timeout),
        "--output", str(tmp_nb),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)

    output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        error_file = Path("/tmp/colab_error.txt")
        error_msg = (
            f"Notebook: {nb_path}\n"
            f"{'=' * 60}\n"
            f"{output}\n"
        )
        error_file.write_text(error_msg)
        print(f"ERROR: {nb_path} failed.")
        print(f"Traceback saved to /tmp/colab_error.txt")
        print(f"Run: cat /tmp/colab_error.txt  → then paste to opencode")
        return {
            "success": False,
            "output": output,
            "error": error_msg,
            "notebook_path": str(nb_path),
        }
    else:
        print(f"SUCCESS: {nb_path} completed.")
        return {
            "success": True,
            "output": output,
            "error": "",
            "notebook_path": str(nb_path),
        }


def run_pipeline(notebooks: list, timeout: int = 3600, drive_root: str = None) -> list:
    """Run multiple notebooks in sequence. Stops on first failure."""
    results = []
    for nb in notebooks:
        print(f"\n{'=' * 60}")
        print(f"Running: {nb}")
        print(f"{'=' * 60}")
        r = run_notebook(nb, timeout=timeout, drive_root=drive_root)
        results.append(r)
        if not r["success"]:
            print(f"\nPipeline stopped at: {nb}")
            print(f"Fix the error, then re-run from this notebook.")
            break
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Execute Jupyter notebooks headlessly.")
    parser.add_argument("notebooks", nargs="+", help="Notebook file(s) to execute")
    parser.add_argument("--drive-root", default=None, help="Override DRIVE_ROOT in notebook config")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-cell timeout in seconds")
    args = parser.parse_args()

    results = run_pipeline(args.notebooks, timeout=args.timeout, drive_root=args.drive_root)

    any_failed = any(not r["success"] for r in results)
    sys.exit(1 if any_failed else 0)

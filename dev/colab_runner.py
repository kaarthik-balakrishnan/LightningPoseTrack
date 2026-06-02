"""
Colab Runner — execute a Jupyter notebook headlessly on Colab,
capture errors, and enable the opencode fix loop.

Usage:
    python dev/colab_runner.py [--path KEY=VALUE ...] notebooks/XX.ipynb

    --path KEY=VALUE  Override variable KEY with VALUE in notebook config.
                      Can be specified multiple times (e.g. --path A=x --path B=y).

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
import subprocess
import tempfile
import shutil
from pathlib import Path


def inject_variables(nb: dict, overrides: dict) -> bool:
    """Replace any variable assignment in notebook code cells with values
    from *overrides*. Works for assignments of the form::

        VAR_NAME = "value"   or   VAR_NAME = f"..."

    Preserves indentation and trailing comments. Returns True if any
    replacement was made.
    """
    if not overrides:
        return False
    modified = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            src = [src]
        for i, line in enumerate(src):
            stripped = line.strip()
            for key, value in overrides.items():
                if stripped.startswith(f"{key} = ") or stripped.startswith(f"{key}="):
                    indent = line[:len(line) - len(line.lstrip())]
                    # Preserve any trailing comment
                    comment = ""
                    if "#" in line:
                        comment = "  " + line.split("#", 1)[1].rstrip("\n")
                    # Quote value if it contains / (looks like a path) and isn't already quoted
                    val_str = value
                    if "/" in value and not (value.startswith('"') or value.startswith("'")):
                        val_str = f'"{value}"'
                    src[i] = f'{indent}{key} = {val_str}{comment}\n'
                    modified = True
                    break  # once matched on this line, move to next line
    return modified


def run_notebook(notebook_path: str, timeout: int = 3600,
                 path_overrides: dict = None) -> dict:
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

    # Inject path overrides into the temp copy if provided
    if path_overrides:
        with open(tmp_nb) as f:
            nb_data = json.load(f)
        if inject_variables(nb_data, path_overrides):
            with open(tmp_nb, "w") as f:
                json.dump(nb_data, f, indent=1, ensure_ascii=False)
            for k, v in path_overrides.items():
                print(f"  Injected {k}={v}")

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


def run_pipeline(notebooks: list, timeout: int = 3600,
                 path_overrides: dict = None) -> list:
    """Run multiple notebooks in sequence. Stops on first failure."""
    results = []
    for nb in notebooks:
        print(f"\n{'=' * 60}")
        print(f"Running: {nb}")
        print(f"{'=' * 60}")
        r = run_notebook(nb, timeout=timeout, path_overrides=path_overrides)
        results.append(r)
        if not r["success"]:
            print(f"\nPipeline stopped at: {nb}")
            print(f"Fix the error, then re-run from this notebook.")
            break
    return results


if __name__ == "__main__":
    import argparse

    def parse_path_arg(val: str):
        """Parse ``KEY=VALUE`` arguments."""
        if "=" not in val:
            raise argparse.ArgumentTypeError(
                f"Expected KEY=VALUE, got '{val}'"
            )
        k, _, v = val.partition("=")
        return k.strip(), v.strip()

    parser = argparse.ArgumentParser(description="Execute Jupyter notebooks headlessly.")
    parser.add_argument("notebooks", nargs="+", help="Notebook file(s) to execute")
    parser.add_argument(
        "--path", action="append", type=parse_path_arg, default=[],
        dest="path_pairs",
        help="Override a variable in notebook config (e.g. --path DRIVE_ROOT=/foo)"
    )
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-cell timeout in seconds")
    # Deprecated — kept for backward compatibility
    parser.add_argument("--drive-root", default=None,
                        help="Shorthand for --path DRIVE_ROOT=...")
    args = parser.parse_args()

    overrides = dict(args.path_pairs)
    if args.drive_root:
        overrides["DRIVE_ROOT"] = args.drive_root

    results = run_pipeline(args.notebooks, timeout=args.timeout,
                           path_overrides=overrides)

    any_failed = any(not r["success"] for r in results)
    sys.exit(1 if any_failed else 0)

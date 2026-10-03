"""``python -m tinyllm lab`` - Jupyter Lab wired to this machine's settings.

Jupyter Lab is a fourth front end for the same services the CLI drives. The
point of this command is that the kernel is not a bare Python process: it is
registered with the device, thread count and preset that the environment scan
resolved, so a notebook session and a terminal run behave identically.

    python -m tinyllm lab --install     # install the notebook requirements
    python -m tinyllm lab               # register the kernel and open Lab
    python -m tinyllm lab --list        # show what was found
    python -m tinyllm lab --check       # exit 1 if Lab is not usable yet
    python -m tinyllm lab --no-browser --port 8890
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

from ..core.environment import environment_vars, resolve_settings
from ..core.paths import ROOT

KERNEL_NAME = "tinyllm"
KERNEL_DISPLAY_NAME = "TinyLLM (scanned environment)"
NOTEBOOK_DIR = ROOT / "notebooks"
REQUIREMENTS_FILE = ROOT / "requirements-notebook.txt"


# --------------------------------------------------------------------------- #
# what is installed
# --------------------------------------------------------------------------- #
def module_version(name: str) -> str | None:
    """Installed version of a module, or ``None`` when it is absent."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def lab_version() -> str | None:
    return module_version("jupyterlab")


def lab_installed() -> bool:
    return lab_version() is not None and find_spec("jupyterlab") is not None


def ipykernel_installed() -> bool:
    return module_version("ipykernel") is not None and find_spec("ipykernel") is not None


def notebook_files(directory: Path = NOTEBOOK_DIR) -> list[Path]:
    """The shipped notebooks, in a stable order."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.ipynb"))


# --------------------------------------------------------------------------- #
# the kernel
# --------------------------------------------------------------------------- #
def kernel_dir(base: Path | None = None, name: str = KERNEL_NAME) -> Path:
    """Where the kernel spec is written.

    Defaults to the user data directory when Jupyter is installed, so the spec
    is visible to JupyterLab even when TinyLLM is installed as a package
    somewhere else on disk.
    """
    if base is not None:
        return Path(base) / name
    try:
        from jupyter_core.paths import jupyter_data_dir
        return Path(jupyter_data_dir()) / "kernels" / name
    except Exception:
        return Path(sys.prefix) / "share" / "jupyter" / "kernels" / name


def write_kernel_spec(settings=None, base: Path | None = None,
                      name: str = KERNEL_NAME) -> Path:
    """Register a kernel that starts with the resolved settings in its env.

    The ``env`` block is what makes the notebook session match the terminal:
    ``TINYLLM_DEVICE``, ``TINYLLM_THREADS`` and ``OMP_NUM_THREADS`` are set
    before the kernel's Python starts.
    """
    settings = settings or resolve_settings()
    spec_dir = kernel_dir(base=base, name=name)
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": KERNEL_DISPLAY_NAME,
        "language": "python",
        "name": name,
        "env": environment_vars(settings),
    }
    spec_path = spec_dir / "kernel.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec_path


def read_kernel_spec(base: Path | None = None,
                     name: str = KERNEL_NAME) -> dict | None:
    path = kernel_dir(base=base, name=name) / "kernel.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# the notebooks
# --------------------------------------------------------------------------- #
def validate_notebooks(directory: Path = NOTEBOOK_DIR) -> list[str]:
    """Structural problems with the shipped notebooks; empty means healthy.

    Checked here rather than trusted: a hand-edited ``.ipynb`` with malformed
    JSON or a cell that no longer parses fails the notebook at open time, long
    after the mistake was made.
    """
    problems: list[str] = []
    for path in notebook_files(directory):
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}: invalid JSON ({exc})")
            continue
        if not isinstance(nb, dict) or nb.get("nbformat") != 4:
            problems.append(f"{path.name}: not an nbformat 4 notebook")
            continue
        kernel = nb.get("metadata", {}).get("kernelspec", {}).get("name")
        if kernel != KERNEL_NAME:
            problems.append(f"{path.name}: kernel is {kernel!r}, expected "
                            f"{KERNEL_NAME!r}")
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        if not code_cells:
            problems.append(f"{path.name}: no code cells")
        for index, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            try:
                compile("".join(cell.get("source", [])), f"{path.name}:cell{index}",
                        "exec")
            except SyntaxError as exc:
                problems.append(f"{path.name}: cell {index} does not parse ({exc.msg})")
            if cell.get("outputs"):
                problems.append(f"{path.name}: cell {index} has stored outputs")
    return problems


def write_notebook_cells(nb: dict, cells: list[tuple[str, str]]) -> dict:
    """Build a minimal, output-free notebook document from (type, source) pairs.

    The shipped notebooks are plain JSON files; this builder is what generates
    them, and the tests check its output so the shape cannot drift.
    """
    built = []
    for kind, source in cells:
        lines = source.strip("\n").splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        built.append(cell)
    nb["cells"] = built
    return nb


def new_notebook(cells: list[tuple[str, str]]) -> dict:
    return write_notebook_cells({
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": KERNEL_DISPLAY_NAME,
                           "language": "python", "name": KERNEL_NAME},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }, cells)


# --------------------------------------------------------------------------- #
# launching
# --------------------------------------------------------------------------- #
def lab_command(notebook_dir: Path, port: int, ip: str, no_browser: bool,
                extra: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "jupyterlab",
           f"--notebook-dir={notebook_dir}",
           f"--port={port}", f"--ip={ip}"]
    if no_browser:
        cmd.append("--no-browser")
    cmd += list(extra or [])
    return cmd


def launch(notebook_dir: Path = NOTEBOOK_DIR, port: int = 8888,
           ip: str = "127.0.0.1", no_browser: bool = False,
           env: dict[str, str] | None = None) -> int:
    """Run Jupyter Lab in a child process, forwarding the resolved settings."""
    cmd = lab_command(notebook_dir, port, ip, no_browser)
    child_env = {**os.environ, **(env or {})}
    print(f"[lab]    {' '.join(cmd)}")
    print(f"[lab]    kernel {KERNEL_NAME} -> {kernel_dir()}")
    try:
        return subprocess.call(cmd, env=child_env)
    except KeyboardInterrupt:               # pragma: no cover - interactive
        return 0


def install_requirements() -> int:
    """``pip install -r requirements-notebook.txt`` into the active env."""
    if not REQUIREMENTS_FILE.exists():
        print(f"[error] missing {REQUIREMENTS_FILE}", file=sys.stderr)
        return 1
    print(f"$ pip install -r {REQUIREMENTS_FILE.name}", flush=True)
    return subprocess.call([sys.executable, "-m", "pip", "install", "-r",
                            str(REQUIREMENTS_FILE)])


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tinyllm lab",
        description="Open Jupyter Lab with a kernel configured for this machine")
    ap.add_argument("--install", action="store_true",
                    help="install requirements-notebook.txt first")
    ap.add_argument("--register-only", action="store_true",
                    help="write the kernel spec and exit, without launching Lab")
    ap.add_argument("--check", action="store_true",
                    help="report whether Lab is usable; exit 1 if not")
    ap.add_argument("--list", action="store_true",
                    help="list the shipped notebooks and the kernel spec")
    ap.add_argument("--no-browser", action="store_true",
                    help="start the server without opening a browser")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--ip", default="127.0.0.1",
                    help="default 127.0.0.1; use 0.0.0.0 to expose on the LAN")
    ap.add_argument("--notebook-dir", default=str(NOTEBOOK_DIR),
                    help="directory Lab opens (default: the shipped notebooks)")
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:N | mps | xpu | dml")
    return ap


def _status_lines(settings) -> list[str]:
    return [
        f"  jupyterlab     : {lab_version() or 'NOT INSTALLED'}",
        f"  ipykernel      : {module_version('ipykernel') or 'NOT INSTALLED'}",
        f"  kernel         : {KERNEL_NAME} -> {kernel_dir()}",
        f"  device         : {settings.device} ({settings.backend})",
        f"  threads        : {settings.threads}",
        f"  preset         : {settings.preset} "
        f"(batch {settings.batch_size}, block {settings.block_size})",
        f"  amp            : {settings.amp}",
        f"  notebooks      : {len(notebook_files())} in {NOTEBOOK_DIR}",
    ]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.install:
        code = install_requirements()
        if code != 0:
            print("[error] notebook requirements failed to install", file=sys.stderr)
            return code

    try:
        settings = resolve_settings(args.device)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print("=" * 68)
    print("TinyLLM - Jupyter Lab")
    print("=" * 68)
    print("\n".join(_status_lines(settings)))

    problems = validate_notebooks()
    if problems:
        print("notebook problems:")
        for problem in problems:
            print(f"  !! {problem}")

    if args.list:
        for path in notebook_files():
            print(f"  {path.name}")
        spec = read_kernel_spec()
        if spec:
            print(f"  kernel.json env: {json.dumps(spec.get('env', {}))}")
        else:
            print("  kernel.json: not registered yet")
        return 1 if problems else 0

    missing = not lab_installed() or not ipykernel_installed()
    if missing:
        print("\n[error] Jupyter Lab is not installed. Run:")
        print("  python -m tinyllm lab --install")
        return 1

    spec_path = write_kernel_spec(settings)
    print(f"[kernel]  {spec_path}")

    if args.check:
        print("RESULT: Jupyter Lab is usable")
        return 1 if problems else 0

    if args.register_only:
        print("RESULT: kernel registered (Lab not started)")
        return 0

    notebook_dir = Path(args.notebook_dir)
    notebook_dir.mkdir(parents=True, exist_ok=True)
    print()
    return launch(notebook_dir, port=args.port, ip=args.ip,
                  no_browser=args.no_browser, env=settings.env)


if __name__ == "__main__":
    raise SystemExit(main())

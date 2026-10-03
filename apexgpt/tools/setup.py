"""Cross-platform bootstrapper.

Detects the OS, Python and graphics backend, then installs the matching
PyTorch wheel plus the remaining dependencies. Run it instead of a bare
``pip install -r requirements.txt`` when you want the GPU build chosen for you::

    python -m apexgpt setup              # detect and show the plan
    python -m apexgpt setup --install    # detect and run pip
    python -m apexgpt setup --backend cpu  # force a backend
    python -m apexgpt setup --backend auto --cuda 12.4
    python -m apexgpt setup --backend dml --install   # Windows AMD/Intel GPU
    python -m apexgpt setup --notebook --install      # plus Jupyter Lab
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys

CUDA_INDEX = "https://download.pytorch.org/whl/cu{tag}"
ROCM_INDEX = "https://download.pytorch.org/whl/rocm{tag}"
XPU_INDEX = "https://download.pytorch.org/whl/xpu"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
DEFAULT_CUDA = "124"
DEFAULT_ROCM = "6.2"
DEFAULT_PYTHON = "3.10"


def detect_python_tag() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}"


def detect_backend() -> str:
    """Which accelerator this machine can use, without importing torch."""
    system = platform.system()
    if system == "Darwin":
        # Apple Silicon gets Metal via the default PyPI wheel; Intel macs are CPU
        if platform.machine() in ("arm64", "aarch64"):
            return "mps"
        return "cpu"
    if system == "Linux":
        if sys.platform.startswith("linux") and _nvidia_smi_works():
            return "cuda"
        if _rocm_smi_works():
            return "rocm"
        return "cpu"
    if system == "Windows":
        return "cuda" if _nvidia_smi_works() else "cpu"
    return "cpu"


def _tool_works(*names: str) -> bool:
    """True if any of the given executables runs successfully."""
    for name in names:
        try:
            if subprocess.run([name], capture_output=True, timeout=10).returncode == 0:
                return True
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
    return False


def _nvidia_smi_works() -> bool:
    return _tool_works("nvidia-smi", "nvidia-smi.exe")


def _rocm_smi_works() -> bool:
    return _tool_works("rocm-smi", "rocm-smi.exe")


def torch_spec(backend: str, cuda_tag: str,
               rocm_tag: str = DEFAULT_ROCM) -> tuple[str, str | None]:
    """Return (package spec, extra index URL) for the chosen backend."""
    if backend == "cpu":
        return "torch", CPU_INDEX
    if backend == "cuda":
        return "torch", CUDA_INDEX.format(tag=cuda_tag)
    if backend == "rocm":
        # ROCm wheels are indexed by ROCm version (rocm6.2), not by CUDA tag.
        return "torch", ROCM_INDEX.format(tag=rocm_tag)
    if backend == "xpu":
        return "torch", XPU_INDEX
    if backend == "dml":
        # DirectML reaches AMD/Intel/NVIDIA GPUs on Windows; PyPI only.
        return "torch-directml", None
    if backend == "mps":
        # MPS ships in the default wheel; no special index
        return "torch", None
    raise ValueError(f"unknown backend: {backend}")


def build_plan(backend: str, cuda_tag: str, rocm_tag: str = DEFAULT_ROCM,
               groups: tuple[str, ...] = ()) -> list[list[str]]:
    """The pip commands to run, in order."""
    spec, index = torch_spec(backend, cuda_tag, rocm_tag)
    torch_cmd = ["pip", "install", spec]
    if index:
        torch_cmd += ["--index-url", index]
    plan = [torch_cmd, ["pip", "install", "-r", "requirements.txt"]]
    for group in groups:
        filename = {"api": "requirements-api.txt",
                    "notebook": "requirements-notebook.txt",
                    "hub": "requirements-hub.txt"}.get(group)
        if filename:
            plan.append(["pip", "install", "-r", filename])
    return plan


def has_non_nvidia_adapter() -> bool:
    """True when a display adapter is present that CUDA cannot drive."""
    from ..core.device import display_adapters
    for name in display_adapters():
        lowered = name.lower()
        if not any(vendor in lowered for vendor in ("nvidia", "geforce", "quadro", "tesla")):
            return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="apexgpt setup",
                                 description="Install ApexGPT for this machine")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "cpu", "cuda", "rocm", "xpu", "mps", "dml"])
    ap.add_argument("--cuda", default=None,
                    help=f"CUDA major.minor for the wheel index (default {DEFAULT_CUDA})")
    ap.add_argument("--rocm", default=DEFAULT_ROCM,
                    help=f"ROCm version for the wheel index (default {DEFAULT_ROCM})")
    ap.add_argument("--notebook", action="store_true",
                    help="also install requirements-notebook.txt (Jupyter Lab)")
    ap.add_argument("--api", action="store_true",
                    help="also install requirements-api.txt (HTTP service)")
    ap.add_argument("--hub", action="store_true",
                    help="also install requirements-hub.txt (Kaggle, Hub models)")
    ap.add_argument("--install", action="store_true",
                    help="actually run pip; without it the plan is only printed")
    args = ap.parse_args(argv)

    backend = detect_backend() if args.backend == "auto" else args.backend
    cuda_tag = args.cuda or DEFAULT_CUDA
    rocm_tag = args.rocm or DEFAULT_ROCM

    print("=" * 66)
    print("ApexGPT setup")
    print("=" * 66)
    print(f"  OS              : {platform.system()} {platform.release()}")
    print(f"  architecture    : {platform.machine()}")
    print(f"  Python          : {platform.python_version()} (tag {detect_python_tag()})")
    print(f"  detected backend: {backend}")
    if backend == "cuda":
        print(f"  wheel index     : CUDA {cuda_tag}")
    if backend == "rocm":
        print(f"  wheel index     : ROCm {rocm_tag}")
    print()

    try:
        spec, index = torch_spec(backend, cuda_tag, rocm_tag)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if platform.system() != "Windows" and backend == "dml":
        print("[error] torch-directml is Windows-only; on Linux use --backend rocm",
              file=sys.stderr)
        return 1
    if platform.system() == "Linux" and backend == "mps":
        print("[error] mps is only available on macOS", file=sys.stderr)
        return 1
    if platform.system() == "Darwin" and backend in ("cuda", "rocm", "xpu", "dml"):
        print(f"[error] {backend} is not available on macOS", file=sys.stderr)
        return 1
    if platform.python_version_tuple()[:2] > ("3", "13") and backend in ("cuda", "xpu"):
        print(f"[note] Python {detect_python_tag()} wheels may not exist for every "
              f"accelerator yet; 3.10-3.12 is the safest choice")

    groups = tuple(g for g, on in (("api", args.api), ("notebook", args.notebook),
                               ("hub", args.hub)) if on)
    plan = build_plan(backend, cuda_tag, rocm_tag, groups)
    print("Plan:")
    for cmd in plan:
        print("  " + " ".join(cmd))
    print()

    if backend == "cpu" and has_non_nvidia_adapter():
        print("[note] a non-NVIDIA adapter is present, so CUDA will stay unavailable.")
        print("       AMD/Intel GPUs: ROCm on Linux, or --backend dml on Windows")
        print("       (DirectML is experimental and slower than the CPU for training).")
        print()

    if not args.install:
        print("Nothing installed. Re-run with --install to apply:")
        print(f"  python -m apexgpt setup --install --backend {backend}")
        print()
        print("Verify afterwards with:  python -m apexgpt doctor")
        return 0

    for cmd in plan:
        print(f"$ {' '.join(cmd)}", flush=True)
        code = subprocess.call([sys.executable, "-m"] + cmd)
        if code != 0:
            print(f"[error] failed: {' '.join(cmd)}", file=sys.stderr)
            return code
    print()
    print("Done. Verify with:  python -m apexgpt doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
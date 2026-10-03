"""Device selection, thread configuration, and precision policy.

ApexGPT runs on whatever accelerator PyTorch exposes on the host:

===============  =====================================================
backend          how it is detected
===============  =====================================================
``cuda``         NVIDIA (any CUDA build, 10.2+), and AMD ROCm builds
``mps``          Apple Silicon (macOS 12.3+)
``xpu``          Intel GPUs via the ``intel``/oneAPI PyTorch build
``dml``          Windows AMD/NVIDIA GPUs via the optional ``torch-directml``
``cpu``          always available; fp32 or bf16
===============  =====================================================

``--device auto`` (the default) picks the fastest one present. An explicit
request for a backend that is missing raises instead of silently falling back,
so a typo never turns into a confusing CPU-only run.

AMD GPUs are reachable two ways: a ROCm PyTorch build, which reports itself as
``cuda`` (:func:`rocm_build` says which), or ``torch-directml`` on Windows,
which is opt-in through ``--device dml`` because DirectML has no fused
attention kernel and is usually slower than a full CPU thread pool for
training. See ``python -m apexgpt doctor`` for what a host actually offers.
"""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass

import torch

# Preference order when ``--device auto`` is used: fastest first.
# DirectML is deliberately absent: it is opt-in, never auto-selected.
_BACKEND_PRIORITY = ("cuda", "xpu", "mps")


def cuda_available() -> bool:
    return torch.cuda.is_available()


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    try:
        return bool(backend and backend.is_available() and backend.is_built())
    except Exception:
        return False


def xpu_available() -> bool:
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def dml_available() -> bool:
    """Whether the optional ``torch-directml`` package can reach a GPU.

    DirectML is Windows-only and reaches AMD Radeon, Intel Arc and NVIDIA
    cards through the DirectX 12 backend. It is never auto-selected - see the
    module docstring - so this probe only decides whether ``--device dml``
    is a legal request.
    """
    if sys.platform != "win32":
        return False
    try:
        import torch_directml
    except Exception:
        return False
    try:
        return torch_directml.device() is not None
    except Exception:
        return False


def rocm_build() -> str | None:
    """The ROCm/HIP version this torch was built against, if any.

    A ROCm build reports ``torch.cuda.is_available()`` like a CUDA build, so
    ``cuda available: True`` alone cannot tell an AMD user whether their GPU is
    being used. This is what disambiguates it.
    """
    return getattr(torch.version, "hip", None)


def available_backends(include_experimental: bool = False) -> list[str]:
    """Every accelerator backend usable on this host, fastest first.

    ``include_experimental`` adds DirectML, which is opt-in via ``--device dml``
    rather than picked by ``auto``.
    """
    found: list[str] = []
    if cuda_available():
        found.append("cuda")
    if xpu_available():
        found.append("xpu")
    if mps_available():
        found.append("mps")
    if include_experimental and dml_available():
        found.append("dml")
    return found


def get_device(preferred: str | None = None) -> torch.device:
    """Resolve the device to train or run on.

    ``preferred`` may be ``"auto"`` (default), ``"cpu"``, ``"cuda"``,
    ``"mps"``, ``"xpu"``, ``"dml"``, or an index form such as ``"cuda:1"``.
    """
    if preferred in (None, "", "auto"):
        backends = available_backends()
        if not backends:
            return torch.device("cpu")
        return torch.device(_BACKEND_PRIORITY[0])

    name = preferred.split(":", 1)[0]
    if name == "cpu":
        return torch.device(preferred)
    if name == "dml":
        if not dml_available():
            raise RuntimeError(
                "'dml' requested but torch-directml is not installed. "
                "Install it with: python -m apexgpt setup --backend dml --install"
            )
        import torch_directml
        return torch_directml.device()
    if name not in available_backends(include_experimental=True):
        raise RuntimeError(
            f"{name!r} requested but not available on this machine. "
            f"Available: {available_backends() or ['cpu']}. "
            f"Detected adapters: {display_adapters() or 'none'}"
        )
    return torch.device(preferred)


def device_flavor(device: torch.device) -> str:
    """The autocast dtype family for a device."""
    if device.type in ("cuda", "xpu", "mps"):
        return "float16"
    if device.type == "privateuseone":
        # torch-directml's device type. Autocast is not honoured there, so the
        # caller must leave AMP off rather than wrap ops in a no-op context.
        return "float32"
    return "bfloat16"


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Autocast dtype: fp16 on accelerators, bf16 on CPU."""
    return torch.float16 if device_flavor(device) == "float16" else torch.bfloat16


def supports_bf16(device: torch.device) -> bool:
    """Whether bf16 is native on this CPU.

    Zen2 and older x86 parts emulate it, which is dramatically slower, so the
    precision policy checks this before enabling autocast on CPU.
    """
    if device.type != "cpu":
        return True
    try:
        return bool(torch.cpu._is_cpu_support_avx512_vfma()) or "avx512_bf16" in (
            " ".join(platform.processor().lower().split()))
    except Exception:
        return False


def use_amp_by_default(device: torch.device) -> bool:
    """Mixed precision: on for accelerators, off for CPUs without native bf16.

    Measured on a 4-core / 8-thread CPU-only machine with the cpu-tiny preset:
    195 s/iter with bf16 autocast plus gradient checkpointing, versus
    2.95 s/iter with plain fp32 and no checkpointing - a 66x difference.
    """
    if device.type != "cpu":
        return device.type != "privateuseone"
    return supports_bf16(device)


def use_checkpointing_by_default(device: torch.device) -> bool:
    """Gradient checkpointing trades compute for memory.

    Worth it only where memory is genuinely tight or the extra compute is
    cheap. On CPU it blocks the fused attention path and is a large loss, so it
    stays off there.
    """
    if device.type == "cuda":
        return total_vram_gb(device) < 12.0
    return False


def configure_threads(num_threads: int | None = None) -> int:
    """Set torch CPU threads, clamped to the real core count."""
    physical = os.cpu_count() or 4
    n = physical if num_threads is None else int(num_threads)
    n = max(1, min(n, physical))
    torch.set_num_threads(n)
    return n


def total_vram_gb(device: torch.device | None = None) -> float:
    """Total accelerator memory in GB, or 0.0 when unknown."""
    try:
        if device is None:
            device = get_device()
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            return props.total_memory / 1e9
        if device.type == "mps":
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5)
            return int(out.strip()) / 1e9
        if device.type == "xpu":
            total = getattr(torch.xpu, "get_device_properties", lambda *a: None)(device)
            mem = getattr(total, "total_memory", 0)
            return (mem or 0) / 1e9
    except Exception:
        pass
    return 0.0


def total_ram_gb() -> float:
    """Total system RAM in GB, using only the standard library."""
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / 1e9
        except Exception:
            pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (AttributeError, ValueError, OSError):
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return 0.0


@dataclass
class HardwareProfile:
    """What this machine can actually do, used to pick a default preset."""
    backend: str
    device_name: str
    accelerator_gb: float
    ram_gb: float
    threads: int

    def recommend_preset(self) -> str:
        """Choose a training preset that fits this hardware."""
        if self.backend in ("cuda", "xpu") and self.accelerator_gb >= 10:
            return "full"
        if self.backend in ("cuda", "xpu") and self.accelerator_gb >= 6:
            return "medium"
        if self.backend == "privateuseone":
            # DirectML has no fused attention kernel; only sanity runs are sane.
            return "smoke"
        if self.backend == "cuda" or self.threads >= 8:
            return "cpu-tiny"
        return "smoke"

    def describe(self) -> str:
        mem = f"{self.accelerator_gb:.1f} GB" if self.accelerator_gb else "unknown"
        return (f"{self.backend} ({self.device_name}), {mem} accelerator, "
                f"{self.ram_gb:.0f} GB RAM, {self.threads} threads")


def profile() -> HardwareProfile:
    """Detect the current machine's capabilities."""
    device = get_device()
    backend = device.type
    name = "CPU"
    if backend == "cuda":
        name = torch.cuda.get_device_name(device)
        if rocm_build():
            name = f"{name} (AMD ROCm {rocm_build()})"
    elif backend == "mps":
        name = "Apple Silicon (Metal)"
    elif backend == "xpu":
        name = "Intel GPU"
    elif backend == "privateuseone":
        name = "GPU via DirectML"
    else:
        name = platform.processor() or platform.machine() or "CPU"

    return HardwareProfile(
        backend=backend,
        device_name=name,
        accelerator_gb=total_vram_gb(device),
        ram_gb=total_ram_gb(),
        threads=os.cpu_count() or 4,
    )


def describe_device(device: torch.device) -> str:
    """One-line label for logs. Never raises, even for an absent backend."""
    if device.type == "cuda":
        try:
            return f"{device} ({torch.cuda.get_device_name(device)})"
        except Exception:
            return f"{device} (CUDA device not present)"
    if device.type == "mps":
        return "mps (Apple Silicon / Metal)"
    if device.type == "xpu":
        name = "xpu (Intel GPU)"
    if device.type == "privateuseone":
        return f"{device} (DirectML, experimental)"
    return f"cpu ({platform.processor() or 'no accelerator present'})"


def display_adapters() -> list[str]:
    """Best-effort list of the host's graphics adapters."""
    system = platform.system()
    if system == "Windows":
        try:
            import subprocess
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                text=True, stderr=subprocess.DEVNULL, timeout=10)
            return [ln.strip() for ln in out.splitlines() if ln.strip()]
        except Exception:
            return []
    if system == "Linux":
        found = []
        try:
            import subprocess
            out = subprocess.check_output(["lspci"], text=True,
                                          stderr=subprocess.DEVNULL, timeout=10)
            for line in out.splitlines():
                if "VGA" in line or "3D controller" in line:
                    found.append(line.split(": ", 1)[-1])
        except Exception:
            pass
        return found
    if system == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True, timeout=15)
            for line in out.splitlines():
                if "Chipset Model" in line:
                    return [line.split(":", 1)[-1].strip()]
        except Exception:
            pass
    return []


def cuda_build_version() -> str | None:
    return torch.version.cuda
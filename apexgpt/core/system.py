"""What the machine is doing *right now*, not just what it is.

The static spec in :mod:`.device` answers "what does this CPU support". This
module answers "how loaded is it at this moment", which is what decides whether
8 threads and batch 4 are safe to use:

===================  =====================================================
metric               source
===================  =====================================================
CPU utilisation      ``psutil``, else ``/proc/stat``, else ``GetSystemTimes``
available RAM        ``psutil``, else ``GlobalMemoryStatusEx`` / ``MemAvailable``
load average         ``os.getloadavg()`` (Linux, macOS, BSD)
disk for a path      ``shutil.disk_usage``
running processes    ``psutil`` (optional: ``pip install psutil``)
===================  =====================================================

Every probe degrades to ``None`` instead of raising, so a missing ``psutil`` or a
locked-down ``/proc`` changes how much is reported, never whether the scan works.
"""
from __future__ import annotations

import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: How long to sample CPU counters for a utilisation figure.
SAMPLE_SECONDS = 0.25


@dataclass(frozen=True)
class ProcessSnapshot:
    """One running process, as far as the platform will tell us."""
    name: str
    pid: int
    cpu_percent: float | None = None
    memory_mb: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SystemLoad:
    """A point-in-time reading of the machine's load."""
    cpu_percent: float | None = None
    load_average: tuple[float, ...] | None = None
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_percent: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_percent: float = 0.0
    disk_path: str = ""
    process_count: int | None = None
    busy_processes: list[ProcessSnapshot] = field(default_factory=list)
    memory_hogs: list[ProcessSnapshot] = field(default_factory=list)
    has_psutil: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def busy(self) -> bool:
        """Whether other work is already competing for the CPU."""
        return (self.cpu_percent or 0.0) >= 85.0

    @property
    def memory_pressure(self) -> bool:
        """Whether free RAM has fallen to the point of swapping."""
        return 0.0 < self.ram_available_gb <= 1.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["busy"] = self.busy
        data["memory_pressure"] = self.memory_pressure
        data["busy_processes"] = [p.to_dict() for p in self.busy_processes]
        data["memory_hogs"] = [p.to_dict() for p in self.memory_hogs]
        return data

    def to_text(self) -> list[str]:
        """The load block, in the same shape ``env`` prints."""
        def fmt(value, unit="%") -> str:
            return "unknown" if value is None else f"{value:.0f}{unit}"

        ram = f"{self.ram_available_gb:.1f} GB free of {self.ram_total_gb:.1f} GB"
        disk = f"{self.disk_free_gb:.1f} GB free of {self.disk_total_gb:.1f} GB"
        if self.disk_path:
            disk += f" on {self.disk_path}"
        lines = [
            f"  CPU in use      : {fmt(self.cpu_percent)}"
            + (f" (load {' '.join(f'{v:.1f}' for v in self.load_average)})"
               if self.load_average else ""),
            f"  RAM             : {ram} ({fmt(self.ram_percent)})",
            f"  disk            : {disk}",
            f"  processes       : {self.process_count if self.process_count is not None else 'unknown'}"
            f"{' (psutil)' if self.has_psutil else ' (install psutil for per-process detail)'}",
        ]
        for proc in self.busy_processes[:3]:
            cpu = "?" if proc.cpu_percent is None else f"{proc.cpu_percent:.0f}%"
            lines.append(f"    cpu  {cpu:>5}  {proc.name} (pid {proc.pid})")
        for proc in self.memory_hogs[:3]:
            mem = "?" if proc.memory_mb is None else f"{proc.memory_mb:.0f} MB"
            lines.append(f"    mem  {mem:>5}  {proc.name} (pid {proc.pid})")
        return lines


# --------------------------------------------------------------------------- #
# optional dependency
# --------------------------------------------------------------------------- #
def _psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def has_psutil() -> bool:
    return _psutil() is not None


# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #
def cpu_percent(sample: float = SAMPLE_SECONDS) -> float | None:
    """System-wide CPU utilisation, 0-100, or ``None`` if unknowable.

    Utilisation is a *ratio over time*, so it cannot be read from a single
    counter: every backend samples twice and subtracts.
    """
    psutil = _psutil()
    if psutil is not None:
        try:
            return float(psutil.cpu_percent(interval=sample))
        except Exception:
            pass

    system = platform.system()
    if system == "Linux":
        return _cpu_percent_proc_stat(sample)
    if system == "Windows":
        return _cpu_percent_windows(sample)
    try:                                     # macOS: load average over cores
        load = os.getloadavg()[0]
        return min(100.0, load / max(1, os.cpu_count() or 1) * 100.0)
    except (AttributeError, OSError):
        return None


def _cpu_percent_proc_stat(sample: float) -> float | None:
    try:
        with open("/proc/stat", encoding="utf-8") as fh:
            first = fh.readline().split()[1:]
        idle1, total1 = _split_cpu_line(first)
        time.sleep(sample)
        with open("/proc/stat", encoding="utf-8") as fh:
            second = fh.readline().split()[1:]
        idle2, total2 = _split_cpu_line(second)
        delta_total = total2 - total1
        delta_idle = idle2 - idle1
        if delta_total <= 0:
            return None
        return max(0.0, min(100.0, (1.0 - delta_idle / delta_total) * 100.0))
    except (OSError, ValueError, IndexError):
        return None


def _split_cpu_line(fields: list[str]) -> tuple[float, float]:
    values = [float(v) for v in fields]
    # user, nice, system, idle, iowait, irq, softirq, steal
    idle = values[3] + (values[4] if len(values) > 4 else 0.0)
    return idle, sum(values)


def _cpu_percent_windows(sample: float) -> float | None:
    """``GetSystemTimes`` twice: idle, kernel and user FILETIME counters."""
    try:
        import ctypes
        from ctypes import wintypes

        class _Filetime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

            def value(self) -> int:
                return (self.high << 32) | self.low

        idle, kernel, user = _Filetime(), _Filetime(), _Filetime()

        def read() -> tuple[int, int, int]:
            if not ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                raise OSError("GetSystemTimes failed")
            # kernel time includes idle time
            return idle.value(), kernel.value(), user.value()

        idle1, kernel1, user1 = read()
        time.sleep(sample)
        idle2, kernel2, user2 = read()
        total = (kernel2 - kernel1) + (user2 - user1)
        idle = idle2 - idle1
        if total <= 0:
            return None
        return max(0.0, min(100.0, (1.0 - idle / total) * 100.0))
    except Exception:
        return None


def load_average() -> tuple[float, ...] | None:
    try:
        return tuple(os.getloadavg())
    except (AttributeError, OSError):
        return None


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def available_ram_gb() -> float | None:
    """RAM that can be had without swapping, in GB."""
    psutil = _psutil()
    if psutil is not None:
        try:
            return psutil.virtual_memory().available / 1e9
        except Exception:
            pass
    if os.name == "nt":
        value = _win_available_ram_gb()
        if value is not None:
            return value
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        pass
    if platform.system() == "Darwin":
        try:
            import subprocess
            free_pages = int(subprocess.check_output(
                ["sysctl", "-n", "vm.page_free_in_files"], text=True,
                stderr=subprocess.DEVNULL, timeout=5).strip())
            return free_pages * 4096 / 1e9
        except Exception:
            return None
    return None


def _win_available_ram_gb() -> float | None:
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
            return status.ullAvailPhys / 1e9
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# disk + processes
# --------------------------------------------------------------------------- #
def disk_usage_gb(path: Path | str) -> tuple[float, float, float]:
    """``(total, free, used_percent)`` in GB for the filesystem holding ``path``."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(str(probe))
    except OSError:
        return 0.0, 0.0, 0.0
    total = usage.total / 1e9
    free = usage.free / 1e9
    percent = 100.0 * (usage.used / usage.total) if usage.total else 0.0
    return total, free, percent


def process_count() -> int | None:
    """Number of running processes, when the platform exposes it cheaply."""
    psutil = _psutil()
    if psutil is not None:
        try:
            return len(psutil.pids())
        except Exception:
            return None
    if platform.system() == "Linux":
        try:
            return len([p for p in os.listdir("/proc") if p.isdigit()])
        except OSError:
            return None
    return None


def top_processes(limit: int = 5) -> tuple[list[ProcessSnapshot], list[ProcessSnapshot]]:
    """The busiest and hungriest processes. Empty without ``psutil``.

    ``cpu_percent`` is sampled over a short interval by ``psutil``, which needs a
    prior call to produce meaningful numbers; the first call reports zeros.
    """
    psutil = _psutil()
    if psutil is None:
        return [], []
    try:
        # prime psutil's per-process CPU counters; without this the first
        # reading is always 0.0 and the "busiest" list is meaningless
        psutil.cpu_percent(interval=None)
        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
            info = proc.info
            pid = int(info.get("pid") or 0)
            # Windows exposes the kernel as pid 0 "System Idle Process" and
            # pid 4 "System"; their cpu_percent is a counter, not a rate, and
            # they would otherwise top the "busiest" list forever
            skip = pid <= 4 if os.name == "nt" else pid == 0
            if skip:
                continue
            rss = getattr(info.get("memory_info"), "rss", None)
            procs.append(ProcessSnapshot(
                name=str(info.get("name") or "?")[:32],
                pid=pid,
                cpu_percent=info.get("cpu_percent"),
                memory_mb=(rss / 1e6) if rss else None,
            ))
    except Exception:
        return [], []

    busy = sorted((p for p in procs if p.cpu_percent),
                  key=lambda p: p.cpu_percent or 0.0, reverse=True)[:limit]
    hogs = sorted((p for p in procs if p.memory_mb),
                  key=lambda p: p.memory_mb or 0.0, reverse=True)[:limit]
    return busy, hogs


def snapshot(path: Path | str | None = None,
             sample: float = SAMPLE_SECONDS) -> SystemLoad:
    """Read the whole machine once. Never raises."""
    from .device import total_ram_gb
    from .paths import DATA_DIR

    probe = Path(path) if path is not None else DATA_DIR
    notes: list[str] = []
    total_ram = total_ram_gb()
    available = available_ram_gb()
    if available is None:
        available = total_ram / 2 if total_ram else 0.0
        notes.append("available RAM is unknown; assuming half of total")
    disk_total, disk_free, disk_percent = disk_usage_gb(probe)
    busy, hogs = top_processes()
    if not has_psutil():
        notes.append("psutil is not installed: no per-process detail")
    if disk_total and disk_free < 2.0:
        notes.append(f"only {disk_free:.1f} GB free on the data volume")

    return SystemLoad(
        cpu_percent=cpu_percent(sample),
        load_average=load_average(),
        ram_total_gb=total_ram,
        ram_available_gb=available,
        ram_percent=(100.0 * (total_ram - available) / total_ram) if total_ram else 0.0,
        disk_total_gb=disk_total,
        disk_free_gb=disk_free,
        disk_percent=disk_percent,
        disk_path=str(probe),
        process_count=process_count(),
        busy_processes=busy,
        memory_hogs=hogs,
        has_psutil=has_psutil(),
        notes=notes,
    )
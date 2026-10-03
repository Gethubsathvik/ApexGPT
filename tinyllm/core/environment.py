"""One scan of the host: hardware spec, requirements, and the settings to run with.

Every front end (terminal, Tk GUI, HTTP API, Jupyter kernel) resolves its
configuration through this module, so a machine is detected exactly once and the
answer is the same everywhere::

    from tinyllm.core.environment import bootstrap
    report, settings = bootstrap()        # scan, then apply to this process
    print(report.to_text())

The scan is read-only apart from applying settings, and never raises for a
missing optional package - an absent accelerator is reported, not hidden.
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import device as dev
from .paths import ROOT

#: Requirement files scanned by :func:`collect_requirements`, by group name.
REQUIREMENT_FILES: dict[str, str] = {
    "core": "requirements.txt",
    "api": "requirements-api.txt",
    "notebook": "requirements-notebook.txt",
}

#: Groups that only some front ends need; missing entries are not fatal.
OPTIONAL_GROUPS = ("api", "notebook")

_SHELLS = ("auto", "bash", "powershell", "cmd")


# --------------------------------------------------------------------------- #
# the machine spec
# --------------------------------------------------------------------------- #
def cpu_name() -> str:
    """The host CPU's marketing name, without importing anything heavy."""
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"],
                text=True, stderr=subprocess.DEVNULL, timeout=10)
            first = out.strip().splitlines()
            if first:
                return first[0].strip()
        except Exception:
            pass
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[-1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _physical_cores() -> int | None:
    """Physical core count where the OS exposes it (logical count otherwise)."""
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object -Property "
                 "NumberOfCores -Sum).Sum"],
                text=True, stderr=subprocess.DEVNULL, timeout=10)
            value = int(out.strip())
            return value if value > 0 else None
        except Exception:
            return None
    if platform.system() == "Linux":
        try:
            out = subprocess.check_output(["lscpu"], text=True,
                                          stderr=subprocess.DEVNULL, timeout=10)
            cores = per_socket = None
            for line in out.splitlines():
                if line.startswith("Core(s) per socket"):
                    cores = int(line.split(":", 1)[1])
                elif line.startswith("Socket(s)"):
                    per_socket = int(line.split(":", 1)[1])
            if cores and per_socket:
                return cores * per_socket
        except Exception:
            return None
    return None


def tool_version(*command: str) -> str | None:
    """Version string of an external tool, or ``None`` if it is not installed.

    Used to explain *why* a backend is missing (``nvidia-smi`` and ``nvcc``
    absent is a different situation from a driver mismatch).
    """
    for name in command:
        try:
            done = subprocess.run([name, "--version"], capture_output=True,
                                  text=True, timeout=10)
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
        text = (done.stdout or done.stderr or "").strip()
        if done.returncode == 0 and text:
            return text.splitlines()[0].strip()
    return None


# --------------------------------------------------------------------------- #
# requirements
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Requirement:
    """One dependency line from a requirements file, plus what is installed."""
    name: str
    specifier: str
    group: str
    installed: str | None = None
    satisfied: bool | None = None

    @property
    def ok(self) -> bool:
        if self.satisfied is None:
            return self.installed is not None
        return self.satisfied

    @property
    def status(self) -> str:
        if self.installed is None:
            return "missing"
        return "ok" if self.ok else "outdated"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status
        return data


def _specifier_satisfied(spec_text: str, installed: str) -> bool | None:
    """Whether ``installed`` satisfies ``spec_text``; ``None`` if unknowable."""
    try:
        from packaging.requirements import Requirement as _Req
        from packaging.version import Version
    except Exception:
        return None
    try:
        return bool(_Req(spec_text).specifier.contains(Version(installed),
                                                        prereleases=True))
    except Exception:
        return None


def parse_requirement_file(path: Path, group: str) -> list[Requirement]:
    """Read one requirements file into :class:`Requirement` records."""
    if not path or not path.exists():
        return []
    found: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = line.split("[", 1)[0]
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in name:
                name = name.split(sep, 1)[0]
        name = name.strip()
        if not name:
            continue
        found.append(Requirement(name=name, specifier=line, group=group))
    return found


def collect_requirements(groups: tuple[str, ...] = ("core",)) -> list[Requirement]:
    """Every requirement across the requested groups, annotated with status.

    Reading the requirements files rather than hard-coding a list means the
    scan cannot drift away from what ``setup`` actually installs.
    """
    seen: set[str] = set()
    collected: list[Requirement] = []
    for group in groups:
        filename = REQUIREMENT_FILES.get(group)
        if not filename:
            continue
        for req in parse_requirement_file(ROOT / filename, group):
            if req.name in seen:
                continue
            seen.add(req.name)
            try:
                installed = metadata.version(req.name)
            except Exception:
                installed = None
            collected.append(Requirement(
                name=req.name, specifier=req.specifier, group=group,
                installed=installed,
                satisfied=None if installed is None
                else _specifier_satisfied(req.specifier, installed),
            ))
    return collected


# --------------------------------------------------------------------------- #
# accelerator + settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AcceleratorInfo:
    """Which GPU path this machine has, and which one is actually usable."""
    backends: list[str]
    experimental: list[str]
    selected: str
    name: str
    memory_gb: float
    cuda_available: bool
    cuda_build: str | None
    rocm_build: str | None
    mps_available: bool
    xpu_available: bool
    dml_available: bool
    adapters: list[str]
    nvidia_smi: str | None
    nvcc: str | None
    rocm_smi: str | None
    amp: bool
    checkpointing: bool
    precision: str
    notes: list[str] = field(default_factory=list)

    @property
    def cuda_usable(self) -> bool:
        return self.cuda_available

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cuda_usable"] = self.cuda_usable
        return data


@dataclass(frozen=True)
class RuntimeSettings:
    """The resolved configuration every entry point runs with."""
    device: str
    backend: str
    threads: int
    amp: bool
    checkpointing: bool
    preset: str
    batch_size: int
    block_size: int
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def preset_plan(preset: str) -> dict:
    """The shape behind a preset name.

    Imported lazily: ``features.training.service`` imports ``core``, so a
    module-level import here would be circular.
    """
    from ..features.training.service import PRESETS
    return dict(PRESETS[preset])


def resolve_settings(device: str | None = None) -> RuntimeSettings:
    """Work out the settings for this machine without changing anything."""
    torch_device = dev.get_device(device)
    hardware = dev.profile()
    amp = dev.use_amp_by_default(torch_device)
    checkpointing = dev.use_checkpointing_by_default(torch_device)
    threads = dev.configure_threads(None)
    preset = hardware.recommend_preset()
    try:
        plan = preset_plan(preset)
        batch_size = int(plan["batch_size"])
        block_size = int(plan["block_size"])
    except Exception:                      # pragma: no cover - defensive
        batch_size, block_size = 2, 64

    settings = RuntimeSettings(
        device=str(torch_device),
        backend=torch_device.type,
        threads=threads,
        amp=amp,
        checkpointing=checkpointing,
        preset=preset,
        batch_size=batch_size,
        block_size=block_size,
    )
    return replace(settings, env=environment_vars(settings))


def environment_vars(settings: RuntimeSettings) -> dict[str, str]:
    """The variables that carry these settings into child processes.

    A Jupyter kernel, a ``serve`` process or a second terminal all read these
    instead of re-deriving the configuration, which is how the notebook kernel
    ends up with the same threads and device as the shell that launched it.
    """
    return {
        "TINYLLM_DEVICE": settings.device,
        "TINYLLM_BACKEND": settings.backend,
        "TINYLLM_THREADS": str(settings.threads),
        "TINYLLM_PRESET": settings.preset,
        "TINYLLM_AMP": "1" if settings.amp else "0",
        "TINYLLM_CHECKPOINTING": "1" if settings.checkpointing else "0",
        "OMP_NUM_THREADS": str(settings.threads),
    }


def apply_settings(settings: RuntimeSettings,
                   export: bool = True) -> RuntimeSettings:
    """Apply the resolved settings to this process (and optionally children)."""
    dev.configure_threads(settings.threads)
    if export:
        os.environ.update(settings.env)
    return settings


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
@dataclass
class EnvironmentReport:
    """Everything the scan learned, in one serialisable object."""
    generated_at: str
    python: str
    python_tag: str
    platform: str
    machine: str
    cpu: str
    cores: int | None
    threads: int
    ram_gb: float
    torch_version: str
    accelerator: AcceleratorInfo
    requirements: list[Requirement]
    settings: RuntimeSettings
    notes: list[str] = field(default_factory=list)

    # ----------------------------------------------------------- properties
    @property
    def missing(self) -> list[Requirement]:
        return [r for r in self.requirements if r.installed is None]

    @property
    def outdated(self) -> list[Requirement]:
        return [r for r in self.requirements if r.status == "outdated"]

    @property
    def ok(self) -> bool:
        """True when nothing required is missing or too old."""
        return not self.missing and not self.outdated

    # -------------------------------------------------------------- renderers
    def spec_lines(self) -> list[str]:
        """The machine spec, as a small two-column table."""
        acc = self.accelerator
        gpu = ", ".join(acc.adapters) if acc.adapters else "none detected"
        if acc.cuda_build and acc.rocm_build:
            cuda = f"Not an NVIDIA CUDA build - AMD ROCm {acc.rocm_build}"
        elif acc.cuda_available:
            cuda = f"available (build {acc.cuda_build or 'unknown'})"
        elif acc.mps_available:
            cuda = "not applicable - Apple Metal (MPS) is the GPU path"
        elif acc.xpu_available:
            cuda = "not applicable - Intel XPU is the GPU path"
        elif acc.dml_available:
            cuda = "not available - optional DirectML path is installed, use --device dml"
        else:
            cuda = "not available - no NVIDIA GPU or driver on this machine"
        cores = f"{self.cores} cores / {self.threads} threads" if self.cores \
            else f"{self.threads} threads"
        memory = f"{self.ram_gb:.1f} GB RAM"
        if acc.memory_gb:
            memory += f", {acc.memory_gb:.1f} GB accelerator"
        return [
            f"  CPU            : {self.cpu}",
            f"  cores          : {cores}",
            f"  GPU            : {gpu}",
            f"  CUDA           : {cuda}",
            f"  usable backend : {acc.selected} ({acc.name})",
            f"  memory         : {memory}",
            f"  torch          : {self.torch_version}",
            f"  python         : {self.python} ({platform.python_implementation()})",
        ]

    def to_text(self) -> str:
        acc = self.accelerator
        lines = ["=" * 68, "TinyLLM environment", "=" * 68]
        lines.append("Machine spec")
        lines += self.spec_lines()
        lines += ["-" * 68, "Requirements"]
        for req in self.requirements:
            mark = "ok " if req.status == "ok" else "!! "
            got = req.installed or "MISSING"
            lines.append(f"  {mark}{req.name:<22} {req.specifier:<18} {got}")
        lines += ["-" * 68, "Resolved settings"]
        s = self.settings
        lines += [
            f"  device         : {s.device}",
            f"  torch threads  : {s.threads}",
            f"  amp            : {s.amp} (precision {acc.precision})",
            f"  checkpointing  : {s.checkpointing}",
            f"  preset         : {s.preset} (batch {s.batch_size}, "
            f"block {s.block_size})",
        ]
        for note in self.notes:
            lines.append(f"  note           : {note}")
        lines += ["-" * 68]
        lines.append("RESULT: " + ("ready" if self.ok else
                                  f"{len(self.missing)} missing, "
                                  f"{len(self.outdated)} outdated"))
        lines.append("=" * 68)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """A notebook-friendly table of the same facts."""
        acc = self.accelerator
        rows = [
            ("Python", self.python),
            ("Platform", self.platform),
            ("CPU", self.cpu),
            ("Threads", str(self.threads)),
            ("RAM", f"{self.ram_gb:.1f} GB"),
            ("GPU", ", ".join(acc.adapters) or "none detected"),
            ("CUDA", "yes" if acc.cuda_available else "no"),
            ("ROCm build", acc.rocm_build or "-"),
            ("MPS", "yes" if acc.mps_available else "no"),
            ("XPU", "yes" if acc.xpu_available else "no"),
            ("DirectML", "yes" if acc.dml_available else "no"),
            ("Selected device", f"{acc.selected} ({acc.name})"),
            ("AMP", str(acc.amp)),
            ("Gradient checkpointing", str(acc.checkpointing)),
            ("Preset", self.settings.preset),
            ("Missing packages", ", ".join(r.name for r in self.missing) or "none"),
        ]
        out = ["| Item | Value |", "|------|-------|"]
        out += [f"| {k} | {v} |" for k, v in rows]
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "machine": {
                "python": self.python,
                "python_tag": self.python_tag,
                "platform": self.platform,
                "machine": self.machine,
                "cpu": self.cpu,
                "cores": self.cores,
                "threads": self.threads,
                "ram_gb": round(self.ram_gb, 2),
                "torch": self.torch_version,
            },
            "accelerator": self.accelerator.to_dict(),
            "requirements": [r.to_dict() for r in self.requirements],
            "settings": self.settings.to_dict(),
            "notes": list(self.notes),
            "ok": self.ok,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #
def _accelerator_info(device: torch.device | None = None) -> AcceleratorInfo:
    resolved = device or dev.get_device()
    hardware = dev.profile()
    backends = dev.available_backends()
    experimental = ["dml"] if dev.dml_available() else []
    notes: list[str] = []

    cuda_build = dev.cuda_build_version()
    rocm = dev.rocm_build()
    if rocm:
        notes.append("this is an AMD ROCm build of torch, not NVIDIA CUDA")
    if not backends:
        notes.append("no accelerator backend; training runs on the CPU")
    if not dev.cuda_available() and dev.display_adapters():
        notes.append("a display adapter is present but PyTorch cannot use it; "
                     "AMD/Intel iGPUs need ROCm (Linux) or DirectML (Windows)")
    if resolved.type == "privateuseone":
        notes.append("DirectML is selected: experimental, and slower than the "
                     "CPU for training on most integrated GPUs")

    return AcceleratorInfo(
        backends=backends,
        experimental=experimental,
        selected=str(resolved),
        name=hardware.device_name if resolved.type == hardware.backend
        else dev.describe_device(resolved),
        memory_gb=dev.total_vram_gb(resolved),
        cuda_available=dev.cuda_available(),
        cuda_build=cuda_build,
        rocm_build=rocm,
        mps_available=dev.mps_available(),
        xpu_available=dev.xpu_available(),
        dml_available=dev.dml_available(),
        adapters=dev.display_adapters(),
        nvidia_smi=tool_version("nvidia-smi", "nvidia-smi.exe"),
        nvcc=tool_version("nvcc", "nvcc.exe"),
        rocm_smi=tool_version("rocm-smi", "rocm-smi.exe"),
        amp=dev.use_amp_by_default(resolved),
        checkpointing=dev.use_checkpointing_by_default(resolved),
        precision=dev.device_flavor(resolved),
        notes=notes,
    )


def scan(device: str | None = None,
         groups: tuple[str, ...] = ("core",),
         notes: tuple[str, ...] = ()) -> EnvironmentReport:
    """Read the host and return everything found, without changing state."""
    torch_device = dev.get_device(device)
    settings = resolve_settings(device)
    acc = _accelerator_info(torch_device)
    all_notes = list(acc.notes) + list(notes)
    return EnvironmentReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        python=platform.python_version(),
        python_tag=f"{sys.version_info.major}.{sys.version_info.minor}",
        platform=platform.platform(),
        machine=platform.machine(),
        cpu=cpu_name(),
        cores=_physical_cores(),
        threads=os.cpu_count() or 1,
        ram_gb=dev.total_ram_gb(),
        torch_version=torch.__version__,
        accelerator=acc,
        requirements=collect_requirements(groups),
        settings=settings,
        notes=all_notes,
    )


def bootstrap(device: str | None = None,
              groups: tuple[str, ...] = ("core",),
              export: bool = True) -> tuple[EnvironmentReport, RuntimeSettings]:
    """Scan, then apply. The single entry point every front end should use."""
    report = scan(device=device, groups=groups)
    return report, apply_settings(report.settings, export=export)


# --------------------------------------------------------------------------- #
# shell integration
# --------------------------------------------------------------------------- #
def detect_shell() -> str:
    """Best guess at the user's shell, for ``--export`` output."""
    if os.name == "nt":
        return "powershell"
    return "bash"


def shell_exports(settings: RuntimeSettings, shell: str = "auto") -> list[str]:
    """Lines that carry the settings into the surrounding shell."""
    if shell == "auto":
        shell = detect_shell()
    if shell not in _SHELLS:
        raise ValueError(f"unknown shell {shell!r}; choose from {_SHELLS}")
    pairs = settings.env.items()
    if shell == "powershell":
        return [f'$env:{key} = "{value}"' for key, value in pairs]
    if shell == "cmd":
        return [f"set {key}={value}" for key, value in pairs]
    return [f"export {key}={value}" for key, value in pairs]


def write_report(report: EnvironmentReport, path: Path | str) -> Path:
    """Persist an existing report to ``path``."""
    out = Path(path)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json(), encoding="utf-8")
    return out


def write_json(path: Path | str) -> Path:
    """Scan once more and persist the result to ``path``."""
    report, _ = bootstrap(export=False)
    return write_report(report, path)

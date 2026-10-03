"""Hardware-portability tests: device selection, precision policy, setup.

These run on any machine. Backends that are absent are simulated so the logic
is covered even on a CPU-only host.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apexgpt.core import device as dev
from apexgpt.core.device import (HardwareProfile, autocast_dtype, available_backends,
                                 configure_threads, cuda_available,
                                 describe_device, device_flavor, dml_available,
                                 get_device, mps_available, profile, rocm_build,
                                 supports_bf16, total_ram_gb, total_vram_gb,
                                 use_amp_by_default, use_checkpointing_by_default,
                                 xpu_available)
from apexgpt.tools import setup as bootstrap


# --------------------------------------------------------------- backends
def test_every_backend_probe_is_callable():
    for probe in (cuda_available, mps_available, xpu_available, dml_available,
                  rocm_build):
        probe()


def test_available_backends_always_includes_something_runnable():
    backends = available_backends()
    assert isinstance(backends, list)
    assert get_device("auto").type in (backends + ["cpu"])


def test_directml_is_never_auto_selected():
    """Opt-in only: DirectML loses to the CPU for training on most iGPUs."""
    assert "dml" not in available_backends()
    if dml_available():
        assert "dml" in available_backends(include_experimental=True)


def test_directml_device_is_privateuseone_with_amp_off():
    device = torch.device("privateuseone:0")
    assert device_flavor(device) == "float32"
    assert use_amp_by_default(device) is False
    assert use_checkpointing_by_default(device) is False
    assert "DirectML" in describe_device(device)


def test_rocm_build_is_a_string_or_none():
    rocm = rocm_build()
    assert rocm is None or isinstance(rocm, str)


def test_cpu_is_always_selectable():
    assert get_device("cpu").type == "cpu"


def test_auto_never_returns_a_bogus_backend():
    assert get_device("auto").type in {"cpu", "cuda", "mps", "xpu"}


def test_explicit_missing_backend_raises_instead_of_silent_fallback():
    """A typo or absent accelerator must not quietly turn into a CPU run."""
    for name in ("cuda", "mps", "xpu", "dml"):
        if name not in available_backends(include_experimental=True):
            with pytest.raises(RuntimeError, match="not available|torch-directml"):
                get_device(name)


def test_device_index_form_is_accepted():
    assert get_device("cpu").index is None
    # a multi-gpu host should accept cuda:0
    if cuda_available():
        assert get_device("cuda:0").type == "cuda"


# ---------------------------------------------------------------- precision
def test_autocast_dtype_by_backend():
    assert autocast_dtype(torch.device("cuda")) is torch.float16
    assert autocast_dtype(torch.device("mps")) is torch.float16
    assert autocast_dtype(torch.device("xpu")) is torch.float16
    assert autocast_dtype(torch.device("cpu")) is torch.bfloat16


def test_device_flavor_labels():
    assert device_flavor(torch.device("cuda")) == "float16"
    assert device_flavor(torch.device("cpu")) == "bfloat16"


def test_amp_is_recommended_off_on_a_cpu_without_native_bf16():
    """Zen2 and older emulate bf16, which is dramatically slower."""
    if not supports_bf16(torch.device("cpu")):
        assert use_amp_by_default(torch.device("cpu")) is False
    else:                                    # pragma: no cover - newer CPU
        assert use_amp_by_default(torch.device("cpu")) is True


def test_amp_is_recommended_on_for_accelerators():
    for name in ("cuda", "mps", "xpu"):
        assert use_amp_by_default(torch.device(name)) is True


def test_checkpointing_is_off_on_cpu_and_memory_aware_on_cuda():
    assert use_checkpointing_by_default(torch.device("cpu")) is False
    # on CUDA the decision depends on VRAM, so just assert it is a bool
    assert isinstance(use_checkpointing_by_default(torch.device("cuda")), bool)


# ------------------------------------------------------------------ threads
def test_configure_threads_clamps_to_real_cores():
    import os
    cores = os.cpu_count() or 4
    assert configure_threads(None) == cores
    assert configure_threads(10_000) == cores
    assert configure_threads(0) == 1
    assert configure_threads(2) == min(2, cores)


def test_total_ram_gb_is_plausible():
    ram = total_ram_gb()
    assert ram >= 0.5, f"implausible RAM reading: {ram}"


def test_total_vram_gb_on_cpu_is_zero():
    assert total_vram_gb(torch.device("cpu")) == 0.0


def test_describe_device_is_human_readable():
    for name in ("cpu", "cuda", "mps", "xpu"):
        assert describe_device(torch.device(name))


# ------------------------------------------------------------------ profile
@pytest.mark.parametrize("backend,gb,threads,expected", [
    ("cuda", 24.0, 16, "full"),
    ("cuda", 12.0, 16, "full"),
    ("cuda", 8.0, 16, "medium"),
    ("cuda", 6.0, 8, "medium"),
    ("cpu", 0.0, 8, "cpu-tiny"),
    ("cpu", 0.0, 2, "smoke"),
    ("mps", 0.0, 8, "cpu-tiny"),
])
def test_profile_recommends_a_preset_that_fits(backend, gb, threads, expected):
    hw = HardwareProfile(backend=backend, device_name="test",
                         accelerator_gb=gb, ram_gb=16.0, threads=threads)
    assert hw.recommend_preset() == expected


def test_profile_of_this_machine_describes_itself():
    hw = profile()
    assert hw.backend in ("cpu", "cuda", "mps", "xpu")
    assert hw.threads >= 1
    assert "threads" in hw.describe()


# ---------------------------------------------------------------- bootstrap
def test_setup_detects_a_supported_backend():
    backend = bootstrap.detect_backend()
    assert backend in ("cpu", "cuda", "rocm", "xpu", "mps")


def test_setup_tool_probe_survives_missing_executables():
    """nvidia-smi/rocm-smi are usually absent; probing must not raise."""
    assert bootstrap._tool_works("definitely-not-a-real-tool-xyz") is False


@pytest.mark.parametrize("backend,expect_index", [
    ("cpu", "cpu"),
    ("cuda", "cu124"),
    ("rocm", "rocm6.2"),
    ("xpu", "download.pytorch.org/whl/xpu"),
    ("mps", None),
    ("dml", None),
])
def test_setup_picks_the_right_wheel_index(backend, expect_index):
    spec, index = bootstrap.torch_spec(backend, "124")
    if backend == "dml":
        assert spec == "torch-directml"
    else:
        assert spec == "torch"
    if expect_index is None:
        assert index is None
    else:
        assert expect_index in index


def test_every_wheel_index_is_a_real_pytorch_url():
    """A wrong host 404s at install time, which is a confusing way to find out."""
    for backend in ("cpu", "cuda", "rocm", "xpu"):
        _, index = bootstrap.torch_spec(backend, "124")
        assert index.startswith("https://download.pytorch.org/whl/")


def test_rocm_index_uses_the_rocm_version_not_the_cuda_tag():
    _, default = bootstrap.torch_spec("rocm", "124")
    _, overridden = bootstrap.torch_spec("rocm", "124", "6.1")
    assert default.endswith("rocm6.2") and overridden.endswith("rocm6.1")


def test_setup_rejects_an_unknown_backend():
    with pytest.raises(ValueError):
        bootstrap.torch_spec("quantum", "124")


def test_setup_plan_installs_torch_then_requirements():
    plan = bootstrap.build_plan("cpu", "124")
    assert len(plan) == 2
    assert plan[0][0:2] == ["pip", "install"]
    assert "torch" in plan[0]
    assert plan[1][-1] == "requirements.txt"


def test_setup_plan_appends_optional_groups():
    plan = bootstrap.build_plan("cpu", "124", groups=("notebook", "api"))
    assert len(plan) == 4
    assert plan[2][-1] == "requirements-notebook.txt"
    assert plan[3][-1] == "requirements-api.txt"


def test_setup_rejects_directml_off_windows(monkeypatch):
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    assert bootstrap.main(["--backend", "dml"]) == 1


def test_setup_plan_is_printable_without_installing(capsys):
    assert bootstrap.main([]) == 0
    out = capsys.readouterr().out
    assert "pip install" in out
    assert "--install" in out


def test_setup_rejects_mps_on_linux(monkeypatch):
    import platform as plat
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap, "detect_backend", lambda: "mps")
    assert bootstrap.main(["--backend", "mps"]) == 1


def test_setup_rejects_cuda_on_macos(monkeypatch):
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Darwin")
    assert bootstrap.main(["--backend", "cuda"]) == 1
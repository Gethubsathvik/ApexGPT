"""Environment scan tests: hardware spec, requirement collection, settings.

These run anywhere. Backends and packages are simulated where needed so the
logic is covered even on a machine that has neither a GPU nor Jupyter.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyllm.core import device as dev
from tinyllm.core import environment as env
from tinyllm.core.environment import (Requirement, bootstrap, collect_requirements,
                                     cpu_name, parse_requirement_file, preset_plan,
                                     resolve_settings, scan, shell_exports,
                                     tool_version, write_report)
from tinyllm.tools import env as env_cli

ALL_GROUPS = ("core", "api", "notebook")


# --------------------------------------------------------------- the machine
def test_cpu_name_is_a_string_not_an_error():
    assert isinstance(cpu_name(), str) and cpu_name()


def test_a_missing_tool_reports_none_rather_than_raising():
    assert tool_version("definitely-not-a-real-tool-xyz") is None


def test_tool_version_reads_an_executable_that_exists():
    assert tool_version("python", "python3") is not None or os.name == "nt"


def test_scan_reports_physical_cores_and_threads():
    report = scan()
    assert report.threads >= 1
    assert report.cores is None or 1 <= report.cores <= report.threads
    assert report.ram_gb > 0


def test_scan_pins_the_cpu_torch_and_python_versions():
    report = scan()
    assert report.torch_version == torch.__version__
    assert report.python == ".".join(str(p) for p in sys.version_info[:3])
    assert report.python_tag == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_selected_device_is_one_this_host_can_actually_run():
    report = scan()
    assert report.accelerator.selected == str(dev.get_device("auto"))
    assert report.accelerator.selected.split(":")[0] in dev.available_backends() + ["cpu"]


def test_cuda_is_never_claimed_on_a_host_without_it():
    """The AMD iGPU case: an adapter exists, CUDA must still read false."""
    report = scan()
    if not torch.cuda.is_available():
        assert report.accelerator.cuda_available is False
        assert "not available" in "\n".join(report.spec_lines())


def test_rocm_build_is_reported_when_present(monkeypatch):
    """A ROCm build reports itself as cuda, so the note has to disambiguate."""
    monkeypatch.setattr(dev, "rocm_build", lambda: "6.2.0")
    report = scan()
    assert report.accelerator.rocm_build == "6.2.0"
    assert any("ROCm" in note for note in report.notes)


def test_spec_lines_mention_cpu_gpu_and_cuda():
    """The machine spec is the three-row table the README documents."""
    lines = env.scan().spec_lines()
    joined = " ".join(lines).lower()
    assert "cpu" in joined and "gpu" in joined and "cuda" in joined
    assert any(line.strip().startswith("usable backend") for line in lines)


# -------------------------------------------------------------- requirements
def test_requirement_file_parsing_ignores_comments_and_options(tmp_path):
    path = tmp_path / "reqs.txt"
    path.write_text("# a comment\n\ntorch>=2.4\nuvicorn[standard]>=0.27\n"
                    "-r other.txt\nnumpy == 1.26\n", encoding="utf-8")
    found = parse_requirement_file(path, "core")
    names = [r.name for r in found]
    assert names == ["torch", "uvicorn", "numpy"]
    assert "uvicorn" in found[1].specifier


def test_parse_requirement_file_on_a_missing_path_is_empty(tmp_path):
    assert parse_requirement_file(tmp_path / "nope.txt", "core") == []


def test_collect_requirements_reads_the_real_files():
    core = collect_requirements(("core",))
    names = [r.name for r in core]
    assert "torch" in names and "transformers" in names
    assert all(r.group == "core" for r in core)
    # torch is installed in any environment that can import it
    torch_req = next(r for r in core if r.name == "torch")
    assert torch_req.installed == torch.__version__
    assert torch_req.status == "ok"


def test_collect_requirements_merges_groups_without_duplicates():
    every = collect_requirements(ALL_GROUPS)
    names = [r.name for r in every]
    assert len(names) == len(set(names))
    groups = {r.group for r in every}
    assert {"core", "api", "notebook"} <= groups


def test_requirement_status_distinguishes_missing_from_outdated():
    missing = Requirement(name="ghost", specifier="ghost>=1", group="core")
    assert missing.status == "missing" and missing.ok is False
    outdated = Requirement(name="torch", specifier="torch>=999", group="core",
                           installed="2.0.0", satisfied=False)
    assert outdated.status == "outdated" and outdated.ok is False
    fine = Requirement(name="torch", specifier="torch>=1", group="core",
                       installed="2.0.0", satisfied=True)
    assert fine.status == "ok" and fine.ok is True


def test_satisfied_is_none_when_packaging_is_unavailable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("packaging"):
            raise ImportError("no packaging")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert env._specifier_satisfied("torch>=2.4", "2.0.0") is None


# ----------------------------------------------------------------- settings
def test_resolve_settings_matches_the_device_helpers():
    settings = resolve_settings()
    device = dev.get_device("auto")
    assert settings.backend == device.type
    assert settings.amp == dev.use_amp_by_default(device)
    assert settings.checkpointing == dev.use_checkpointing_by_default(device)
    assert settings.threads == dev.configure_threads(None)


def test_resolve_settings_clamps_threads_to_the_core_count():
    settings = replace(resolve_settings(), threads=99_999)
    assert dev.configure_threads(settings.threads) == (os.cpu_count() or 4)


def test_settings_preset_matches_the_hardware_recommendation():
    settings = resolve_settings()
    assert settings.preset == dev.profile().recommend_preset()
    plan = preset_plan(settings.preset)
    assert plan["batch_size"] == settings.batch_size
    assert plan["block_size"] == settings.block_size


def test_cpu_settings_agree_with_the_measured_zen2_policy():
    """On this class of CPU, AMP and checkpointing must be off by default."""
    settings = resolve_settings("cpu")
    if not dev.supports_bf16(torch.device("cpu")):
        assert settings.amp is False
    assert settings.checkpointing is False


def test_environment_vars_carry_the_settings_to_children():
    settings = resolve_settings("cpu")
    variables = env.environment_vars(settings)
    assert variables["TINYLLM_DEVICE"] == "cpu"
    assert variables["TINYLLM_THREADS"] == str(settings.threads)
    assert variables["OMP_NUM_THREADS"] == str(settings.threads)
    assert variables["TINYLLM_PRESET"] == settings.preset
    assert all(isinstance(v, str) for v in variables.values())


def test_apply_settings_configures_threads_and_exports(monkeypatch):
    import os
    monkeypatch.delenv("TINYLLM_DEVICE", raising=False)
    settings = resolve_settings("cpu")
    applied = env.apply_settings(settings)
    assert applied.device == "cpu"
    assert os.environ["TINYLLM_DEVICE"] == "cpu"
    assert torch.get_num_threads() == settings.threads


def test_apply_settings_can_skip_the_environment(monkeypatch):
    import os
    monkeypatch.delenv("TINYLLM_DEVICE", raising=False)
    env.apply_settings(resolve_settings("cpu"), export=False)
    assert "TINYLLM_DEVICE" not in os.environ


def test_bootstrap_scans_and_applies_in_one_call():
    report, settings = bootstrap(device="cpu", export=False)
    assert report.settings.device == settings.device == "cpu"
    assert report.settings.threads == settings.threads


def test_scan_rejects_an_absent_backend_instead_of_falling_back():
    for name in ("cuda", "mps", "xpu", "dml"):
        if name not in dev.available_backends(include_experimental=True):
            with pytest.raises(RuntimeError, match="not available|torch-directml"):
                scan(device=name)


# ------------------------------------------------------------------ reports
def test_report_renders_text_markdown_and_json():
    report = scan()
    text = report.to_text()
    assert "Machine spec" in text and "Requirements" in text
    assert "Resolved settings" in text
    assert report.to_markdown().count("|") > 10
    payload = json.loads(report.to_json())
    assert payload["machine"]["cpu"] == report.cpu
    assert payload["settings"]["device"] == report.settings.device
    assert payload["accelerator"]["cuda_available"] == report.accelerator.cuda_available
    assert isinstance(payload["requirements"], list)


def test_report_ok_tracks_missing_and_outdated(tmp_path):
    report = scan()
    assert report.ok is (not report.missing and not report.outdated)
    report.requirements.append(Requirement("ghost", "ghost>=1", "core"))
    assert report.ok is False
    assert any(r.name == "ghost" for r in report.missing)


def test_write_report_persists_json(tmp_path):
    report = scan()
    path = write_report(report, tmp_path / "nested" / "env.json")
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["ok"] == report.ok


# ------------------------------------------------------------------ shells
@pytest.mark.parametrize("shell,prefix", [
    ("bash", "export TINYLLM_DEVICE="),
    ("cmd", "set TINYLLM_DEVICE="),
    ("powershell", '$env:TINYLLM_DEVICE = "'),
])
def test_shell_exports_per_dialect(shell, prefix):
    lines = shell_exports(resolve_settings("cpu"), shell)
    assert any(line.startswith(prefix) for line in lines)
    for line in lines:
        name = re.sub(r"^(export |set |\$env:)", "", line.split("=")[0]).strip()
        assert name.isidentifier(), f"bad variable name in {shell}: {line}"


def test_shell_export_auto_detects_something_real():
    assert shell_exports(resolve_settings("cpu"), "auto")


def test_shell_export_rejects_an_unknown_shell():
    with pytest.raises(ValueError, match="unknown shell"):
        shell_exports(resolve_settings("cpu"), "tcsh")


# ---------------------------------------------------------------------- cli
def test_env_cli_prints_the_report(capsys):
    from tinyllm.tools.env import main
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Machine spec" in out
    assert "python -m tinyllm lab" in out


def test_env_cli_json_is_parseable(capsys):
    from tinyllm.tools.env import main
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["device"]


def test_env_cli_check_passes_on_this_environment(capsys):
    from tinyllm.tools.env import main
    assert main(["--check", "--json"]) == 0


def test_env_cli_export_prints_shell_lines(capsys):
    from tinyllm.tools.env import main
    assert main(["--export", "--shell", "bash", "--json"]) == 0
    assert "export TINYLLM_DEVICE=" in capsys.readouterr().out


def test_env_cli_saves_a_report(tmp_path, capsys):
    from tinyllm.tools.env import main
    target = tmp_path / "env.json"
    assert main(["--json", "--save", str(target)]) == 0
    capsys.readouterr()
    assert json.loads(target.read_text(encoding="utf-8"))["machine"]["cpu"]


def test_env_cli_rejects_an_absent_backend(capsys):
    from tinyllm.tools.env import main
    assert main(["--device", "definitely-not-a-backend"]) == 1
    assert "not available" in capsys.readouterr().err


def test_env_cli_parser_exposes_the_documented_flags():
    parser = env_cli.build_parser()
    args = parser.parse_args(["--device", "cpu", "--export", "--shell", "powershell",
                              "--check", "--groups", "core,notebook", "--no-apply"])
    assert args.device == "cpu" and args.export and args.check
    assert args.shell == "powershell"
    assert args.groups == "core,notebook"
    assert args.no_apply is True

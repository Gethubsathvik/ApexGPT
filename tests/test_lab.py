"""Jupyter Lab front end: kernel registration, notebook integrity, CLI wiring.

The Lab tests that need jupyterlab itself are skipped when it is absent, but
the kernel-spec and notebook-structure tests always run - they are the ones
that catch a notebook which would fail to open.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apexgpt.core.environment import resolve_settings
from apexgpt.tools import lab
from apexgpt.tools.lab import (KERNEL_DISPLAY_NAME, KERNEL_NAME, NOTEBOOK_DIR,
                               kernel_dir, lab_command, module_version,
                               new_notebook, notebook_files, read_kernel_spec,
                               validate_notebooks, write_kernel_spec)

jupyterlab_installed = lab.lab_installed()
needs_lab = pytest.mark.skipif(not jupyterlab_installed,
                               reason="jupyterlab is not installed "
                                      "(python -m apexgpt lab --install)")


# ------------------------------------------------------------------- kernel
def test_kernel_spec_carries_the_resolved_settings(tmp_path):
    settings = resolve_settings("cpu")
    path = write_kernel_spec(settings, base=tmp_path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert spec["name"] == KERNEL_NAME
    assert spec["display_name"] == KERNEL_DISPLAY_NAME
    assert spec["language"] == "python"
    assert spec["argv"][1:3] == ["-m", "ipykernel_launcher"]
    assert sys.executable in spec["argv"][0]
    assert spec["env"]["APEXGPT_DEVICE"] == "cpu"
    assert spec["env"]["OMP_NUM_THREADS"] == str(settings.threads)


def test_kernel_spec_is_rewritten_not_appended(tmp_path):
    write_kernel_spec(resolve_settings("cpu"), base=tmp_path)
    path = write_kernel_spec(resolve_settings("cpu"), base=tmp_path)
    assert read_kernel_spec(base=tmp_path) is not None
    assert path.parent == kernel_dir(base=tmp_path)


def test_kernel_dir_defaults_to_a_jupyter_data_path():
    path = kernel_dir()
    assert path.name == KERNEL_NAME
    assert "kernels" in path.parts


# ----------------------------------------------------------------- notebooks
def test_notebooks_are_present():
    files = notebook_files()
    assert [p.name for p in files] == ["00_environment.ipynb", "01_train.ipynb",
                                       "02_inference.ipynb"]


def test_shipped_notebooks_are_healthy():
    assert validate_notebooks() == []


def test_notebook_validation_catches_broken_files(tmp_path):
    broken = tmp_path / "broken.ipynb"
    broken.write_text("{not json", encoding="utf-8")
    assert any("invalid JSON" in p for p in validate_notebooks(tmp_path))

    empty = tmp_path / "empty.ipynb"
    empty.write_text(json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 4,
                                 "metadata": {}}), encoding="utf-8")
    problems = validate_notebooks(tmp_path)
    assert any("no code cells" in p for p in problems)
    assert any("kernel is" in p for p in problems)


def test_notebook_validation_catches_a_cell_that_no_longer_parses(tmp_path):
    path = tmp_path / "syntax.ipynb"
    path.write_text(json.dumps(new_notebook([("code", "def broken(:\n    pass")])),
                    encoding="utf-8")
    assert any("does not parse" in p for p in validate_notebooks(tmp_path))


def test_notebook_validation_flags_stored_outputs(tmp_path):
    document = new_notebook([("code", "1 + 1")])
    document["cells"][0]["outputs"] = [{"output_type": "stream", "text": "2"}]
    path = tmp_path / "dirty.ipynb"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert any("stored outputs" in p for p in validate_notebooks(tmp_path))


def test_notebooks_match_their_sources_in_this_module():
    """Regenerating the notebooks must be a no-op: they are generated files."""
    from apexgpt.tools.notebook_sources import build_all

    built = build_all()
    assert sorted(built) == [p.name for p in notebook_files()]
    for name, document in built.items():
        on_disk = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
        assert on_disk == document, (
            f"{name} differs from apexgpt/tools/notebook_sources.py - "
            f"regenerate with: python -m apexgpt.tools.notebook_sources")


def test_new_notebook_shape_is_valid_nbformat():
    document = new_notebook([("markdown", "# title"), ("code", "print(1)")])
    assert document["nbformat"] == 4
    assert [c["cell_type"] for c in document["cells"]] == ["markdown", "code"]
    assert document["cells"][1]["outputs"] == []
    assert document["cells"][1]["execution_count"] is None
    assert document["metadata"]["kernelspec"]["name"] == KERNEL_NAME
    # nbformat validates it when available
    try:
        import nbformat
    except ImportError:
        pytest.skip("nbformat is not installed")
    nbformat.validate(nbformat.reads(json.dumps(document), as_version=4))


@pytest.mark.skipif(not (module_version("nbconvert") and lab.ipykernel_installed()),
                    reason="nbconvert or ipykernel is absent")
def test_environment_notebook_executes_in_the_registered_kernel(tmp_path):
    """End to end: kernel spec -> kernel process -> notebook cells.

    Only the environment notebook runs here. The training and inference
    notebooks need the 1.3 GB corpus and a checkpoint, so they are exercised
    by hand rather than by the test suite.
    """
    script = tmp_path / "run.py"
    script.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "import nbformat\n"
        "from nbclient import NotebookClient\n"
        f"nb = nbformat.read(Path(r'{NOTEBOOK_DIR / '00_environment.ipynb'}'), "
        "as_version=4)\n"
        f"NotebookClient(nb, timeout=600, kernel_name={KERNEL_NAME!r}).execute()\n"
        "text = '\\n'.join(str(o) for c in nb.cells for o in c.get('outputs', []))\n"
        "assert 'ApexGPT environment' in text, text[:2000]\n"
        "print('executed', len(nb.cells), 'cells')\n",
        encoding="utf-8")
    write_kernel_spec(resolve_settings("cpu"))
    done = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr


# ----------------------------------------------------------------- lab cli
def test_lab_command_is_a_jupyterlab_invocation():
    cmd = lab_command(NOTEBOOK_DIR, 8890, "0.0.0.0", no_browser=True)
    assert cmd[1:3] == ["-m", "jupyterlab"]
    assert "--port=8890" in cmd and "--ip=0.0.0.0" in cmd
    assert "--no-browser" in cmd
    assert any(a.startswith("--notebook-dir=") for a in cmd)


def test_lab_command_omits_no_browser_when_not_asked():
    assert "--no-browser" not in lab_command(NOTEBOOK_DIR, 8888, "127.0.0.1", False)


def test_lab_cli_list_reports_notebooks_and_kernel(capsys):
    assert lab.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "00_environment.ipynb" in out
    assert "kernel" in out


def test_lab_cli_registers_the_kernel_without_starting_lab(tmp_path, capsys,
                                                           monkeypatch):
    # what this tests is the kernel spec that gets written, so the lab is
    # stubbed as present: whether jupyterlab is installed on the machine running
    # the suite is a separate question, answered by its own test below
    monkeypatch.setattr(lab, "lab_installed", lambda: True)
    monkeypatch.setattr(lab, "ipykernel_installed", lambda: True)
    monkeypatch.setattr(lab, "kernel_dir", lambda *a, **k: tmp_path / KERNEL_NAME)
    assert lab.main(["--register-only"]) == 0
    out = capsys.readouterr().out
    assert "kernel registered" in out
    spec = json.loads((tmp_path / KERNEL_NAME / "kernel.json").read_text("utf-8"))
    assert spec["env"]["APEXGPT_PRESET"]


def test_lab_cli_reports_a_missing_install_instead_of_crashing(monkeypatch, capsys):
    monkeypatch.setattr(lab, "lab_installed", lambda: False)
    monkeypatch.setattr(lab, "ipykernel_installed", lambda: False)
    assert lab.main([]) == 1
    out = capsys.readouterr().out
    assert "not installed" in out
    assert "--install" in out


def test_lab_cli_check_passes_when_lab_is_installed(capsys):
    if not lab.lab_installed():
        pytest.skip("jupyterlab is not installed")
    assert lab.main(["--check"]) == 0
    assert "usable" in capsys.readouterr().out


@needs_lab
def test_jupyterlab_is_importable_and_has_a_version():
    assert lab.lab_version()
    assert lab.ipykernel_installed()


@needs_lab
def test_jupyter_lab_lists_the_registered_kernel():
    """The kernel spec must be discoverable by jupyter, not just written."""
    from jupyter_client.kernelspec import KernelSpecManager

    write_kernel_spec(resolve_settings("cpu"))
    manager = KernelSpecManager()
    assert KERNEL_NAME in manager.find_kernel_specs()
    spec = manager.get_kernel_spec(KERNEL_NAME)
    assert spec.language == "python"
    assert sys.executable in spec.argv
    assert spec.env.get("APEXGPT_DEVICE") == "cpu"

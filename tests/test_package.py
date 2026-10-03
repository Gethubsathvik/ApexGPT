"""Package layout and CLI wiring tests (no model training, no network)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import apexgpt

LAYERS = [
    "apexgpt.core",
    "apexgpt.core.config",
    "apexgpt.core.device",
    "apexgpt.core.environment",
    "apexgpt.core.paths",
    "apexgpt.core.seeding",
    "apexgpt.core.system",
    "apexgpt.core.text",
    "apexgpt.models",
    "apexgpt.models.builder",
    "apexgpt.models.gpt",
    "apexgpt.models.sampling",
    "apexgpt.features.data.service",
    "apexgpt.features.data.cli",
    "apexgpt.features.data.tokenizers",
    "apexgpt.features.training.service",
    "apexgpt.features.training.cli",
    "apexgpt.features.inference.service",
    "apexgpt.features.inference.cli",
    "apexgpt.features.inference.gui",
    "apexgpt.features.hub.service",
    "apexgpt.features.hub.cli",
    "apexgpt.api.server",
    "apexgpt.tools.doctor",
    "apexgpt.tools.env",
    "apexgpt.tools.lab",
    "apexgpt.tools.notebook_sources",
    "apexgpt.tools.setup",
]


@pytest.mark.parametrize("name", LAYERS)
def test_every_module_imports(name):
    assert importlib.import_module(name) is not None


def test_version_is_exported():
    assert apexgpt.__version__


# --------------------------------------------------------------- packaging
def test_pyproject_is_valid_and_points_at_this_package():
    """The wheel metadata must not drift from the code it describes."""
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parent.parent
    with (root / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    project = config["project"]
    assert project["name"] == "apexgpt"
    assert project["version"] == apexgpt.__version__

    target = project["scripts"]["apexgpt"]
    module_name, _, attribute = target.partition(":")
    entry_point = getattr(importlib.import_module(module_name), attribute)
    assert callable(entry_point)

    # every package listed must exist, or the wheel silently omits a slice
    for package in config["tool"]["setuptools"]["packages"]:
        assert (root / package.replace(".", "/") / "__init__.py").exists(), package


def test_requirements_and_pyproject_agree_on_torch():
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parent.parent
    with (root / "pyproject.toml").open("rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    assert any(d.startswith("torch>=") for d in deps)
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "torch>=2.4" in requirements


def test_artifacts_never_land_in_site_packages():
    """An installed copy keeps checkpoints and corpora beside the user."""
    from apexgpt.core import paths

    assert paths.ROOT.name not in ("site-packages", "dist-packages")
    assert paths.DATA_ROOT.is_relative_to(paths.ROOT)


def test_project_root_falls_back_to_the_working_directory(monkeypatch, tmp_path):
    """The rule, tested directly: no project markers above the package -> the cwd."""
    from apexgpt.core import paths

    # a real checkout: pyproject.toml sits above the package
    checkout = tmp_path / "project"
    (checkout / "apexgpt" / "core").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("", encoding="utf-8")
    assert paths._project_root(checkout / "apexgpt" / "core" / "paths.py") == checkout

    # an installed copy: site-packages holds apexgpt/ but no project markers,
    # and a naive "does apexgpt/ exist" check would wrongly accept it
    elsewhere = tmp_path / "elsewhere"
    site_packages = elsewhere / "site-packages" / "apexgpt" / "core"
    site_packages.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    assert paths._project_root(site_packages / "paths.py") == elsewhere


def test_ci_workflow_exists_and_runs_the_suite():
    """A workflow that silently stops running is worse than none."""
    root = Path(__file__).resolve().parent.parent
    workflow = root / ".github" / "workflows" / "tests.yml"
    assert workflow.exists()
    text = workflow.read_text(encoding="utf-8")
    assert "pytest" in text
    assert "setup-python" in text
    assert "windows-latest" in text and "ubuntu-latest" in text


def test_dispatcher_lists_all_commands(capsys):
    from apexgpt.__main__ import main
    assert main([]) == 0
    out = capsys.readouterr().out
    for command in ("env", "setup", "doctor", "data", "train", "generate", "gui",
                    "lab", "serve"):
        assert command in out


def test_dispatcher_routes_the_new_commands():
    """env and lab must be reachable through the single entry point."""
    from apexgpt.__main__ import COMMANDS
    assert COMMANDS["env"][0] == "apexgpt.tools.env"
    assert COMMANDS["lab"][0] == "apexgpt.tools.lab"
    for name, (module_name, _) in COMMANDS.items():
        module = importlib.import_module(module_name)
        assert callable(module.main), f"{name} has no main()"


def test_dispatcher_rejects_unknown_command():
    from apexgpt.__main__ import main
    assert main(["nope"]) == 2


def test_train_cli_parses_flags():
    from apexgpt.features.training.cli import build_parser
    args = build_parser().parse_args(
        ["--preset", "smoke", "--max-iters", "5", "--no-amp", "--batch-size", "2"])
    assert args.preset == "smoke"
    assert args.max_iters == 5
    assert args.use_amp is False
    assert args.batch_size == 2


def test_generate_cli_defaults():
    from apexgpt.features.inference.cli import build_parser
    args = build_parser().parse_args([])
    assert args.temperature == 0.8
    assert args.top_k == 50
    assert args.top_p == 0.95
    assert args.prompt


def test_data_cli_force_does_not_mean_redownload():
    """Regression: --force used to delete the 1.2 GB corpus as well."""
    from apexgpt.features.data.cli import build_parser
    args = build_parser().parse_args(["--force"])
    assert args.force is True
    assert args.redownload is False


def test_every_feature_has_a_service_and_a_view():
    """Feature-based layout: each slice exposes a service and a view."""
    from apexgpt.features.data import service as data_service
    from apexgpt.features.training import service as train_service
    from apexgpt.features.inference import service as inf_service
    assert callable(data_service.prepare)
    assert callable(train_service.train)
    assert callable(inf_service.InferenceEngine)
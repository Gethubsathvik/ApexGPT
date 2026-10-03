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
    "apexgpt.models",
    "apexgpt.models.builder",
    "apexgpt.models.gpt",
    "apexgpt.models.sampling",
    "apexgpt.features.data.service",
    "apexgpt.features.data.cli",
    "apexgpt.features.training.service",
    "apexgpt.features.training.cli",
    "apexgpt.features.inference.service",
    "apexgpt.features.inference.cli",
    "apexgpt.features.inference.gui",
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
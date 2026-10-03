"""Package layout and CLI wiring tests (no model training, no network)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tinyllm

LAYERS = [
    "tinyllm.core",
    "tinyllm.core.config",
    "tinyllm.core.device",
    "tinyllm.core.environment",
    "tinyllm.core.paths",
    "tinyllm.core.seeding",
    "tinyllm.models",
    "tinyllm.models.builder",
    "tinyllm.models.gpt",
    "tinyllm.models.sampling",
    "tinyllm.features.data.service",
    "tinyllm.features.data.cli",
    "tinyllm.features.training.service",
    "tinyllm.features.training.cli",
    "tinyllm.features.inference.service",
    "tinyllm.features.inference.cli",
    "tinyllm.features.inference.gui",
    "tinyllm.api.server",
    "tinyllm.tools.doctor",
    "tinyllm.tools.env",
    "tinyllm.tools.lab",
    "tinyllm.tools.notebook_sources",
    "tinyllm.tools.setup",
]


@pytest.mark.parametrize("name", LAYERS)
def test_every_module_imports(name):
    assert importlib.import_module(name) is not None


def test_version_is_exported():
    assert tinyllm.__version__


def test_dispatcher_lists_all_commands(capsys):
    from tinyllm.__main__ import main
    assert main([]) == 0
    out = capsys.readouterr().out
    for command in ("env", "setup", "doctor", "data", "train", "generate", "gui",
                    "lab", "serve"):
        assert command in out


def test_dispatcher_routes_the_new_commands():
    """env and lab must be reachable through the single entry point."""
    from tinyllm.__main__ import COMMANDS
    assert COMMANDS["env"][0] == "tinyllm.tools.env"
    assert COMMANDS["lab"][0] == "tinyllm.tools.lab"
    for name, (module_name, _) in COMMANDS.items():
        module = importlib.import_module(module_name)
        assert callable(module.main), f"{name} has no main()"


def test_dispatcher_rejects_unknown_command():
    from tinyllm.__main__ import main
    assert main(["nope"]) == 2


def test_train_cli_parses_flags():
    from tinyllm.features.training.cli import build_parser
    args = build_parser().parse_args(
        ["--preset", "smoke", "--max-iters", "5", "--no-amp", "--batch-size", "2"])
    assert args.preset == "smoke"
    assert args.max_iters == 5
    assert args.use_amp is False
    assert args.batch_size == 2


def test_generate_cli_defaults():
    from tinyllm.features.inference.cli import build_parser
    args = build_parser().parse_args([])
    assert args.temperature == 0.8
    assert args.top_k == 50
    assert args.top_p == 0.95
    assert args.prompt


def test_data_cli_force_does_not_mean_redownload():
    """Regression: --force used to delete the 1.2 GB corpus as well."""
    from tinyllm.features.data.cli import build_parser
    args = build_parser().parse_args(["--force"])
    assert args.force is True
    assert args.redownload is False


def test_every_feature_has_a_service_and_a_view():
    """Feature-based layout: each slice exposes a service and a view."""
    from tinyllm.features.data import service as data_service
    from tinyllm.features.training import service as train_service
    from tinyllm.features.inference import service as inf_service
    assert callable(data_service.prepare)
    assert callable(train_service.train)
    assert callable(inf_service.InferenceEngine)
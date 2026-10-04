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


def test_ci_pipeline_keeps_its_gates():
    """The pipeline's contract: least privilege, bounded jobs, real gates.

    Each of these is a check that was added because something it catches went
    unnoticed: a job with no timeout hangs until the six-hour default, an
    end-to-end run is what proves the program works rather than its parts, and
    a weekly run is how an upstream release announces itself.
    """
    import yaml

    root = Path(__file__).resolve().parent.parent
    text = (root / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    triggers = workflow.get("on", workflow.get(True))

    assert workflow.get("permissions") == {"contents": "read"}
    assert set(triggers) >= {"push", "pull_request", "workflow_dispatch", "schedule"}
    assert workflow["concurrency"]["cancel-in-progress"] is True

    jobs = workflow["jobs"]
    assert {"lint", "test", "package", "smoke"} <= set(jobs)
    for name, job in jobs.items():
        assert job.get("timeout-minutes"), f"job {name} can hang forever"

    assert "ruff check" in yaml.safe_dump(jobs["lint"])
    smoke = yaml.safe_dump(jobs["smoke"])
    for command in ("data prepare", "data tokens", "apexgpt train",
                    "apexgpt generate"):
        assert command in smoke, f"the smoke job never runs {command}"

    # every job that can fail needs a report a stranger can read
    assert "::error" in (root / "tests" / "conftest.py").read_text(encoding="utf-8")


def test_every_relative_link_in_the_documentation_resolves():
    """The README was split into docs/ pages; a stale link is invisible in review.

    A link to a heading that moved, or to a file that was renamed, renders as
    plain text on GitHub and nobody notices until a reader clicks it.
    """
    import re
    import unicodedata
    from urllib.parse import unquote

    root = Path(__file__).resolve().parent.parent
    files = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    assert len(files) > 1, "the reference pages are missing"

    heading = re.compile(r"^(#{1,6}) +(.+?)\s*$")
    fence = re.compile(r"^\s*(```|~~~)")
    link = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

    def slug(text: str) -> str:
        kept = [ch for ch in text
                if ch == " " or ch in "-_"
                or unicodedata.category(ch)[0] in "LNM"
                or unicodedata.category(ch) == "So"]
        return "".join(kept).strip().lower().replace(" ", "-")

    def anchors(path: Path) -> set[str]:
        found, fenced = set(), False
        for line in path.read_text(encoding="utf-8").splitlines():
            if fence.match(line):
                fenced = not fenced
            elif not fenced and (match := heading.match(line)):
                found.add(slug(match.group(2)))
        return found

    broken = []
    for path in files:
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            for target in link.findall(line):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                where, _, fragment = target.partition("#")
                destination = path if not where else (
                    (path.parent / unquote(where)).resolve())
                if not destination.exists():
                    broken.append(f"{path.name}:{number} -> {target}")
                elif fragment and unquote(fragment) not in anchors(destination):
                    broken.append(f"{path.name}:{number} -> {target} "
                                  "(no such heading)")
    assert broken == [], f"documentation links that go nowhere: {broken}"


def test_the_readme_is_the_front_door_and_the_reference_lives_in_docs():
    """A README that grows past a thousand lines stops being read at all."""
    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8").splitlines()
    assert len(readme) <= 1000, f"the README is {len(readme)} lines"

    docs = sorted(p.name for p in (root / "docs").glob("*.md"))
    assert docs, "there are no reference pages"
    text = (root / "README.md").read_text(encoding="utf-8")
    for name in docs:
        assert f"docs/{name}" in text, f"{name} is not linked from the README"


def test_the_release_workflow_publishes_a_tag_and_nothing_else():
    """A release must be one tag push, and must not reach a package index.

    The gates are the point: a tagged commit whose tests never passed is not a
    release, a tag that disagrees with the version produces an artifact nobody
    can install with ``pip install apexgpt==<tag>``, and an accidental upload to
    PyPI cannot be taken back.
    """
    import yaml

    root = Path(__file__).resolve().parent.parent
    path = root / ".github" / "workflows" / "release.yml"
    assert path.exists(), "there is no way to publish a release"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))

    assert triggers["push"]["tags"] == ["v*"]
    assert triggers["workflow_dispatch"]["inputs"]["tag"]["required"] is True
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is False

    jobs = workflow["jobs"]
    assert {"build", "publish"} <= set(jobs)
    for name, job in jobs.items():
        assert job.get("timeout-minutes"), f"job {name} can hang forever"

    # least privilege, per job: the build reads, only the publish writes
    assert jobs["build"]["permissions"] == {"contents": "read", "checks": "read"}
    assert jobs["publish"]["permissions"] == {"contents": "write"}
    assert jobs["publish"]["needs"] == "build"

    build = yaml.safe_dump(jobs["build"])
    for gate in ("python -m build", "twine check", "pyproject.toml",
                 "check-runs", "upload-artifact"):
        assert gate in build, f"the build job never does {gate}"

    publish = yaml.safe_dump(jobs["publish"])
    assert "download-artifact" in publish
    assert "gh release create" in publish
    for forbidden in ("pypi", "twine upload", "packages: write"):
        assert forbidden not in publish.lower(), f"a release must not touch {forbidden}"


def test_the_release_workflow_checks_the_wheel_the_pipeline_checks():
    """The artifact people install gets the same cleanliness check, not a weaker one."""
    root = Path(__file__).resolve().parent.parent
    workflows = root / ".github" / "workflows"
    package = (workflows / "tests.yml").read_text(encoding="utf-8")
    release = (workflows / "release.yml").read_text(encoding="utf-8")

    leaked = 'if n.split("/")[0] in {"tests", "notebooks", "data", "models", ".github"}'
    assert leaked in package
    assert leaked in release


def test_no_source_file_is_hidden_by_gitignore():
    """Regression: `.gitignore` had `models/` and `data/`, which also match
    ``apexgpt/models/`` and ``apexgpt/features/data/`` - so the model layer and
    the whole corpus pipeline were missing from every GitHub clone while the
    local suite passed. CI caught it; this stops it recurring.
    """
    import shutil
    import subprocess

    root = Path(__file__).resolve().parent.parent
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("not a git checkout")

    def git(*args) -> str:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True)
        assert done.returncode == 0, done.stderr
        return done.stdout

    source_suffixes = (".py", ".ipynb", ".toml", ".yml", ".json")
    ignored = [line for line in
               git("ls-files", "--others", "--ignored", "--exclude-standard",
                   "--", "apexgpt", "tests", "notebooks", ".github").splitlines()
               if line.endswith(source_suffixes)]
    assert ignored == [], f"git is ignoring source files: {ignored}"

    tracked = set(git("ls-files").split())
    missing = [path for path in tracked if not (root / path).exists()]
    assert missing == [], f"tracked but absent from disk: {missing}"

    # every package the wheel ships must actually be in the repository
    tomllib = pytest.importorskip("tomllib")
    with (root / "pyproject.toml").open("rb") as fh:
        packages = tomllib.load(fh)["tool"]["setuptools"]["packages"]
    untracked = [p for p in packages
                 if f"{p.replace('.', '/')}/__init__.py" not in tracked]
    assert untracked == [], f"package slices missing from git: {untracked}"


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
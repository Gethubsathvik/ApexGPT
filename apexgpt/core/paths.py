"""Filesystem layout for ApexGPT.

Bulk corpus data is large, so the data root is overridable through the
``APEXGPT_DATA_DIR`` environment variable. Model artifacts always live next to
the source tree so a run is easy to find.
"""
from __future__ import annotations

import os
from pathlib import Path


def _project_root(package_file: Path | str | None = None) -> Path:
    """Where artifacts belong.

    Cloned or pip-installed with ``-e``, that is the repository. Copied into an
    interpreter's ``site-packages`` there is no project at all, and writing
    checkpoints next to ``site-packages`` is never what anyone wants - so an
    installed copy uses the working directory instead.

    ``package_file`` is the module's own path; it is a parameter only so the
    rule can be tested without installing anything.
    """
    here = Path(package_file or __file__).resolve().parent.parent.parent
    # pyproject.toml and requirements.txt only exist together in a checkout;
    # neither is ever installed into site-packages
    if any((here / marker).exists() for marker in ("pyproject.toml",
                                                    "requirements.txt")):
        return here
    return Path.cwd()


ROOT = _project_root()

DATA_ROOT = Path(os.environ.get("APEXGPT_DATA_DIR") or (ROOT / "data")).resolve()

DATA_DIR = DATA_ROOT
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RUNS_DIR = MODELS_DIR / "runs"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
HUB_DIR = DATA_DIR / "hub"
HUB_MODEL_DIR = HUB_DIR / "models"
HUB_DATASET_DIR = HUB_DIR / "datasets"

# Only the data and model roots are created eagerly. Creating a scripts/ folder
# as an import side effect is meaningless and used to be done.
for _d in (DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, RUNS_DIR, TOKENIZER_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    return (
        f"project root : {ROOT}\n"
        f"data root    : {DATA_DIR}\n"
        f"models dir   : {MODELS_DIR}"
    )

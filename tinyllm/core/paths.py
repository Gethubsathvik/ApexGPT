"""Filesystem layout for TinyLLM.

Bulk corpus data is large, so the data root is overridable through the
``TINYLLM_DATA_DIR`` environment variable. Model artifacts always live next to
the source tree so a run is easy to find.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DATA_ROOT = Path(os.environ.get("TINYLLM_DATA_DIR") or (ROOT / "data")).resolve()

DATA_DIR = DATA_ROOT
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RUNS_DIR = MODELS_DIR / "runs"
TOKENIZER_DIR = DATA_DIR / "tokenizer"

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

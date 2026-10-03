"""The notebook sources, so ``notebooks/*.ipynb`` are generated, not hand-edited JSON.

A ``.ipynb`` file is a JSON file with escaped newlines, which is unpleasant to
review and easy to corrupt. The cell sources live here as ordinary Python
strings instead, and the notebooks on disk are their output::

    python -m tinyllm.tools.notebook_sources      # regenerate notebooks/

``tests/test_lab.py`` asserts the two stay in step, so editing a notebook in
Jupyter and forgetting this file is caught rather than silently overwritten.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..core.paths import ROOT
from .lab import NOTEBOOK_DIR, new_notebook

# Shared by every notebook: makes `import tinyllm` work from a clone.
IMPORT_CELL = '''import sys
from pathlib import Path

# Works whether TinyLLM is pip-installed or merely cloned next to this notebook.
def _repo_root(start):
    for candidate in [start, *start.parents]:
        if (candidate / "tinyllm" / "__init__.py").exists():
            return candidate
    return start

ROOT = _repo_root(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import tinyllm
print("tinyllm", tinyllm.__version__, "from", Path(tinyllm.__file__).parent)'''

ENVIRONMENT: list[tuple[str, str]] = [
    ("markdown", """
# 00 - Environment

This notebook is the Jupyter front end of the same scan the terminal runs:

```bash
python -m tinyllm env
```

It reports the machine spec, every requirement from `requirements*.txt`, and
the settings TinyLLM resolves for this host - then applies them, so the
training and inference cells below use the same device, thread count and
precision policy as the command line.
"""),
    ("code", IMPORT_CELL),
    ("code", '''
from tinyllm.core.environment import bootstrap

report, settings = bootstrap()
print(report.to_text())
'''),
    ("markdown", "## Machine spec as a table"),
    ("code", "print(report.to_markdown())"),
    ("markdown", "## Resolved settings"),
    ("code", '''
print("device       :", settings.device)
print("backend      :", settings.backend)
print("threads      :", settings.threads)
print("amp          :", settings.amp)
print("checkpointing:", settings.checkpointing)
print("preset       :", settings.preset, "->", report.accelerator.precision)
'''),
    ("markdown", """
## Re-scan for a specific backend

Detection is automatic, but you can ask what the settings *would* be for
another device. Asking for a backend that is absent is an error, never a
silent fallback to the CPU.
"""),
    ("code", '''
from tinyllm.core.environment import scan

for candidate in ("cpu", "cuda", "mps", "xpu", "dml"):
    try:
        print(f"{candidate:>5}: {scan(device=candidate).settings.device}")
    except RuntimeError as exc:
        print(f"{candidate:>5}: unavailable ({str(exc)[:60]}...)")
'''),
    ("markdown", """
## Requirements

The scan reads the requirements files, so it cannot drift from what
`python -m tinyllm setup --install` actually installs.
"""),
    ("code", '''
for req in report.requirements:
    print(f"{req.status:>8}  {req.name:<20} {req.specifier}")

print()
print("missing:", [r.name for r in report.missing] or "none")
'''),
    ("markdown", """
## Notes from this host

On a machine without an NVIDIA GPU the scan says so explicitly instead of
pretending CUDA exists.
"""),
    ("code", '''
for note in report.notes:
    print("-", note)
'''),
]

TRAIN: list[tuple[str, str]] = [
    ("markdown", """
# 01 - Training

Training is the same call the CLI makes: `tinyllm.features.training.service.train`.
Nothing here is notebook-only logic, so a notebook run and
`python -m tinyllm train` produce the same checkpoint.
"""),
    ("code", IMPORT_CELL),
    ("code", '''
from tinyllm.core.config import Config
from tinyllm.core.paths import RUNS_DIR

cfg = Config()
cfg.data.select_dataset("shakespeare")     # 338k tokens: trains in minutes on a CPU
print("corpus        :", cfg.data.dataset)
print("corpus ready  :", cfg.data.is_ready())
print("token binaries:", cfg.data.binary_dir)
print("run directory :", RUNS_DIR)
'''),
    ("markdown", """
## The corpus

Training needs tokenised text. tiny Shakespeare is the fast one - 1.1 MB and
338,025 tokens - so it is the default here. Wikipedia is 1.3 GB and takes about
15 minutes to prepare. Either way the preparation is the same code:

```bash
python -m tinyllm data prepare --source shakespeare
python -m tinyllm data sources             # what else is available
```
"""),
    ("code", '''
# from tinyllm.features.data.service import prepare
# prepare(cfg.data, source="shakespeare")    # ~4 s: download, tokenize, split 80/20
'''),
    ("markdown", """
## Train

`preset="auto"` (the default) sizes the run to this machine. On a CPU-only
laptop that is `cpu-tiny`; the scan in notebook 00 already told you which.
"""),
    ("code", '''
from tinyllm.features.training.service import train

result = train(cfg, preset="auto", max_iters=100, log_interval=10,
               run_name="gpt-shakespeare")
'''),
    ("markdown", "## Loss history"),
    ("code", '''
for row in (result.history or [])[-10:]:
    print(f"step {row['step']:>5}  train {row['train_loss']:.4f}  "
          f"val {row['val_loss']:.4f}  lr {row['lr']:.2e}")

print("best val:", result.best_val_loss)
print("checkpoint:", result.checkpoint)
'''),
    ("markdown", """
## Loss curves

10.82 is `ln(50257)` - the loss of random guessing. Watch it fall.
"""),
    ("code", '''
from IPython.display import Image, display

display(Image(filename=str(result.checkpoint.parent / "loss_curves.png")))
'''),
]

INFERENCE: list[tuple[str, str]] = [
    ("markdown", """
# 02 - Inference

`InferenceEngine` is shared by the CLI, the Tk GUI and the HTTP API, so the
text you get here is the text they produce. The KV cache and the sampling
parameters are identical across all four front ends.
"""),
    ("code", IMPORT_CELL),
    ("code", '''
from tinyllm.features.inference.service import GenerationRequest, InferenceEngine
from tinyllm.core.paths import RUNS_DIR

candidates = sorted(RUNS_DIR.glob("*/checkpoint.pt"))
print("checkpoints found:")
for path in candidates:
    print("  ", path)
'''),
    ("code", '''
CHECKPOINT = None                     # set to a path above to override
CHECKPOINT = CHECKPOINT or "models/runs/gpt-shakespeare/checkpoint.pt"

engine = InferenceEngine(device="auto")
if Path(CHECKPOINT).exists():
    engine.load(CHECKPOINT)
elif candidates:
    engine.load(candidates[-1])
else:
    print("no checkpoint yet - run 01_train.ipynb, or: python -m tinyllm train")
'''),
    ("markdown", "## One-shot generation"),
    ("code", '''
request = GenerationRequest(
    prompt="ROMEO:",
    max_new_tokens=120,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
print(request.prompt, end="")

for piece in engine.stream(request):
    print(piece, end="", flush=True)
print()
'''),
    ("markdown", """
## Streaming with and without the KV cache

Decoding with the cache appends one token per step. Turning it off recomputes
the whole prefix every step: slower, but a useful correctness check - both
paths must produce the same text.
"""),
    ("code", '''
import time

for use_cache in (True, False):
    start = time.time()
    text = "".join(engine.stream(GenerationRequest(
        prompt="ROMEO:", max_new_tokens=40, use_cache=use_cache)))
    print(f"use_cache={use_cache!s:<5} {time.time() - start:6.2f}s  {text[:70]}...")
'''),
    ("markdown", "## Sampling parameters"),
    ("code", '''
for temperature in (0.1, 0.7, 1.2):
    text = "".join(engine.stream(GenerationRequest(
        prompt="KING RICHARD II:", max_new_tokens=30, temperature=temperature)))
    print(f"T={temperature}: {text}")
'''),
]

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "00_environment.ipynb": ENVIRONMENT,
    "01_train.ipynb": TRAIN,
    "02_inference.ipynb": INFERENCE,
}


def build_all() -> dict[str, dict]:
    """Every notebook document, keyed by file name."""
    return {name: new_notebook(cells) for name, cells in NOTEBOOKS.items()}


def write_all(directory: Path = NOTEBOOK_DIR) -> list[Path]:
    """Write the notebooks to disk, creating the directory if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, document in build_all().items():
        path = directory / name
        path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    directory = Path(argv[0]) if argv else NOTEBOOK_DIR
    for path in write_all(directory):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

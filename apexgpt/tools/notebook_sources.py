"""The notebook sources, so ``notebooks/*.ipynb`` are generated, not hand-edited JSON.

A ``.ipynb`` file is a JSON file with escaped newlines, which is unpleasant to
review and easy to corrupt. The cell sources live here as ordinary Python
strings instead, and the notebooks on disk are their output::

    python -m apexgpt.tools.notebook_sources      # regenerate notebooks/

``tests/test_lab.py`` asserts the two stay in step, so editing a notebook in
Jupyter and forgetting this file is caught rather than silently overwritten.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..core.paths import ROOT
from .lab import NOTEBOOK_DIR, new_notebook

# Shared by every notebook: makes `import apexgpt` work from a clone.
IMPORT_CELL = '''import sys
from pathlib import Path

# Works whether ApexGPT is pip-installed or merely cloned next to this notebook.
def _repo_root(start):
    for candidate in [start, *start.parents]:
        if (candidate / "apexgpt" / "__init__.py").exists():
            return candidate
    return start

ROOT = _repo_root(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import apexgpt
print("apexgpt", apexgpt.__version__, "from", Path(apexgpt.__file__).parent)'''

ENVIRONMENT: list[tuple[str, str]] = [
    ("markdown", """
# 00 - Environment

This notebook is the Jupyter front end of the same scan the terminal runs:

```bash
python -m apexgpt env
```

It reports the machine spec, every requirement from `requirements*.txt`, and
the settings ApexGPT resolves for this host - then applies them, so the
training and inference cells below use the same device, thread count and
precision policy as the command line.
"""),
    ("code", IMPORT_CELL),
    ("code", '''
from apexgpt.core.environment import bootstrap

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
from apexgpt.core.environment import scan

for candidate in ("cpu", "cuda", "mps", "xpu", "dml"):
    try:
        print(f"{candidate:>5}: {scan(device=candidate).settings.device}")
    except RuntimeError as exc:
        print(f"{candidate:>5}: unavailable ({str(exc)[:60]}...)")
'''),
    ("markdown", """
## Requirements

The scan reads the requirements files, so it cannot drift from what
`python -m apexgpt setup --install` actually installs.
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

Training is the same call the CLI makes: `apexgpt.features.training.service.train`.
Nothing here is notebook-only logic, so a notebook run and
`python -m apexgpt train` produce the same checkpoint.
"""),
    ("code", IMPORT_CELL),
    ("code", '''
from apexgpt.core.config import Config
from apexgpt.core.paths import RUNS_DIR

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
python -m apexgpt data prepare --source shakespeare
python -m apexgpt data sources             # what else is available
```
"""),
    ("code", '''
# from apexgpt.features.data.service import prepare
# prepare(cfg.data, source="shakespeare")    # ~4 s: download, tokenize, split 80/20
'''),
    ("markdown", """
## Train

`preset="auto"` (the default) sizes the run to this machine. On a CPU-only
laptop that is `cpu-tiny`; the scan in notebook 00 already told you which.
"""),
    ("code", '''
from apexgpt.features.training.service import train

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
from apexgpt.features.inference.service import GenerationRequest, InferenceEngine
from apexgpt.core.paths import RUNS_DIR

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
    print("no checkpoint yet - run 01_train.ipynb, or: python -m apexgpt train")
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
    ("markdown", """
## What the model expects next

Sampling draws one token and throws the rest of the distribution away.
`predict_next` keeps it: the ranking the loss function was actually minimising,
plus the entropy of the whole distribution. `ln(vocab_size)` nats would be a
uniform model that has learned nothing.
"""),
    ("code", '''
import math

for prompt in ("ROMEO:", "KING RICHARD II:"):
    rows, entropy = engine.predict_next_with_entropy(prompt, top_k=5)
    print(f"\\n{prompt!r}")
    for row in rows:
        print(f"  {row.rank}. p={row.probability:.4f} logp={row.logprob:>7.3f}  {row.label}")
    print(f"  entropy {entropy:.3f} nats of a possible {math.log(engine.metadata()['vocab_size']):.3f}")
'''),
    ("markdown", """
The same thing is one CLI flag, on any checkpoint:

```
python -m apexgpt generate --prompt "KING RICHARD II:" --predict 5
```
"""),
    ("markdown", """
## Hugging Face and Kaggle

The same next-token question can be asked of a pretrained model instead of one
trained here, which is the honest way to compare a 30M-parameter run against a
real LLM. It needs the optional Hub extras:

```
python -m apexgpt setup --hub --install
python -m apexgpt hub check
python -m apexgpt hub model gpt2 --predict "KING RICHARD II:" --top-k 5
```
"""),
    ("code", '''
from apexgpt.features.hub import check as hub_check

for label, available, detail in hub_check():
    print(f"  [{'ok' if available else '--'}] {label:<42} {detail}")
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

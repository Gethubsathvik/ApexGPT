# 🧠 ApexGPT

> A GPT-style transformer language model built from scratch in PyTorch — dataset pipeline, training loop, CLI, desktop GUI, Jupyter Lab, and an optional HTTP inference service.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-375%20tests-success.svg)](tests)

## 📁 Project layout

```
ApexGPT/
├── apexgpt/
│   ├── __main__.py            # single command dispatcher
│   ├── core/                  # config, paths, device, environment, seeding, system, text
│   ├── models/                # gpt.py, builder.py, sampling.py: M
│   ├── features/
│   │   ├── data/              # service.py, cli.py, sources.py, tokenizers.py
│   │   ├── training/          # service.py, cli.py
│   │   ├── inference/         # service.py, cli.py, gui.py
│   │   └── hub/               # service.py, cli.py: Hugging Face / Kaggle
│   ├── api/server.py           # optional FastAPI service
│   └── tools/                  # setup.py, env.py, doctor.py, lab.py, notebook_sources.py
├── notebooks/                  # generated .ipynb: environment, train, inference
├── docs/                       # reference material split out of this README
├── tests/                      # one file per feature, plus conftest.py
├── models/runs/<name>/         # checkpoint.pt, history.json, loss_curves.png
├── data/raw/<corpus>/          # cached corpus text
├── data/binary/<corpus>/       # train.bin, val.bin (uint16)
├── data/hub/                   # Hugging Face / Kaggle downloads
├── apexgpt.settings.json       # persisted overrides (written by --save-settings)
├── pyproject.toml              # packaging: pip install -e . -> the apexgpt command
├── .github/workflows/tests.yml   # CI: pytest on Linux/Windows, wheel build
├── .github/workflows/release.yml # a v* tag becomes a GitHub Release
├── requirements.txt
├── requirements-api.txt
├── requirements-hub.txt
├── requirements-notebook.txt
└── README.md
```

---

## 📑 Contents

- [✨ What it does](#%E2%9C%A8-what-it-does)
- [🖥️ Reference configuration](#%F0%9F%96%A5%EF%B8%8F-reference-configuration)
- [🔍 Environment scan](#%F0%9F%94%8D-environment-scan)
- [🚀 Quick start](#%F0%9F%9A%80-quick-start)
- [📓 Jupyter Lab](#%F0%9F%93%93-jupyter-lab)
- [📚 Corpora](#%F0%9F%93%9A-corpora)
- [🎭 Tiny Shakespeare in 5 minutes](#%F0%9F%8E%AD-tiny-shakespeare-in-5-minutes)
- [🏗️ Architecture (MVC + service + feature-based)](#%F0%9F%8F%97%EF%B8%8F-architecture-mvc--service--feature-based)
- [🤖 Model](#%F0%9F%A4%96-model)
- [🔤 Tokenizers](#%F0%9F%94%A4-tokenizers)
- [🎓 Training](#%F0%9F%8E%93-training)
- [💬 Inference](#%F0%9F%92%AC-inference)
- [🖥️ GUI](#%F0%9F%96%A5%EF%B8%8F-gui)
- [🧪 Tests](#%F0%9F%A7%AA-tests)
- [📁 Project layout](#%F0%9F%93%81-project-layout)

Reference material, in `docs/`:

- [📐 The mathematics behind ApexGPT](docs/mathematics.md) - Every formula the code implements, and why each one is shaped the way it is.
- [💻 Hardware portability](docs/portability.md) - Every backend ApexGPT probes for, what it costs, and what happens when one is absent.
- [🪟 Platform setup](docs/platform-setup.md) - Per-platform dependencies, and the installs that need more than `pip`.
- [🌐 HTTP API](docs/http-api.md) - The inference service: one command, every endpoint, and how to reach it from another machine.
- [🌐 Hugging Face & Kaggle](docs/hub.md) - Pretrained models, hub datasets, and the reference training scripts.
- [🤖 Android](docs/android.md) - Driving the inference API from a phone.
- [📏 Measured](docs/measurements.md) - Runs of this repository, recorded: the tokenizer comparison and two completed trainings.
- [🐛 Bugs found and fixed](docs/bugs-found-and-fixed.md) - Defects this project found in itself, and what each one taught.

---

## ✨ What it does

- **Scans the machine once** — CPU, cores, RAM, adapters, every PyTorch backend, and every requirement — then resolves and applies the settings ApexGPT will run with
- Runs on **CPU, NVIDIA CUDA, AMD ROCm, Intel XPU, Apple Metal, and Windows DirectML** — detected automatically, never silently
- Trains a **GPT from scratch** — no `transformers` model classes, every layer hand-written
- **123.8M parameters** at the requested 12 layers / 768 hidden / 12 heads
- Corpora: **tiny Shakespeare**, **Wikipedia**, **WikiText-2**, **TinyStories**, **OpenWebText**, any **Hugging Face** dataset, any **Kaggle** dataset, or a local file
- Next-token prediction with AdamW, warmup + cosine decay, and gradient clipping
- Samples with **temperature, top-k, top-p**, and repetition penalty
- Four front ends over one implementation: **CLI**, **Tkinter GUI**, **Jupyter Lab**, **HTTP service**
- **Automated tests** — 375 of them, covering causality, the KV cache, sampling, portability, corpus fetching, the live system scan, the next-token distribution, notebooks and the GUI

---

## 🖥️ Reference configuration

The published timings in this README come from a deliberately modest reference
box: **a 4-core / 8-thread CPU, no discrete GPU, about 8 GB of RAM**, Python
3.10+ with the CPU PyTorch wheel. No claim is made about faster hardware —
`python -m apexgpt env` prints the real numbers for whatever machine it runs on,
and `--preset auto` sizes a run to what it finds:

```
Machine spec
  CPU            : <your CPU>
  cores          : 4 cores / 8 threads
  GPU            : <your adapter, or none>
  CUDA           : not available - no NVIDIA GPU or driver on this machine
  usable backend : cpu (x86_64)
  memory         : 7.9 GB RAM
  torch          : 2.14.1+cpu
  python         : 3.13.5 (CPython)
```

> ### ⚠️ What that reference box cannot do
>
> **1. No CUDA.** PyTorch CUDA kernels need an NVIDIA GPU and driver. On a
> machine without one, `torch.cuda.is_available()` is `False` and training runs
> on CPU. **Every entry point still auto-selects the best available backend**, so
> the same code uses an NVIDIA GPU, an Intel GPU or Apple Metal with no edits —
> see [Hardware portability](docs/portability.md#%F0%9F%92%BB-hardware-portability).
>
> **2. All logical threads, 4 physical cores.** The thread count is what the
> presets and the live scan size themselves around.
>
> **3. AMP and gradient checkpointing are switched off automatically.** They are
> normally big wins on a GPU, but on a CPU **without native bf16** they are a
> **66x penalty**:
>
> | Setting | Time/iter | Throughput |
> |---------|-----------|-----------|
> | bf16 autocast + gradient checkpointing | 195.1 s | 4 tok/s |
> | plain fp32, no checkpointing | **2.95 s** | **260 tok/s** |
>
> A CPU without native bf16 emulates it, and checkpointing's recompute blocks
> the fused attention path. This is detected, not hard-coded: CPUs with native
> bf16 keep AMP on. Force either back on with `--amp` / `--checkpointing`.

---

> **Run it on localhost, one line, no arguments:**
>
> ```bash
> python -m apexgpt serve
> ```
>
> Binds **http://127.0.0.1:8000**, finds the most recent real checkpoint by
> itself, loads the tokenizer that checkpoint recorded, and serves the GUI-free
> API: `/health`, `/generate`, `/predict`, `/stream`, `/v1/completions`,
> `/v1/models`, a landing page at `/`, and interactive docs at `/docs`. Spell
> the defaults out with
> `python -m apexgpt serve --host 127.0.0.1 --port 8000`; bind `0.0.0.0`
> instead to let other machines on your network reach it. Full detail in
> [🌐 HTTP API](docs/http-api.md#%F0%9F%8C%90-http-api).
---

## 💻 Hardware portability

Every backend ApexGPT probes, and what it does when one is missing: [docs/portability.md](docs/portability.md).

---

## 🔍 Environment scan

One command answers "can this machine run ApexGPT, and with what settings?" — the
same scan backs the CLI, the GUI, the API and the Jupyter kernel.

```bash
python -m apexgpt env                 # scan, report, and apply in this process
python -m apexgpt env --json          # machine-readable, for scripts and CI
python -m apexgpt env --check         # exit 1 if a required package is missing
python -m apexgpt env --export        # shell lines that carry the settings out
python -m apexgpt env --device cuda   # resolve settings for a specific backend
python -m apexgpt env --save env.json # keep the report next to your run
```

It prints the machine spec, every requirement parsed from `requirements*.txt`
with its installed version and whether that version satisfies the pin, the
resolved settings, and the exact next command to run.

The same scan is a library, so scripts and notebooks resolve settings instead of
guessing:

```python
from apexgpt.core.environment import bootstrap

report, settings = bootstrap()   # scan, then apply (threads + env vars)
print(report.to_text())         # or report.to_markdown() / report.to_json()
print(settings.device, settings.threads, settings.amp, settings.preset)
```

`apply_settings()` exports `APEXGPT_DEVICE`, `APEXGPT_THREADS`, `APEXGPT_PRESET`,
`APEXGPT_AMP`, `APEXGPT_CHECKPOINTING` and `OMP_NUM_THREADS`, so a notebook
kernel, a `serve` process and a second terminal all inherit the same
configuration instead of re-deriving it. `APEXGPT_DATASET` selects the corpus.

### 🎚️ Auto-tuning to the machine's current load

The scan is not a one-off. Every run re-measures the box, and the scan reads the
process table as well as the hardware, so a machine that is busy *right now* gets
fewer threads and a smaller batch than an idle one:

```bash
python -m apexgpt env                        # live scan, with the busiest processes
python -m apexgpt env --no-load-scan         # skip the measurement entirely
python -m apexgpt env --max-cpu-percent 70   # hand back a thread earlier
python -m apexgpt env --reserve-ram-gb 3.0   # keep more memory free
```

Two rules, both conservative and both reported in the scan:

| Condition | Effect | Default |
| --- | --- | --- |
| CPU utilisation at or above the threshold | hand back one thread | `85%` |
| free RAM below the reserve | halve the batch size | `1.5 GB` |

```text
[load]  cpu 12.4%  ram 1.0/7.8 GB free  processes 285  disk 84.1/465.6 GB
        busy : python.exe 41.2%  chrome.exe 33.8%  Code.exe 22.1%
        ram  : chrome.exe 2418.2 MB  Code.exe 1502.7 MB  Teams.exe 884.1 MB
[load]  cpu below 85.0% and 1.0 GB free >= reserve 1.5 GB: keeping measured settings
```

Anything you set yourself is never touched by the load rules — an explicit
override always wins, and a machine under pressure is *your* call, not the
scan's.

### 🧷 Persisted overrides

Flags can be saved to `apexgpt.settings.json` so the next run starts from them:

```bash
python -m apexgpt env --threads 6 --batch-size 16 --save-settings
python -m apexgpt env --reset-settings       # back to measured defaults
python -m apexgpt env                        # shows each value and where it came from
```

The file starts as an all-null template, so only what you deliberately set is
ever pinned:

```json
{
  "threads": 6,
  "batch_size": 16,
  "tokenizer": "char",
  "reserve_ram_gb": null,
  "max_cpu_percent": null
}
```

Precedence, highest first: **command-line flag → `apexgpt.settings.json` or
`APEXGPT_SET_*` → live measurement.** The same values can be set through
`APEXGPT_SET_THREADS=6` and friends for CI. The scan labels every resolved value
with its origin:

```text
[settings] threads 6  (flag:--threads)  batch 16  (file)  tokenizer char  (env:APEXGPT_SET_TOKENIZER)
            amp off  (auto)  reserve 1.5 GB  (auto)  max cpu 85.0%  (auto)
```

Measured readings never persist themselves: they are re-taken on every run, so a
value written while the machine was busy cannot freeze a stale measurement into
the config.

---

## 🚀 Quick start

```bash
git clone https://github.com/Gethubsathvik/ApexGPT.git
cd ApexGPT

python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

python -m apexgpt setup --install            # installs torch for YOUR hardware
python -m apexgpt env                        # scan this machine, apply its settings
python -m apexgpt doctor                     # verify the environment

python -m apexgpt data prepare --source shakespeare    # ~1 MB, ~4 s
python -m apexgpt train --dataset shakespeare --preset cpu-tiny
python -m apexgpt generate --prompt "ROMEO:" --max-new-tokens 200
python -m apexgpt generate --prompt "ROMEO:" --predict 8   # what it expects next

python -m apexgpt lab --install              # Jupyter Lab, kernel preconfigured
python -m apexgpt setup --hub --install       # optional: Hugging Face + Kaggle
python -m apexgpt hub check                   # what the Hub can do here
```

The full-size route on a machine with room for it:

```bash
python -m apexgpt data prepare              # Wikipedia, ~15 min, ~1.3 GB
python -m apexgpt train --preset auto       # train, sized to this machine
python -m apexgpt gui                       # desktop GUI
```

Prefer a released build to the clone? Every tag has one, built and checked by
CI, and the wheel is pure Python — `torch` still comes from the platform
install in `setup --install`:

```bash
pip install https://github.com/Gethubsathvik/ApexGPT/releases/download/v1.2.0/apexgpt-1.2.0-py3-none-any.whl
python -m apexgpt setup --install
python -m apexgpt doctor
```

Releases are cut by the [`release`](.github/workflows/release.yml) workflow, not
by hand: push a `v*` tag whose version matches `pyproject.toml` and CI checks
that the tagged commit's test run is green, builds the wheel and the sdist, runs
`twine check`, refuses to ship a wheel carrying the tests or the corpus, and
attaches both files to the release page. Nothing is uploaded to PyPI.

All commands share one entry point:

```bash
python -m apexgpt            # list every command
python -m apexgpt train --help
```

Installing the package instead of running it from the clone gives you an
`apexgpt` command as well, and keeps the checkpoints and corpora next to you
rather than inside `site-packages`:

```bash
pip install -e .             # or: pip install .
apexgpt env
apexgpt train --preset smoke
```

---

## 🪟 Windows

Dependencies and installs: [docs/platform-setup.md](docs/platform-setup.md).

---

## 🐧 Linux

Dependencies and installs: [docs/platform-setup.md](docs/platform-setup.md).

---

## 🍎 macOS

Dependencies and installs: [docs/platform-setup.md](docs/platform-setup.md).

---

## 📓 Jupyter Lab

Jupyter Lab is a **fourth front end over the same services** — there is no
notebook-only model logic. `python -m apexgpt lab` installs the notebook
requirements, registers a kernel that starts with the device, thread count and
preset the scan resolved, validates the shipped notebooks, and starts the server.

```bash
python -m apexgpt lab --install     # jupyterlab + ipykernel (requirements-notebook.txt)
python -m apexgpt lab               # register the kernel and open Lab
python -m apexgpt lab --register-only   # write the kernel spec, don't launch
python -m apexgpt lab --check           # exit 1 if Lab isn't usable
python -m apexgpt lab --list            # notebooks, kernel path, kernel env
python -m apexgpt lab --no-browser --port 8890 --ip 0.0.0.0
```

The registered kernel spec carries the resolved environment:

```json
{
  "argv": [".../python.exe", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "ApexGPT (scanned environment)",
  "name": "apexgpt",
  "env": {"APEXGPT_DEVICE": "cpu", "OMP_NUM_THREADS": "8", "APEXGPT_PRESET": "cpu-tiny", ...}
}
```

### Notebooks

| Notebook | What it does |
|----------|--------------|
| `00_environment.ipynb` | The scan itself: machine spec, requirements, resolved settings, what each backend would do |
| `01_train.ipynb` | `Config` + `train()` — the same call the CLI makes — then the loss table and curves |
| `02_inference.ipynb` | `InferenceEngine`: one-shot generation, KV-cache comparison, temperature sweep |

They are generated from `apexgpt/tools/notebook_sources.py` so the cells stay
readable Python instead of escaped JSON:

```bash
python -m apexgpt.tools.notebook_sources      # regenerate notebooks/
```

`python -m apexgpt lab` refuses to be quiet about damage: it validates every
notebook (JSON, nbformat 4, kernel name, every code cell parses, no stored
outputs) and reports problems before launching.

Notebook deps live in [`requirements-notebook.txt`](requirements-notebook.txt);
nothing else in the project imports them.

---

## 📚 Corpora

`python -m apexgpt data sources` lists everything selectable. Every source ends
as one plain UTF-8 text file, so the tokenizer, the `uint16` binaries and the
training loop are identical no matter where the text came from.

| `--source` | Origin | Size | Tokens | Notes |
|------------|--------|------|--------|-------|
| `shakespeare` | Project Gutenberg (public domain) | 1.1 MB | **338,025** | Fastest end-to-end demo here |
| `wikipedia` *(default)* | HF `wikimedia/wikipedia` `20231101.en` | 772 MB | ~193M | Parquet shards streamed to disk |
| `wikitext` | HF `Salesforce/wikitext` `wikitext-2-raw-v1` | 12 MB | ~3M | Wikipedia's benchmark set |
| `tinystories` | HF `roneneldan/TinyStories` | 2 GB | ~500M | Synthetic stories; small models learn grammar fast |
| `openwebtext` | HF `Skylion007/openwebtext` | 12 GB | ~3B | Streamed and cut at `--target-mb` |
| `hf:<repo_id>` | any Hugging Face dataset | — | — | Streamed, cut at `--target-mb` |
| `kaggle:<slug>` | any Kaggle dataset | — | — | Needs `kaggle.json`; `kagglehub` or the `kaggle` CLI |
| `local:<path>` | a file on disk | — | — | `.txt`, `.md` are read where they lie; `.csv`, `.json`, `.jsonl`, `.parquet` are converted into `data/raw/local/<source>.txt` |
| `url:<link>` | plain text over HTTP | — | — | Any text URL |

```bash
python -m apexgpt data sources                              # what is available
python -m apexgpt data prepare --source shakespeare         # tiny Shakespeare
python -m apexgpt data prepare --source wikitext --target-mb 50
python -m apexgpt data prepare --source tinystories --target-mb 500
python -m apexgpt data prepare --source hf:roneneldan/TinyStories
python -m apexgpt data prepare --source kaggle:user/dataset-slug
python -m apexgpt data prepare --source local:my_notes.txt
python -m apexgpt data tokens --tokenizer char --top 20   # every id, with its value

python -m apexgpt train --dataset shakespeare              # train on it
```

Each corpus gets its own directory (`data/raw/<key>/`, `data/binary/<key>/`), so
several can be built side by side. `APEXGPT_DATASET=shakespeare` sets the default
for every entry point.

**Hugging Face datasets are streamed, not downloaded.** `load_dataset(...,
streaming=True)` is cut off at `--target-mb`, so a 12 GB corpus is usable on a
laptop with 8 GB of RAM, and the raw text is cached so a second run tokenizes
offline.

**Tokenization** is GPT-2 byte-level BPE by default (vocab **50257**,
round-trip verified exact), written as flat `uint16` streams and split 80/20.
`--tokenizer char` switches to the 257-id byte-level vocabulary instead — see
[🔤 Tokenizers](#%F0%9F%94%A4-tokenizers):

```
[tokenize] 338,025 tokens (0.34M), dtype=uint16
[split]    80/20 -> train 270,420 tok | val 67,605 tok
```

Useful flags: `--tokenizer`, `--target-mb`, `--num-shards`, `--train-split`,
`--keep-parquet`, `--force` (re-tokenize from the cached text), `--redownload`
(refetch), and `--no-report`.

### 🔢 Every token, with its id

`python -m apexgpt data tokens` is the corpus as the model actually receives it:
one numbered row per id, with the text it stands for, how often it occurs, and
what share of the corpus that is. Counts of ids — they answer *what is in the
vocabulary*; the successor ranking answers *what comes next*.

```bash
python -m apexgpt data tokens --tokenizer char --top 10
python -m apexgpt data tokens --source local:my_notes.txt --limit 40
```

> A one-off `--source` names a file to inspect, not a corpus to build: a
> `.txt` or `.md` file is read where it lies, so two different files with the
> same name can never answer with each other's tokens. Formats that need
> converting are cached under a name derived from the whole path, for the same
> reason. Build it for training only if you mean to: the command prints the
> `data prepare` line that does that.

```
======================================================================
TOKEN TABLE - shakespeare.txt
======================================================================
tokenizer      : char
vocabulary     : 257 ids (eos = 256)
corpus         : 1,115,394 characters
tokens         : 1,115,394
distinct ids   : 65 (25.3% of the vocabulary)
compression    : 1.00 tokens per character
======================================================================
     id  token               count    share  rank
----------------------------------------------------------------------
     32  \u0020            169,892  15.232%  #1
    101  e                  94,611   8.482%  #2
    116  t                  67,009   6.008%  #3
    ...
-- the 10 most frequent of 65 ids: 62.5% of the corpus; use --top 0 for every id
```

By default every id that occurs is listed, in id order; `--top N` shows only the
N most frequent (with the share of the corpus they account for) and `--limit N`
caps how many rows print. The same corpus with the GPT-2 BPE default is
`338,025` tokens over `11,706` distinct ids, which is the whole tokenizer
comparison, measured on the same bytes:

| `--tokenizer` | ids | tokens | distinct ids used | characters per token |
|---------------|-----|--------|-------------------|----------------------|
| `char` | 257 | 1,115,394 | 65 | 1.00 |
| `gpt2` | 50,257 | 338,025 | 11,706 | 3.30 |

Two extras make the table checkable rather than decorative:

```bash
python -m apexgpt data tokens --slice-at 5000          # a 24-token slice, id by id
python -m apexgpt data tokens --predict "KING RICHARD II:"
```

```
-- prompt, id by id --
text : 'KING RICHARD II:'
ids  : [75, 73, 78, 71, 32, 82, 73, 67, 72, 65, 82, 68, 32, 73, 73, 58]

-- next token, from bigram counts in this corpus --
last id : 58 = ':'
seen 10,316 times in this corpus
   1. id=10      count=  8,762 p=84.936%  '\n'
   2. id=32      count=  1,513 p=14.667%  ' '
```

`--predict` ranks successors by counting pairs in the corpus, so it needs no
model at all; a trained checkpoint does better, and
[🔮 Predict the next word](#%F0%9F%94%AE-predict-the-next-word-with-its-probability) prints
the model's own ranking. `tokens` reads text, so it never downloads the 772 MB
Wikipedia corpus — it asks you to `data prepare --source wikipedia
--target-mb 5` first.

### Models and training scripts from Hugging Face

Datasets, pretrained models and Kaggle datasets all have their own command —
[🌐 Hugging Face & Kaggle](docs/hub.md#%F0%9F%8C%90-hugging-face--kaggle) covers `hub check`, `hub files`,
`hub model`, `hub dataset` and `hub kaggle`, plus what `huggingface/transformers`
and `huggingface/trl` are the equivalents of here.

---

## 🎭 Tiny Shakespeare in 5 minutes

The fastest way to see next-token prediction actually work, on any machine, with
no GPU:

```bash
python -m apexgpt data prepare --source shakespeare     # 1.1 MB -> 338,025 tokens
python -m apexgpt train --dataset shakespeare \
    --preset cpu-tiny --max-iters 400 --run-name gpt-shakespeare
python -m apexgpt generate --checkpoint models/runs/gpt-shakespeare/checkpoint.pt \
    --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
python -m apexgpt generate --checkpoint models/runs/gpt-shakespeare/checkpoint.pt \
    --prompt "KING RICHARD II:" --predict 5
```

The `data prepare` report is itself the proof that the data is next-token shaped:

```
-- Next-token-prediction pair (x, y) --
x = [25, 475, 783, 198, 4366, 8181, 805, 1276, 1234, 319, 616, 32296, 290, 3830, 502, 198]
y = [475, 783, 198, 4366, 8181, 805, 1276, 1234, 319, 616, 32296, 290, 3830, 502, 198, 8496]
y is x shifted left by 1: True
decode(x) = ': but now\nSome hangman must put on my shroud and lay me\n'
decode(y) = ' but now\nSome hangman must put on my shroud and lay me\nWhere'
```

Sample output from the `cpu-tiny` (30.0M) model after 400 steps —
loss 10.82 → 5.62, so it has learned character names, verse line breaks and
Elizabethan syntax, not yet meaning:

```
prompt : ROMEO:
ROMEO:
  Nay, but a father.
And that you'st the mother's blood

FUS:
Thou I am all him, Thou
```

And with a longer prompt (`--temperature 0.7 --top-k 40`):

```
prompt : KING RICHARD II:
What must the king do now? Must he submit,

KING RICHARD II:
That he is I are a king is all my soul is so, and he had what I am a heart.

DUKE OF YORK:
LADY:
To make me;
And is this As you shall be a more.

KING RICHARD III:
```

That is the expected shape of the result: grammar, speaker labels and
Elizabethan cadence, with the sense falling apart — 30M parameters over 270k
tokens is a budget of about 0.6 epochs. `wikitext` or `tinystories` at a larger
step budget produces noticeably better English; the `full` preset on an NVIDIA
GPU produces prose.

---

## 🏗️ Architecture (MVC + service + feature-based)

Three organising principles, applied together:

```
apexgpt/
├── core/           cross-cutting: config, paths, device, environment, seeding
│   └── environment.py     the one scan: spec, requirements, settings, apply
│
├── models/         ── M ── the domain model
│   ├── gpt.py              GPT, attention, MLP, blocks, KV cache
│   ├── builder.py          construction, parameter counts, reporting
│   └── sampling.py         temperature / top-k / top-p / penalty
│
├── features/       ── feature-based vertical slices ──
│   ├── data/                service.py + cli.py + sources.py (corpus registry)
│   ├── training/            service.py  + cli.py
│   └── inference/           service.py  + cli.py + gui.py
│
├── api/            the one standalone service
│   └── server.py            optional FastAPI app
│
└── tools/          the operator-facing commands
    ├── setup.py             install for this hardware
    ├── env.py               scan, collect requirements, apply settings
    ├── doctor.py            verify the environment
    ├── lab.py               Jupyter Lab + the kernel spec
    └── notebook_sources.py  generates notebooks/*.ipynb
```

**MVC mapping**

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Model** | `models/` | Architecture and sampling maths |
| **Service** | `features/*/service.py`, `core/environment.py` | Business logic: data, training, generation, configuration |
| **View** | `features/*/cli.py`, `gui.py`, `tools/*.py`, `notebooks/` | Presentation only — no model logic |
| **Controller** | `cli.py` entry points | Parse args → call service → render |

**Why feature-based.** Each feature is a complete vertical — its service plus
its views. `features/training` does not import from `features/inference`, so
any slice can be lifted out into its own deployable without touching the rest.

**Why one service, not microservices.** Splitting data preparation and training
into separate network processes would add ports, health checks and
partial-failure modes to a workflow that is offline and single-user, without
making it better. So ApexGPT ships exactly one deployable — the inference API
— and keeps every other boundary a clean in-process service interface.

---

## 📐 The mathematics behind ApexGPT

Every formula the code implements, with the notation key: [docs/mathematics.md](docs/mathematics.md).

---

## 🤖 Model

```bash
python -m apexgpt train --preset full --summary-only
```

Everything is hand-written: packed QKV projection, `scaled_dot_product_attention`,
GELU MLP, pre-LayerNorm residual blocks, final LayerNorm, weight-tied head.

**Requested configuration — 12 layers / 768 hidden / 12 heads**

```
layers (n_layer)   : 12
hidden size        : 768
attention heads    : 12  (head_dim=64)
context/block size : 256
vocab size         : 50257
weight tying       : yes (lm_head shares wte)
total parameters   : 123,849,984  (123.8M)
embedding params   : 38,793,984   (38.8M)
transformer params : 85,056,000   (85.1M)
```

```
wte (token emb)      38,597,376  (38.60M)
wpe (pos emb)           196,608  ( 0.20M)
attention            28,348,416  (28.35M)
mlp                  56,669,184  (56.67M)
layernorms               38,400  ( 0.04M)
per transformer block  7,087,872  ( 7.09M)
```

**Presets**

| Preset | Layers | d_model | Heads | Params | CPU speed | 4000 iters |
|--------|--------|---------|-------|--------|-----------|-----------|
| `full` | 12 | 768 | 12 | 123.8M | 55.5 s/it | **~62 h** |
| `medium` | 8 | 512 | 8 | ~50M | — | — |
| `cpu-tiny` | 6 | 384 | 6 | 30.0M | 3.0 s/it | 3.3 h |
| `smoke` | 2 | 128 | 4 | 6.8M | — | seconds |

`--preset auto` is the default. It reads the detected backend, the accelerator's
VRAM and the core count, then picks from the table above — `full` with 24 GB of
VRAM, `medium` at 12 GB, `cpu-tiny` at 8 GB, `smoke` on a 2-core machine.

**KV cache** — cached decoding must use a *bottom-right aligned* mask. A plain
`is_causal=True` would let every query attend to future keys; the tests assert
cached decoding matches a full forward pass to `1e-4`.

---

## 🔤 Tokenizers

Tiny Shakespeare has its own vocabulary: no merge table, no download, no
`<unk>`, and nothing that can be thrown off by a character the corpus never
contained. The byte-level tokenizer maps each byte to an id `0–255` and uses
`256` for the eos token, so **every** input is representable — emoji, other
scripts, and a multi-byte character sliced in half at the block edge.

```bash
python -m apexgpt prepare --dataset shakespeare --tokenizer char
python -m apexgpt train  --dataset shakespeare --tokenizer char --preset cpu-tiny --max-iters 400
python -m apexgpt generate --checkpoint models/runs/gpt-shakespeare-char/checkpoint.pt --prompt "ROMEO:"
```

```python
from apexgpt.features.data.tokenizers import load_tokenizer, spec_for

spec = spec_for("char")            # 257 ids, eos 256, no merges
tok = load_tokenizer("char")
tok.decode([ord("R"), ord("O")])   # "RO"
```

The tokenizer is part of the run, not a global choice: the prepared corpus stores
its spec in `corpus.json`, the checkpoint records the tokenizer and its vocab
size in its metadata, and `generate` decodes with whatever the checkpoint was
trained on — loading a `char` checkpoint never silently runs it through GPT-2
BPE.
> Measured runs moved to [docs/measurements.md](docs/measurements.md).
## 🎓 Training

```bash
python -m apexgpt train --preset cpu-tiny --max-iters 600
python -m apexgpt train --show-history gpt-shakespeare
python -m apexgpt train --dataset shakespeare --resume models/runs/gpt-shakespeare/checkpoint.pt --max-iters 800
```

Forward → cross-entropy next-token loss → backward → gradient clipping → AdamW
step, with warmup and cosine decay, periodic validation, checkpointing, tqdm
progress, and matplotlib loss curves.
> Measured runs moved to [docs/measurements.md](docs/measurements.md).
> Measured runs moved to [docs/measurements.md](docs/measurements.md).
### ⚠️ On "2–3 epochs"

The Wikipedia corpus is **242M training tokens**. One epoch at 2048 tokens per
step is **118,206 steps** — weeks on this CPU. Training is therefore
**step-budgeted, not epoch-budgeted**. `--epochs` does work (it derives the count
from the real corpus size and warns you), but for a laptop run use `--max-iters`.

At 460k tokens (~0.19% of one epoch) the model learns English word and sentence
structure but not coherence. That is the expected result for 30M parameters at
this budget: coherent output needs the `full` preset on an NVIDIA GPU.

---

## 💬 Inference

```bash
python -m apexgpt generate --prompt "The history of the city is"
python -m apexgpt generate --prompt "Once upon a time" \
    --temperature 0.7 --top-k 40 --top-p 0.95 --max-new-tokens 300
python -m apexgpt generate --interactive
```

Interactive commands: `:t <temp>`, `:k <top-k>`, `:p <top-p>`, `:n <tokens>`,
`:seed <n>`, `:quit`.

All four front ends share `models/sampling.py`, so they behave identically.

### 🔮 Predict the next word, with its probability

Sampling answers "what did the model happen to write". `--predict` answers the
other question — *what did the model expect?* — by keeping the distribution
instead of sampling it and showing the ranked candidates:

```bash
python -m apexgpt generate --prompt "The history of the city is" --predict 8
python -m apexgpt generate --prompt "To be, or not" --next-word
```

```text
[predict] next token after 'The history of the city is'
    1. id=11     p= 0.0917 logp= -2.39  ,          ##########
    2. id=198    p= 0.0487 logp= -3.02  \n          ##
    3. id=257    p= 0.0391 logp= -3.24  \u0020a     ##
    4. id=616    p= 0.0218 logp= -3.83  \u0020my    #
    5. id=11     p= 0.0163 logp= -4.12  ,           #
    6. id=326    p= 0.0148 logp= -4.21  \u0020that  #
    7. id=465    p= 0.0135 logp= -4.31  \u0020his   #
    8. id=534    p= 0.0125 logp= -4.39  \u0020your
        top-8 hold 0.2400 of the mass; entropy 6.632 nats (uniform would be 10.825 over 50,257 tokens)
```

Read the last line as the health check it is: a uniform model over 50,257 tokens
has entropy `ln(50257) = 10.825` nats. **6.632 nats means the model has real
structure and is not guessing** — while the top 8 candidates holding only 24% of
the mass says the same thing from the other side. Whitespace is escaped
(`\u0020`, `\n`) because a column of blank-looking rows cannot be read or
copy-pasted.

The same model is confident where the corpus makes it confident, and the table
shows it — a character name expects a line break:

```text
[predict] next token after 'KING RICHARD II:'
    1. id=198    p= 0.9346 logp= -0.068  \n          #####################################
    2. id=314    p= 0.0054 logp= -5.223  \u0020I
    3. id=290    p= 0.0022 logp= -6.129  \u0020and
    4. id=475    p= 0.0015 logp= -6.482  \u0020but
    5. id=644    p= 0.0013 logp= -6.671  \u0020what
        top-5 hold 0.9449 of the mass; entropy 0.650 nats (uniform would be 10.825 over 50,257 tokens)
```

0.650 nats against 10.825 — the model has learned the shape of the corpus. The
`logp` column is the same quantity the training loop minimises, per candidate.

The same numbers are a library call and a REST field:

```python
from apexgpt.features.inference import InferenceEngine

engine = InferenceEngine(device="cpu").load()
rows = engine.predict_next("The history of the city is", top_k=5)
rows[0].text, rows[0].probability, rows[0].logprob
engine.predict_next_text("To be, or not")     # just the single best token
rows, entropy = engine.predict_next_with_entropy("To be, or not")
```

`--interactive` prints the same table before every completion.

---

## 🌐 Hugging Face & Kaggle

Pretrained models, hub datasets and the reference training scripts: [docs/hub.md](docs/hub.md).

---

## 🖥️ GUI

```bash
python -m apexgpt gui
```

A dark-themed Tk window: prompt box, **real-time token-by-token streaming** on
a background thread (the UI never freezes), live sliders for temperature,
top-k, top-p, max tokens, seed and repetition penalty, a KV-cache toggle, a
**Stop** button mid-generation, and a tok/s readout. **Show next token** prints
the ranked candidates into the output pane before generation starts.

Under the prompt box sits the **token bar**: the value of every id the prompt
turns into, and what the model expects to put after it. It is the same table
[`data tokens`](#%F0%9F%94%A2-every-token-with-its-id) prints and the same ranking
`generate --predict` prints, refreshed as you type (debounced, on a worker
thread, with a stale result dropped rather than shown):

```
26 id(s)  |  shakespeare.txt: 1,115,394 tokens, 65 distinct
      84  T                    7,015   0.629%
     104  h                   51,310   4.600%
     101  e                   94,611   8.482%
      32  \u0020             169,892  15.232%
     104  h                   51,310   4.600%
     105  i                   45,537   4.083%
     115  s                   49,696   4.455%
     116  t                   67,009   6.008%
     111  o                   65,798   5.899%
     114  r                   48,889   4.383%
     121  y                   20,448   1.833%
      32  \u0020             169,892  15.232%
  ... 14 more id(s)
next:  \u0020 21.4%  t 12.8%  h 12.5%  e 10.6%  o 7.9%   entropy 2.68 nats
```

Measured, not mocked: `gpt-shakespeare-char` (10.8M parameters) reading
`The history of the city is`.

The counts come from the corpus **this checkpoint was trained on** — the dataset
name is stored in the checkpoint, and the table is cached against the file's
size and modification time, so a rebuilt corpus invalidates it. Counting a large
corpus takes seconds, so it happens on a worker thread and the bar says
`counting...` until it is ready. With no corpus on disk the bar still shows the
ids and says so, rather than inventing a frequency.

> The 30M model produces word-like but incoherent text. That is the model, not
> the GUI — lower the temperature to `0.1` and output becomes repetitive, which
> confirms sampling is wired up correctly.

---

## 🌐 HTTP API

The service in one command, then the full endpoint reference: [docs/http-api.md](docs/http-api.md).

---

## 🤖 Android

Driving the API from a phone: [docs/android.md](docs/android.md).

---

## 🧪 Tests

```bash
pip install pytest httpx
python -m pytest tests -q
```

Every push and pull request runs the whole pipeline in
[`.github/workflows/tests.yml`](.github/workflows/tests.yml). Each failing test
is reported as a GitHub annotation, so a red run names itself instead of only
its exit code.

| Job | What it proves | Where |
|---|---|---|
| `lint` | `ruff` with only the rules that catch real defects — undefined names, redefinitions, unparseable syntax. Seconds, and it needs no torch | `apexgpt tests` |
| `test` | the suite on **Linux and Windows × Python 3.11 and 3.13**, then `lab --check` and `doctor`; the junit report is uploaded per job | the matrix |
| `newest-python` | the next Python release has not broken anything yet (3.14 today) | `ubuntu-latest` |
| `smoke` | **the program, not its parts**: build a corpus → print the token table → predict from it → train 30 real steps → generate → predict, keeping the loss curves | `ubuntu-latest`, ~4 min |
| `package` | the wheel builds, installs, runs the CLI from it, carries no tests/corpus/checkpoints, and is uploaded as an artifact | `ubuntu-latest` |

Two more things the workflow does deliberately: a **weekly cron run**, so an
upstream release that breaks something says so on a Monday morning instead of on
the day someone tries to use the project; and `permissions: contents: read` with
a `timeout-minutes` on every job, because a pipeline that can neither write to
the repository nor hang forever is a pipeline that fails where you can see it.
`tests/test_package.py` asserts this contract, so the gates cannot quietly
disappear in a later edit.

| File | Covers |
|------|--------|
| `tests/test_model.py` | architecture, causality, KV cache, sampling, LR schedule |
| `tests/test_data_and_inference.py` | tokenize/split/batch, engine streaming, token-value reports |
| `tests/test_data_sources.py` | corpus registry, URL/local/HF/Kaggle fetch, conversions, per-corpus paths |
| `tests/test_token_table.py` | `data tokens`: id table, counts and share, `--top`/`--limit`, slices, successor ranking, the shared inventory |
| `tests/test_device.py` | backend probes, precision policy, presets, wheel indexes |
| `tests/test_system.py` | live CPU/RAM/disk/process probes, their fallbacks, threshold arithmetic |
| `tests/test_environment.py` | the scan: spec, requirements, settings, overrides, load scan, shell export, CLI |
| `tests/test_tokenizers.py` | byte-level vocabulary, corpus specs, checkpoint metadata, per-tokenizer decoding |
| `tests/test_predict_and_hub.py` | next-token distribution and entropy, per-token logprobs, stop reasons, prediction CLI, Hub/Kaggle capability, `--allow` plumbing |
| `tests/test_lab.py` | kernel spec, notebook integrity, Lab CLI, a notebook executed in a real kernel |
| `tests/test_package.py` | every module imports, CLI wiring, layout, `pyproject.toml`, artifact paths, the CI pipeline's own contract |
| `tests/test_gui.py` | real Tk window, streaming, the token bar and its staleness rule, next-token panel, Stop button |
| `tests/test_api.py` | HTTP endpoints, SSE, logprobs, model discovery, validation, OpenAPI |

The tests that matter most are the ones that catch **silent** bugs: that a
causal mask leaks no future tokens, that cached decoding matches a full forward
pass, that gradient checkpointing is bit-identical to the plain path, and that a
`char` checkpoint is never decoded with GPT-2 BPE.

The GUI, API and Jupyter suites skip themselves automatically when Tk, FastAPI
or jupyterlab is unavailable, so the suite passes headless and without the
notebook extras. CI installs the `notebook` extra anyway, so those suites really
run there instead of quietly skipping.

---


## 🐛 Bugs found and fixed

What was wrong, and what it cost: [docs/bugs-found-and-fixed.md](docs/bugs-found-and-fixed.md).

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🔗 Links

- **Repository** — https://github.com/Gethubsathvik/ApexGPT
- **Dataset (default)** — [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.en`
- **Corpora** — [Tiny Shakespeare](https://github.com/karpathy/char-rnn) · [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) · [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) · [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)
- **Tokenizer** — GPT-2 BPE, vocab 50257


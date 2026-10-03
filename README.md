# 🧠 ApexGPT

> A GPT-style transformer language model built from scratch in PyTorch — dataset pipeline, training loop, CLI, desktop GUI, Jupyter Lab, and an optional HTTP inference service.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-338%20passing-success.svg)](tests)

---

## 📑 Contents

- [✨ What it does](#-what-it-does)
- [🖥️ Reference configuration](#%EF%B8%8F-reference-configuration)
- [💻 Hardware portability](#-hardware-portability)
- [🔍 Environment scan](#-environment-scan)
- [🚀 Quick start](#-quick-start)
- [🪟 Windows](#-windows)
- [🐧 Linux](#-linux)
- [🍎 macOS](#-macos)
- [📓 Jupyter Lab](#-jupyter-lab)
- [📚 Corpora](#-corpora)
- [🎭 Tiny Shakespeare in 5 minutes](#-tiny-shakespeare-in-5-minutes)
- [🏗️ Architecture](#%EF%B8%8F-architecture-mvc--service--feature-based)
- [📐 Mathematics](#%EF%B8%8F-the-mathematics-behind-apexgpt)
- [🤖 Model](#-model)
- [🔤 Tokenizers](#-tokenizers)
- [🎓 Training](#-training)
- [💬 Inference](#-inference)
- [🌐 Hugging Face & Kaggle](#-hugging-face--kaggle)
- [🖥️ GUI](#%EF%B8%8F-gui)
- [🌐 HTTP API](#-http-api)
- [🤖 Android](#-android)
- [🧪 Tests](#-tests)
- [📁 Layout](#-project-layout)
- [🐛 Bugs found and fixed](#-bugs-found-and-fixed)

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
- **Automated tests** — 338 of them, covering causality, the KV cache, sampling, portability, corpus fetching, the live system scan, the next-token distribution, notebooks and the GUI

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
> see [Hardware portability](#-hardware-portability).
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
> `/v1/models`, and interactive docs at `/docs`. Spell the defaults out with
> `python -m apexgpt serve --host 127.0.0.1 --port 8000`; bind `0.0.0.0`
> instead to let other machines on your network reach it. Full detail in
> [🌐 HTTP API](#-http-api).

## 💻 Hardware portability

ApexGPT has no hard-coded CUDA. `core/device.py` probes the machine once and every
script follows the result.

| Backend | Detected via | Precision policy | Checkpointing |
|---------|---------------|------------------|---------------|
| `cuda` | `torch.cuda.is_available()` (NVIDIA **or** an AMD ROCm build) | fp16 autocast + `GradScaler` | on when VRAM < 12 GB |
| `xpu` | `torch.xpu.is_available()` | fp16 autocast + `GradScaler` | off |
| `mps` | `torch.backends.mps.is_available()` | fp16 autocast | off |
| `dml` | optional `torch-directml` package | fp32 (autocast is not honoured) | off |
| `cpu` | always present | bf16 autocast **only if native**, else fp32 | off |

```bash
python -m apexgpt setup --backend auto   # detect backend, print the exact pip plan
python -m apexgpt setup --install        # detect backend, then install
python -m apexgpt doctor                 # report every backend, RAM, VRAM, threads
```

`setup` picks the wheel index that matches your hardware (cu124, cu121, rocm6.2,
xpu, cpu, or the default Metal-capable wheel). Because that choice cannot be
expressed as one pinned version, **`torch` is deliberately unpinned** in
`requirements.txt` — see the comment at the top of that file.

| Machine | Command |
|---------|---------|
| NVIDIA (any CUDA) | `python -m apexgpt setup --install` |
| NVIDIA, pinned CUDA | `python -m apexgpt setup --backend cuda --cuda 121 --install` |
| AMD ROCm (Linux) | `python -m apexgpt setup --backend rocm --rocm 6.2 --install` |
| Intel GPU | `python -m apexgpt setup --backend xpu --install` |
| Apple Silicon | `python -m apexgpt setup --install` (default wheel has Metal) |
| AMD/Intel GPU on Windows | `python -m apexgpt setup --backend dml --install`, then `--device dml` |
| CPU only | `python -m apexgpt setup --backend cpu --install` |
| + Jupyter Lab | add `--notebook` |
| + HTTP API | add `--api` |

Device selection is automatic (`--device auto`, the default) but never silently
wrong: asking for a backend that is absent is a hard error, not a CPU fallback.

```bash
python -m apexgpt train --device cuda:1     # multi-GPU index form
python -m apexgpt train --device cpu        # force the CPU
python -m apexgpt train --device dml        # opt into DirectML
python -m apexgpt train --preset auto       # size the run to this machine
```

`--preset auto` reads the detected backend, its VRAM and the core count, then
picks a preset that actually fits instead of defaulting to something that will
page or thrash.

> **About AMD GPUs.** There is no CUDA path for them, so `doctor` and `env` say
> `cuda available: False` rather than pretending otherwise. PyTorch reaches AMD
> hardware through a **ROCm build** (which reports itself as `cuda`; `doctor`
> prints `rocm build:` to disambiguate) or, on Windows, through **DirectML**.
> DirectML has no fused attention kernel, is usually *slower* than a full CPU
> thread pool for training, and is therefore opt-in. For a CPU-only machine the
> honest answer is the CPU.

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

<details open>
<summary><b>PowerShell</b></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m apexgpt setup --install
python -m apexgpt env
python -m apexgpt data prepare --source shakespeare
python -m apexgpt train --dataset shakespeare
python -m apexgpt gui
```

</details>

**If `Activate.ps1` is blocked** by the execution policy, use the interpreter directly — no activation needed:

```powershell
.\.venv\Scripts\python.exe -m apexgpt env
.\.venv\Scripts\python.exe -m apexgpt gui
```

Or unblock it once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Keep the corpus off the system drive.** By default data lands in `.\data`. If `C:` is tight, point it elsewhere:

```powershell
$env:APEXGPT_DATA_DIR = "C:\apexgpt_data"
```

<details>
<summary><b>GPU notes for Windows</b></summary>

`setup` detects `nvidia-smi` and installs the matching CUDA wheel automatically.
On a machine with only an AMD or Intel iGPU you will correctly get `cpu`,
because neither has a CUDA path in PyTorch. To try DirectML:

```powershell
python -m apexgpt setup --backend dml --install
python -m apexgpt env                      # dml available: True
python -m apexgpt train --device dml --preset smoke
```

</details>

---

## 🐧 Linux

```bash
sudo apt install python3-venv python3-pip     # Debian/Ubuntu
# Fedora: sudo dnf install python3 python3-pip

python3 -m venv .venv
source .venv/bin/activate

python -m apexgpt setup --install              # picks cu124 / rocm / cpu for you
python -m apexgpt env
python -m apexgpt data prepare --source shakespeare
python -m apexgpt train --dataset shakespeare
```

The GUI needs Tk, which is not always installed:

```bash
sudo apt install python3-tk      # Debian/Ubuntu
sudo dnf install python3-tkinter # Fedora
```

Run in a headless session? `train`, `generate` and `serve` all work without a display. Only `gui` needs one.

<details>
<summary><b>GPU notes for Linux</b></summary>

```bash
# NVIDIA
python -m apexgpt setup --backend cuda --cuda 124 --install

# AMD ROCm
python -m apexgpt setup --backend rocm --rocm 6.2 --install

# Intel
python -m apexgpt setup --backend xpu --install
```

For ROCm, PyTorch needs the matching `rocm-smi` libraries present; verify with
`rocm-smi` before trusting the detection. `python -m apexgpt env` then shows
`rocm build:` instead of a CUDA build.

</details>

---

## 🍎 macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m apexgpt setup --install
# Tk is bundled, but you may need:
xcode-select --install

python -m apexgpt env
python -m apexgpt data prepare --source shakespeare
python -m apexgpt train --dataset shakespeare
```

On **Apple Silicon** `setup` detects `mps` and installs the default wheel, which
includes Metal support. `doctor` will report `mps available: True`, and
`--device auto` selects it automatically — MPS is **not** CUDA, so check the
`mps` line rather than `cuda`. On Intel Macs it falls back to `cpu`.

> The Tk GUI on macOS is the least-tested surface here; the CLI and API are the
> dependable paths.

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
| `local:<path>` | a file on disk | — | — | `.txt`, `.csv`, `.json`, `.jsonl`, `.parquet` |
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
[🔤 Tokenizers](#-tokenizers):

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
[🔮 Predict the next word](#-predict-the-next-word-with-its-probability) prints
the model's own ranking. `tokens` reads text, so it never downloads the 772 MB
Wikipedia corpus — it asks you to `data prepare --source wikipedia
--target-mb 5` first.

### Models and training scripts from Hugging Face

Datasets, pretrained models and Kaggle datasets all have their own command —
[🌐 Hugging Face & Kaggle](#-hugging-face--kaggle) covers `hub check`, `hub files`,
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

Every formula below is either **used by this code** — with the file that
implements it — or listed as **not used**, with the reason. Nothing is decorative.
Where the distinction matters, the honest answer is more useful than a formula
that looks impressive.

### 📇 Formula index

The complete list, with the notation each symbol carries. Sections further down
derive *why* each one is here; this table is for looking one up.

| # | What it computes | Formula | Notation | Implemented |
|---|---|---|---|---|
| 1 | Training objective | $\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log\,\mathrm{softmax}(z_t)_{x_t}$ | $T$ tokens per block, $z_t$ logits at position $t$, $x_t$ the token that came next, $\theta$ the weights | `models/gpt.py:229` |
| 2 | Output layer / sampling distribution | $p_i = \dfrac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}$ | $z\in\mathbb{R}^{V}$ logits, $V$ vocabulary (50257 or 257) | `models/sampling.py:103` |
| 3 | Multi-head causal attention | $\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | $d_k = d/H$ per-head width (`n_head` × `head_dim` in the code), $Q=XW_Q,\ K=XW_K,\ V=XW_V$ | `models/gpt.py:63` |
| 4 | Causality mask | $M_{ij}=0$ if $j\le i$, $-\infty$ otherwise | $i$ query position, $j$ key position; $-\infty$ kills the softmax term | `models/gpt.py:71` |
| 5 | Every parameter's entry point | $z = Wx+b$ | $W$ weight matrix, $b$ bias vector | `nn.Linear`, `nn.Embedding` |
| 6 | MLP nonlinearity | $\mathrm{GELU}(z)=z\,\Phi(z),\quad \Phi(z)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z}e^{-t^2/2}\,dt$ | $\Phi$ = standard normal CDF | `models/gpt.py:88` |
| 7 | Feature-wise standardisation | $\mathrm{LN}(z)=\gamma\odot\frac{z-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ | $\mu,\sigma^2$ over the $d$ features of **one** token, $\gamma,\beta$ per-feature scale/shift, $\odot$ elementwise | `nn.LayerNorm` |
| 8 | Residual path | $x \leftarrow x + F(\mathrm{LN}(x))$ | pre-norm: keeps the skip path an identity | `models/gpt.py:107` |
| 9 | Weight tying | $W_{lm} = E_{token}$ | one embedding matrix, used as input lookup and output layer | `models/gpt.py:127` |
| 10 | Inverted dropout | $\tilde{z}_i=\dfrac{z_i}{1-p}\cdot m_i,\quad m_i\sim\mathrm{Bernoulli}(1-p)$ | $p=0.1$ on embeddings, attention weights and MLP outputs; scaled by $1/(1-p)$ during training, identity at inference | `models/gpt.py:40` |
| 11 | Initialisation | $w\sim\mathcal{N}(0,0.02^2)$; residual projections $w\sim\mathcal{N}\!\left(0,\left(\tfrac{0.02}{\sqrt{2L}}\right)^2\right)$ | $L$ layers; scales residual branches by $1/\sqrt{2L}$ | `models/gpt.py:131` |
| 12 | Optimiser (AdamW) | $\theta\leftarrow\theta-\alpha\frac{\hat m_t/(1-\beta_1^t)}{\sqrt{\hat v_t/(1-\beta_2^t)}+\epsilon}$ | $\alpha$ learning rate, $\hat m_t,\hat v_t$ bias-corrected moments, $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$ | `features/training/service.py:65` |
| 13 | Decoupled weight decay | $\theta \leftarrow \theta - \lambda\theta$ | $\lambda=0.1$, matrices only ($p.\mathrm{dim}\ge2$) | `features/training/service.py:64` |
| 14 | Gradient clipping | $g\leftarrow g\cdot\min\!\left(1,\frac{\tau}{\lVert g\rVert_2}\right)$ | $\tau=1.0$ | `features/training/service.py:307` |
| 15 | Learning-rate schedule | $\eta_t=\eta_{max}\frac{t+1}{T_w}$, then $\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi p)$ | $T_w$ warmup steps $= \max(10, 0.05T)$, $p$ decay progress, $\eta_{max}=3\times10^{-4}$, $\eta_{min}=3\times10^{-5}$ | `features/training/service.py:48` |
| 16 | Sampling temperature | $p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$ | $T>1$ flattens, $T<1$ sharpens, $T\le 0$ greedy `argmax` | `models/sampling.py:98` |
| 17 | Top-$k$ filter | $z_i\leftarrow-\infty$ when $z_i<z_{(k)}$ | $z_{(k)}$ = $k$-th largest logit | `models/sampling.py:34` |
| 18 | Nucleus (top-$p$) filter | keep the smallest $m$ with $\sum_{i\le m}p_{(i)}\ge p$ | sorted descending; the most likely token is always kept | `models/sampling.py:52` |
| 19 | Repetition penalty | $z_i\leftarrow z_i/\lambda$ if $z_i>0$, else $\lambda z_i$ | for already-generated ids; $\lambda=1.0$ means off | `models/sampling.py:23` |
| 20 | Predictive entropy | $H(p)=-\sum_i p_i\log p_i\in[0,\ln V]$ | natural log, so **nats**; $\ln V$ = uniform guessing | `models/sampling.py:76` |
| 21 | Reported log-probability | $\log p_i$ on the **untruncated** softmax at the same $T$ | top-$p$ would renormalise a certain token to $\log 1 = 0$ | `models/sampling.py:154` |
| 22 | KV cache | $K_{t}=[\,K_{<t}\,;\,k_t\,]$, attend over the concatenation | $O(T)$ per generated token instead of $O(T^2)$ | `models/gpt.py:53` |
| 23 | Steps per epoch | $\left\lfloor \dfrac{N_{train}}{B\cdot T_{blk}}\right\rfloor$ | $B$ batch size, $T_{blk}$ block size | `features/training/service.py:72` |
| 24 | Perplexity | $\mathrm{PPL}=e^{\mathcal{L}}$, comparable as $\mathcal{L}$ per character | only comparable within one tokenizer | reported as `val loss` |
| 25 | Forward-pass cost | $\mathrm{FLOPs}\approx 6N+12LHd^2T$ | $N$ parameters, $L$ layers, $H$ heads, $d$ hidden, $T$ tokens | `models/builder.py:26` |
| 26 | Machine statistics | $\mu=\frac1n\sum x_i$, $\sigma^2=\frac1n\sum(x_i-\mu)^2$, $\rho=\frac{\mathrm{Cov}}{\sigma_X\sigma_Y}$ | scan inputs: CPU, RAM, cores, per-process cost | `core/system.py` |
| 27 | CPU utilisation | $100\left(1-\frac{\Delta\,\mathrm{idle}}{\Delta\,\mathrm{idle}+\Delta\,\mathrm{kernel}+\Delta\,\mathrm{user}}\right)$ | counters are cumulative since boot, so **differences** of two samples | `core/system.py:164` (psutil), `:204` (Win32 `GetSystemTimes`) |
Four conventions that the table alone would hide:

- **Padding is excluded, not predicted.** Targets of `-1` are dropped from the
  mean (`ignore_index=-1`), so a padded position contributes no loss term.
- **No label smoothing.** The target is exactly one-hot; smoothing
  ($\mathcal{L} = (1-\varepsilon)\mathcal{L}_{x_t} + \frac{\varepsilon}{V}\sum_i \mathcal{L}_i$)
  is deliberately not applied, because the point of this project is to report
  the model's real uncertainty.
- **Everything is in nats.** Loss, entropy and log-probabilities all use the
  natural log, so $\ln V$ is the no-information baseline. Divide by
  $\ln 2 = 0.693$ for bits.
- **Log-probability is measured before truncation.** A token drawn after top-$p$
  collapse still reports its probability under the untouched softmax at the same
  temperature, which is the model's own uncertainty rather than an artefact of
  the filter.

The classification formulas that are deliberately *absent* — accuracy, precision,
recall, $F_1$, confusion matrix — are listed with their reasons in
[Deliberately not implemented](#-deliberately-not-implemented).

**Notation key.** One row per symbol, so no formula above needs a detour to
read:

| Symbol | Means | Symbol | Means |
|---|---|---|---|
| $T$ | tokens in one training block, or in the sampled window | $\theta$ | the full parameter vector of the network |
| $t$ | index of a token position, $1 \dots T$ | $\epsilon$ | numerical floor ($10^{-8}$ in AdamW, $10^{-5}$ in LayerNorm) |
| $x_t$ | the **target** token at position $t$ (the one that came next) | $\alpha$, $\eta_{max}$ | learning rate, its peak value |
| $z$, $z_t$ | logits, before softmax | $\eta_{min}$ | learning-rate floor after decay |
| $p_i$ | probability of token id $i$ | $\eta_t$, $T_w$ | learning rate at step $t$, warmup length |
| $V$ | vocabulary size: 50257 (GPT-2 BPE) or 257 (byte-level) | $\tau$ | gradient-clipping threshold (1.0) |
| $d$ | hidden width (`n_embd`) | $\lambda$ | weight decay (0.1) or repetition penalty ($\ge 1$) |
| $L$ | number of transformer blocks | $g_t$, $\hat m_t$, $\hat v_t$ | gradient, its first and second moment |
| $H$ | number of attention heads (`n_head` in the code) | $\beta_1$, $\beta_2$ | moment decay rates (0.9, 0.95) |
| $d_k$ | per-head width, $d/H$ | $p$ | nucleus mass (top-$p$) or schedule progress |
| $Q,K,V$ | query, key, value matrices | $k$ | number of tokens kept by top-$k$ |
| $W, b$ | weight matrix, bias vector | $B$, $T_{blk}$ | micro-batch size, block (context) length |
| $\gamma$, $\beta$ | LayerNorm scale and shift (not Adam's $\beta$) | $N$, $N_{train}$ | parameter count, training tokens |
| $\odot$ | elementwise (Hadamard) product | $\Phi$ | standard normal CDF |
| $\mu$, $\sigma^2$, $\sigma$ | mean, variance, standard deviation | $\rho$ | Pearson correlation |
| $m, v$ | Adam's uncorrected moments | $\lVert g\rVert_2$ | Euclidean norm of the gradient |
| $F$ | attention or MLP sublayer | $\ln$, $e$ | natural logarithm, $e^x$ |

### The one loss function this project trains

Language modelling is next-token prediction. For a token sequence
$x_1, \dots, x_T$ the model predicts each token from the ones before it, and the
training objective is the **cross-entropy** between its predicted distribution and
the one-hot distribution of the token that actually came next:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}),
\qquad
p_\theta(x_t \mid x_{<t}) = \mathrm{softmax}(z_t)_i \;\text{ at } i = x_t
$$

Cross-entropy is the KL divergence between the model's distribution and the
target distribution, with the target's own entropy $H(y)$ dropped because it is a
constant that cannot be optimised:

$$
D_{\mathrm{KL}}(P\|Q) = \sum_i P(x_i)\log\frac{P(x_i)}{Q(x_i)},
\qquad
\underbrace{H(P,Q)}_{\text{cross-entropy}} = \underbrace{D_{\mathrm{KL}}(P\|Q)}_{\text{optimised}} + \underbrace{H(P)}_{\text{constant}}
$$

This is the entire objective. Everything else — depth, attention, sampling — is a
way of estimating or using $p_\theta$. It is implemented in
`models/gpt.py` (the `targets` branch of `GPT.forward`) and reported as
`val loss`.

**How to read the numbers in this README.** The loss is in **nats**, i.e. the
average of $-\log p$. For a fresh model that is exactly $\ln V$:

| Vocabulary | Fresh model | Perfect model |
|---|---|---|
| GPT-2 BPE, $V = 50257$ | $\ln 50257 = 10.82$ nats/token | 0 |
| byte-level, $V = 257$ | $\ln 257 = 5.55$ nats/char | 0 |

So a byte-level model can reach a lower *number* than a BPE model and still be
worse at text. The comparable quantity is nats **per character**
(`nats/token ÷ chars-per-token`), which is why the tokenizer comparison in
[🔤 Tokenizers](#-tokenizers) is decided on 1.702 vs 2.461 nats/char rather than
on 5.6159 vs 2.4614.

### Attention

Scaled dot-product attention, one head:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\qquad Q = XW_Q,\; K = XW_K,\; V = XW_V
$$

The $1/\sqrt{d_k}$ factor is what keeps the softmax out of its saturating region:
if the entries of $q\cdot k$ have variance $\propto d_k$, the logits' variance
grows with $d_k$ too, `softmax` becomes a hard argmax, and the gradient vanishes.
Dividing by $\sqrt{d_k}$ restores unit variance. Implemented in
`models/gpt.py::CausalSelfAttention` via
`torch.nn.functional.scaled_dot_product_attention` with a causal mask, so the
mask, the scaling and the softmax are the fused kernel's, not a re-implementation.

### The neuron and its activations

Every parameter in the network enters through an affine map, and the network is a
stack of these with a nonlinearity between them — without the nonlinearity,
stacking $L$ layers is algebraically the same as one layer:

$$
z = Wx + b,
\qquad
\text{ReLU}(z) = \max(0, z),
\qquad
\text{GELU}(z) = z\,\Phi(z),
\qquad
\Phi(z) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z} e^{-t^2/2}\,dt
$$

ApexGPT's MLP is `GELU → Linear(4d) → Linear(d)` (`models/gpt.py::MLP`), and
`GELU` is used because it is smooth, which matters when the network is deep. The
positional and token embeddings are the exception: those are pure affine maps,
with no activation in between, because a lookup has nothing to be nonlinear.

### Normalisation

LayerNorm standardises each token's feature vector across the hidden dimension,
then rescales it so the network can still express whatever magnitude it needs:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} z_i,
\qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(z_i - \mu)^2,
\qquad
\mathrm{LayerNorm}(z) = \gamma \odot \frac{z-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
$$

The statistics are taken over features, never over the batch, which is why a
LayerNorm network is independent of batch size. The blocks here are
**pre-LayerNorm** ($x + \mathrm{Attn}(\mathrm{LN}(x))$), which keeps the residual
path an identity and is what makes a 12-layer stack trainable at this scale.
Implemented in `models/gpt.py` as `LayerNorm`.

### Optimisation

Gradient descent, and the two refinements this trainer uses:

$$
\theta_{t+1} = \theta_t - \alpha\,\nabla_\theta \mathcal{L}(\theta_t)
$$

$$
\underbrace{\hat{m}_t = \beta_1 \hat{m}_{t-1} + (1-\beta_1)\,g_t}_{\text{momentum}},
\qquad
\underbrace{\hat{v}_t = \beta_2 \hat{v}_{t-1} + (1-\beta_2)\,g_t^2}_{\text{second moment}}
$$

$$
\theta_{t+1} = \theta_t - \alpha\left(\frac{\hat{m}_t}{1-\beta_1^t}\right)\Bigg/\left(\sqrt{\frac{\hat{v}_t}{1-\beta_2^t}} + \epsilon\right)
$$

That is AdamW's update, where $g_t = \nabla_\theta\mathcal{L}$ and this project
uses $\beta_1 = 0.9$, $\beta_2 = 0.95$ (`core/config.py`), $\epsilon = 10^{-8}$
— note that $\beta_2$, not the canonical $0.999$: this trainer runs short,
CPU-sized runs where the extra memory of a slower-decaying second moment buys
nothing. Dividing by $\sqrt{\hat v_t}$ makes the step
size roughly $\alpha$ regardless of gradient magnitude, and the bias correction
$m/(1-\beta^t)$ fixes the fact that $\hat m_t$ and $\hat v_t$ start at zero and
would otherwise bias the first steps towards zero. **Decoupled** weight decay
(`AdamW`) applies the penalty to the weights directly instead of folding it into
the gradient, which decouples it from the adaptive rescaling; it is applied to
matrices only ($p.\mathrm{dim} \ge 2$), never to biases or LayerNorm gains, with
$\lambda = 0.1$.

Three more pieces of the training loop:

| | Formula | Where |
|---|---|---|
| gradient clipping | $\min(1,\ \tau/\lVert g\rVert_2)\cdot g$ | bounds a single bad batch; $\tau = 1.0$ |
| linear warmup | $\eta_t = \eta_{max}\dfrac{t+1}{T_w}$ for $t < T_w$ | $T_w = \max(10, 0.05\,T)$ — 5% of the run |
| cosine decay | $\eta_t = \eta_{min} + \tfrac12(\eta_{max}-\eta_{min})(1+\cos(\pi p))$, $p = (t-T_w)/(T-T_w)$ | the `lr` column in every loss table; $\eta_{max}=3\times10^{-4}$, $\eta_{min}=3\times10^{-5}$ |
| backpropagation | $\dfrac{\partial \mathcal{L}}{\partial w} = \dfrac{\partial \mathcal{L}}{\partial a}\cdot\dfrac{\partial a}{\partial w}$, applied by the chain rule backwards through every op | `loss.backward()` |

### Sampling, and why temperature is a division

At generation time the logits are turned into a distribution and sampled from,
after three optional filters — `models/sampling.py`:

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Raising the temperature $T > 1$ **flattens** the distribution (logits are divided,
so differences shrink); $T < 1$ sharpens it; $T = 0$ is greedy decoding and takes
`argmax`. `top_k` keeps the $k$ largest logits, `top_p` keeps the smallest set
whose cumulative probability reaches $p$ (nucleus sampling), and
`repetition_penalty` divides positive logits / multiplies negative ones for tokens
already generated. `--predict` bypasses all of it and shows the untouched
distribution from which sampling would have drawn.

**Entropy** of that distribution, in nats, is the diagnostic used by `--predict`:

$$
H(p) = -\sum_i p_i \log p_i,
\qquad 0 \le H(p) \le \ln V
$$

$H = \ln V$ is uniform guessing; $H = 0$ is a single forced token. The measured
6.632 nats against $\ln 50257 = 10.825$ says the model is far from uniform but
far from confident.

### Evaluation metrics, and which ones apply

Perplexity is the exponentiated loss, and it is the one metric worth quoting for
a language model:

$$
\mathrm{PPL} = e^{\mathcal{L}}
\quad\Rightarrow\quad
\text{BPE val } e^{5.6159} = 275,\qquad
\text{byte-level val } e^{2.4614} = 11.7
$$

Those are **not comparable** across tokenizers — 275 sounds 20× worse than 11.7
while being the better model, because each covers a different amount of text.
Per character, the BPE model's perplexity is $e^{1.702} = 5.48$ against the
byte-level model's $e^{2.4614} = 11.7$.

Classification metrics are **not** what a language model is evaluated with — this
project has no labels and no decision threshold, so there is no confusion matrix
to build one from. For completeness, and because the question always comes up:

$$
\mathrm{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN},
\qquad
\mathrm{Precision} = \frac{TP}{TP+FP},
\qquad
\mathrm{Recall} = \frac{TP}{TP+FN}
$$

$$
F_1 = \frac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}
$$

$F_1$ is their harmonic mean, so it collapses when either one does — which is the
property you want when both matter. The confusion matrix itself is:

$$
C = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}
$$

### Descriptive statistics behind the environment scan

The scan that sizes a run to the machine (`core/system.py`, `core/environment.py`)
is built from these:

$$
\mu = \frac{1}{n}\sum_{i=1}^{n} x_i,
\qquad
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2,
\qquad
\sigma = \sqrt{\sigma^2}
$$

$$
\mathrm{Cov}(X,Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu_X)(y_i-\mu_Y),
\qquad
\rho = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\,\sigma_Y} \in [-1, 1]
$$

CPU utilisation is a *ratio of differences*, because the counters are cumulative
since boot — comparing two samples, not two absolutes:

$$
\text{util} = 100\left(1 - \frac{\Delta\,\text{idle}}{\Delta\,\text{idle} + \Delta\,\text{kernel} + \Delta\,\text{user}}\right)
$$

That is why `_split_cpu_line` exists, and why the scan reports the load average
$\left(\frac{1}{5m}\sum_{i} D_i,\ \frac{1}{15m}\sum_{i} D_i,\ \frac{1}{60m}\sum_{i} D_i\right)$
where $D_i$ is the number of runnable processes — the same quantity the live
scan compares against `DEFAULT_MAX_CPU_PERCENT = 85.0`.

### Deliberately not implemented

Naming these is more useful than quietly implying support:

| Method | Formula | Why not here |
|---|---|---|
| Linear regression | $y = \beta_0 + \beta_1 x$ | nothing here is a continuous target |
| Cost function (MSE) | $J(\theta)=\frac{1}{n}\sum_i (y_i-\hat y_i)^2$ | regression loss; language modelling uses cross-entropy |
| Logistic regression | $p = \sigma(z) = \frac{1}{1+e^{-z}}$, $\ \mathcal{L}=-\frac1n\sum_i\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$ | binary classification — a language model is a softmax over 50,257 classes, which *contains* this as the $K=2$ case |
| Softmax (multiclass) | $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | **used** — `models/sampling.py`, and it *is* the model's output layer |
| Naive Bayes | $P(y\mid x) = \dfrac{P(x\mid y)\,P(y)}{P(x)}$ | conditional independence is false for language |
| K-Means | $\arg\min_c \lVert x_i - \mu_{c_i}\rVert_2^2$ | no clustering step; the tokenizer vocabulary is not learned by clustering |
| SVM | $f(x) = w^\top x + b$, maximise margin subject to $y_i f(x_i) \ge 1$ | no support vectors, no kernel trick |
| L1 / L2 regularisation | $\lambda\sum_i\lvert\beta_i\rvert$ / $\lambda\sum_i \beta_i^2$ | **L2 via AdamW, weight decay only** — no L1, no sparse weights |
| Bias–variance | $\mathbb{E}[(y-\hat f(x))^2] = \mathrm{Bias}^2[\hat f] + \mathrm{Var}[\hat f] + \sigma^2$ | the decomposition is descriptive, not something a run reports |
| Gradient boosting (XGBoost, LightGBM) | $\hat y = \sum_{k} f_k(x)$, $f_k$ fits the residual gradient | trees do not tokenise; a 123.8M-parameter transformer is the model here. `sklearn` is not a dependency |
| Vector database | embeddings + ANN index (HNSW, IVF) | there is no retrieval step: the context is a fixed-size window of the corpus, not a search over a store |
| Mutual information | $I(X;Y) = H(X) - H(Y\mid X) = H(X)+H(Y)-H(X,Y)$ | meaningful, but nothing in this repo measures it; `--predict`'s entropy is the one place it would come from |
| KL divergence | $D_{\mathrm{KL}}(P\|Q)=\sum_i P_i\log\frac{P_i}{Q_i}$ | **used** — it *is* the training loss, cross-entropy minus the target entropy (see above) |

Fine-tuning, reinforcement learning from human feedback, LoRA and quantisation
are equally absent, and saying so is more useful than a stub: a 30M-parameter
model trained on one corpus is not a base model anybody can fine-tune
meaningfully. `--resume` continues *this* trainer's own checkpoints.

### Cost of one forward pass

The `full` preset's estimated FLOPs per token, and the reason the presets differ
so much in wall-clock:

$$
\mathrm{FLOPs} \approx 6N + 12\,LHd^2T
\qquad
(6N:\ \text{matmul},\;\; 12LHd^2T:\ \text{attention scores and values})
$$

with $N$ parameters, $L$ layers, $H$ heads, $d$ hidden, $T$ tokens. The second
term is quadratic in sequence length, which is exactly what the KV cache exists
to avoid: generating token $T+1$ with a cache costs one forward pass over the new
token only, not over the whole window. `models/builder.py::estimate_flops`
implements it.

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

### Measured: GPT-2 BPE vs. byte-level, `cpu-tiny`, 400 steps

| | GPT-2 BPE (`gpt2`) | Byte-level (`char`) |
| --- | --- | --- |
| vocabulary | 50,257 | **257** |
| tokens for the corpus | 338,025 | **1,115,394** |
| characters per token | 3.30 | 1.00 |
| characters actually observed | — | 65 |
| parameters (`cpu-tiny`) | 30.0M | **10.8M** |
| speed on 8 CPU threads | ~230 tok/s | **~500 tok/s** |
| characters seen in 400 steps | 1.01M | 0.31M |
| validation loss | 5.6159 nats/token | 2.4614 nats/char |
| **validation loss per character** | **1.702 nats/char** | 2.4614 nats/char |
| train / validation split | 268,364 / 69,661 | 892,315 / 223,079 |

The two losses are only comparable once they are divided by characters, which is
why the table ends in nats/char. BPE stays the default: at equal steps each of
its sequences covers 3.3× more text, and that outweighs its size. `char` is
worth choosing when you want the smaller model and the faster iteration — 3×
fewer parameters, because the embedding drops from 19.3M to 0.10M, and ~2×
throughput — and its output is visibly rougher at the same step count:

```text
ROMEO:
Wit lllll he, wif me he thome
Fou ase dseis y thers omyongo illll dirold nd are he hey it te mere s
```

---

## 🎓 Training

```bash
python -m apexgpt train --preset cpu-tiny --max-iters 600
python -m apexgpt train --show-history gpt-shakespeare
python -m apexgpt train --dataset shakespeare --resume models/runs/gpt-shakespeare/checkpoint.pt --max-iters 800
```

Forward → cross-entropy next-token loss → backward → gradient clipping → AdamW
step, with warmup and cosine decay, periodic validation, checkpointing, tqdm
progress, and matplotlib loss curves.

### Completed run — `cpu-tiny` on tiny Shakespeare, 400 steps, 29 min

```
[mem]      amp=False grad_checkpointing=False   (CPU: bf16 is emulated ...)
 step  train   val     lr
  100 6.1314 6.1929 2.72e-04
  200 5.3189 5.8664 1.77e-04
  300 5.2917 5.7019 7.44e-05
  400 5.4510 5.6159 3.00e-05
best val: 5.6159 at step 400
```

Validation loss fell from **10.82** (ln 50257 — random guessing) to **5.62** on a
270k-token corpus, in 29 minutes on a CPU with no GPU, at ~230 tok/s.

### Completed run — `cpu-tiny` on Wikipedia, 600 steps, 44 min

```
 step  train   val     lr
   25 9.3828 9.2700 2.50e-04
  125 7.6041 7.6740 2.82e-04
  225 7.3825 7.2816 2.30e-04
  325 7.3503 7.2753 1.58e-04
  425 7.3957 7.1531 8.87e-05
  525 6.9428 7.0375 4.17e-05
  600 7.0739 7.2499 3.00e-05
best val: 7.0375 at step 525
```

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

```bash
python -m apexgpt hub check                              # what this machine can do
python -m apexgpt hub files gpt2                         # what is in a repo
python -m apexgpt hub dataset Salesforce/wikitext        # download dataset files
python -m apexgpt hub model gpt2 --predict "Once upon a time"
python -m apexgpt hub kaggle inria/tiny-imagenet         # download a Kaggle dataset
```

### 🎯 Run a pretrained model and ask it the same question

`hub model <repo_id> --predict "<prompt>"` downloads the repo, loads it with
`transformers`, and prints the next-token distribution — the identical output
format `--predict` gives for a model trained here, so the two are directly
comparable:

```bash
python -m apexgpt hub model gpt2 \
    --predict "Once upon a time" --top-k 8 --max-new-tokens 200 --device auto
```

Two details worth knowing. First, `hub model` fetches **only** the config, the
tokenizer and the weights (`*.json`, `*.txt`, `*.safetensors`, `*.bin`); the
`gpt2` repo also ships the same model as ONNX, TensorFlow, Flax and Rust, which
is 3.5 GB of files PyTorch never reads. `--allow` overrides the list. Second,
`transformers` is imported lazily inside the call, so a machine that only trains
ApexGPT never loads it.

> If a download fails with a **404 on `xet-read-token`**, the xet storage
> backend cannot authenticate anonymously on that network. Retry with
> `HF_HUB_DISABLE_XET=1` set, or log in with `HF_TOKEN`. Both are surfaced by
> `hub check`.

### 📥 Datasets

```bash
python -m apexgpt data prepare --source hf:roneneldan/TinyStories   # streamed, cut at --target-mb
python -m apexgpt data prepare --source kaggle:user/dataset-slug    # Kaggle credentials required
python -m apexgpt hub dataset Salesforce/wikitext --allow '*.parquet'
python -m apexgpt hub kaggle user/dataset-slug
```

`hub dataset` and `hub kaggle` put the raw files on disk under `data/hub/`
without converting them, for when the parquet or JSONL layout matters. `data
prepare` is the other path: it ends with one plain UTF-8 text file, which is what
the tokenizer and the `uint16` binaries are built from. Both land in the same
`data/raw/<key>/` convention afterwards.

**Kaggle credentials are not a pip package.** Create an API token at
*Kaggle → Settings → API* and save it as `~/.kaggle/kaggle.json`
(`%USERPROFILE%\.kaggle\kaggle.json` on Windows), then:

```bash
python -m apexgpt setup --hub --install    # kagglehub, or use the kaggle CLI
python -m apexgpt hub check                # confirms client + credentials
```

`hub check` prints the whole picture in one screen — Hub client, `transformers`,
`datasets`, Kaggle client, Kaggle credentials, token present, and the two
directories downloads land in:

```text
  models   : .../data/hub/models
  datasets : .../data/hub/datasets

  [ok] download Hugging Face datasets             public repos need no token; private ones need HF_TOKEN
  [ok] stream Hugging Face datasets into a corpus python -m apexgpt data prepare --source hf:<repo_id>
  [ok] download and run a Hugging Face model      pip install -r requirements-hub.txt
  [--] Hub authentication                         huggingface-cli login, or set HF_TOKEN
  [--] download Kaggle datasets                   pip install kagglehub (or the kaggle CLI) and put kaggle.json in ~/.kaggle/
```

### 🧑‍🏫 Reference training scripts

ApexGPT trains its own GPT rather than loading one, so the Hub's training scripts
are the equivalent of `features/training/service.py` for pretrained models —
read them, do not depend on them:

| Upstream | Script | Equivalent here |
|----------|--------|-----------------|
| `huggingface/transformers` | `examples/pytorch/language-modeling/run_clm.py` | `features/training/service.py` — corpus → batches → AdamW → checkpoints |
| `huggingface/trl` | `SFTTrainer` | supervised fine-tuning of a pretrained checkpoint; the same loop with a different data source |
| `huggingface/tokenizers` | `Tokenizer` training | `features/data/tokenizers.py` for the byte-level vocabulary |

Fine-tuning a released checkpoint is `--resume`, which restores the model, the
optimizer and the step count rather than starting over:

```bash
python -m apexgpt train --dataset shakespeare --resume <checkpoint.pt> --max-iters 800
```

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
[`data tokens`](#-every-token-with-its-id) prints and the same ranking
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

Optional, and the one part of ApexGPT that runs as its own process.

### The one-line command

```bash
python -m apexgpt serve
```

That is the whole thing: it binds **http://127.0.0.1:8000** (the defaults) and
finds the most recent real checkpoint on its own, loading the tokenizer that
checkpoint recorded. Written out in full — the same thing, with the defaults
spelled out:

```bash
python -m apexgpt serve --host 127.0.0.1 --port 8000
```

`--host`, `--port`, `--checkpoint` and `--device` only override what it would
have picked. Bind `0.0.0.0` instead of `127.0.0.1` to let other machines on your
network reach it. Check it is alive with:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","model":{"checkpoint":"models/runs/gpt-shakespeare-char/checkpoint.pt",
 "parameters_m":10.8,"block_size":192,"vocab_size":257,"tokenizer":"char",
 "step":400,"val_loss":2.461369639635086,"device":"cpu"}}
```

Interactive API docs, including every field of every request, are at
**http://127.0.0.1:8000/docs**.

### Everything else

```bash
pip install -r requirements-api.txt
python -m apexgpt serve --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | liveness + model metadata |
| `/generate` | POST | one-shot generation, JSON in / JSON out, with logprobs |
| `/predict` | POST | the ranked next-token distribution, nothing sampled |
| `/stream` | GET | server-sent events, one `data:` line per token |
| `/v1/completions` | POST | OpenAI-compatible alias for third-party clients |
| `/v1/models` | GET | model discovery, which OpenAI clients probe first |
| `/docs` | GET | interactive OpenAPI docs |

> `/predict` takes the same request body as generation, so the number of
> candidates is **`top_k`** — not `k`. An unrecognised field is ignored, so
> `{"prompt": "…", "k": 5}` quietly returns all 50 rows instead of 5.

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The history of the city is","max_new_tokens":40,"temperature":0.8}'

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The history of the city is","top_k":5}'

curl -N "http://localhost:8000/stream?prompt=hello&max_new_tokens=20"
```

```
data: {"type": "meta", "parameters_m": 30.0, ...}
data: {"type": "token", "text": " List", "token_id": 2534, "logprob": -1.83, "stop_reason": null}
data: {"type": "token", "text": " Faction", "token_id": 18965, "logprob": -2.41, "stop_reason": null}
data: {"type": "done"}
```

Every event carries the token's own log-probability — the number the training
loop minimises — so a client can score the model instead of only reading it:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"apexgpt","prompt":"hello","max_tokens":20}'
```

```json
{
  "id": "cmpl-1759000000",
  "object": "text_completion",
  "choices": [{
    "index": 0,
    "text": "...",
    "logprobs": {"tokens": [" List", " Faction"], "token_logprobs": [-1.83, -2.41]},
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 2, "completion_tokens": 20, "total_tokens": 22},
  "apexgpt": {"elapsed_s": 0.31, "device": "cuda:0", "...": "..."}
}
```

`logprobs` used to be hardcoded to `null` and `finish_reason` to `"length"`; both
are now measured — `finish_reason` is `eos` when the model emitted the end token
and `length` when it ran out of budget.

> `stream: true` is refused there with a `400` pointing at `/stream`: OpenAI
> streams token objects, ApexGPT streams bare text, and faking that shape would
> break more clients than it would serve.

This is what makes the project usable from a phone or another program without
linking against Python.

**Security.** No authentication, no rate limiting, no TLS. Bind to `127.0.0.1`
unless you mean otherwise.

---

## 🤖 Android

Being straight about this: **PyTorch training on Android is not a realistic target**, and this project does not ship an Android app. What is genuinely possible:

**1. Termux — run the CLI on the phone** (slow but real)

```bash
pkg install python clang libjpeg-turbo
python -m apexgpt setup --install          # Termux has no GPU, so this picks cpu
python -m apexgpt env
python -m apexgpt train --dataset shakespeare --preset smoke
```

Expect CPU-only training on a phone to be roughly **20-50x slower** than a laptop — and phone SoCs are usually ARM with far less memory bandwidth. Practical for the CLI, the test suite and inference; not for real training runs. tiny Shakespeare is the corpus to use here: 1.1 MB instead of 1.3 GB.

**2. A client for the HTTP API** — the supported route

Run `serve` on your PC, bind it to the LAN, and call it from any phone browser or app:

```bash
python -m apexgpt serve --host 0.0.0.0 --port 8000
```

```bash
# from the phone's browser or any HTTP client
curl "http://192.168.1.50:8000/stream?prompt=hello&max_new_tokens=20"
```

Because `/v1/completions` speaks the **OpenAI completion shape**, existing
Android/Flutter clients work without modification — point them at your machine
and set the model name to anything:

```bash
curl -X POST http://192.168.1.50:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"apexgpt","prompt":"hello","max_tokens":20}'
```

> **Do not expose this to the internet as-is.** There is no authentication,
> rate limit or TLS. Bind it to `127.0.0.1` or a trusted LAN, and put a reverse
> proxy in front of it if it needs to leave your network.

**3. On-device inference** would need an ONNX/TFLite export plus a Kotlin or
Flutter client. That is a real piece of work with a genuine accuracy cost from
conversion, and it is **not implemented here**.

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

## 📁 Project layout

```
ApexGPT/
├── apexgpt/
│   ├── __main__.py            # single command dispatcher
│   ├── core/                   # config, paths, device, environment, system scan, text, seeding
│   ├── models/                 # M: GPT architecture, builder, sampling
│   ├── features/
│   │   ├── data/               # service.py + cli.py + sources.py + tokenizers.py
│   │   ├── training/           # service.py + cli.py
│   │   ├── inference/          # service.py + cli.py + gui.py
│   │   └── hub/                # service.py + cli.py: Hugging Face / Kaggle
│   ├── api/server.py           # optional FastAPI service
│   └── tools/                  # setup, env, doctor, lab, notebook_sources
├── notebooks/                  # generated .ipynb: environment, train, inference
├── tests/
├── models/runs/<name>/         # checkpoint.pt, history.json, loss_curves.png
├── data/raw/<corpus>/          # cached corpus text
├── data/binary/<corpus>/       # train.bin, val.bin (uint16)
├── data/hub/                   # Hugging Face / Kaggle downloads
├── apexgpt.settings.json       # persisted overrides (written by --save-settings)
├── pyproject.toml              # packaging: pip install -e . -> the apexgpt command
├── .github/workflows/tests.yml # CI: pytest on Linux/Windows, wheel build
├── requirements.txt
├── requirements-api.txt
├── requirements-hub.txt
├── requirements-notebook.txt
└── README.md
```

---

## 🐛 Bugs found and fixed

Notable defects caught during the audit and regression-tested:

| Bug | Impact |
|-----|--------|
| KV cache used `is_causal=False` on cached multi-token steps | queries attended to **future tokens** |
| `set_gradient_checkpointing()` folded the flag with `.training` | silently no-op'd unless called after `.train()` |
| `--epochs` accepted and ignored | users got a different run than they asked for |
| Resume discarded history and best-val | loss curves silently restarted |
| `np.memmap` reopened per batch | re-read a 461 MB file every step |
| `np.unique` on 242M tokens | sorted the whole corpus for a log line |
| `transformers` 5.18 `save_pretrained` wrote an empty tokenizer | silently produced a **zero-token corpus** |
| `--force` deleted the 1.2 GB corpus | 20-minute rebuild, no warning |
| `_split_binary` returned total tokens instead of train count | validation split came out empty |
| `/stream` handler contained a `yield` | FastAPI consumed the generator and returned an **empty body** |
| Pydantic models defined inside a factory | unresolved forward refs, schema generation crashed |
| Test-artifact checkpoints shadowed real runs | `generate` silently loaded the wrong model |
| GUI read a Tk variable from a worker thread | `RuntimeError: main thread is not in main loop` |
| `python -m apexgpt data prepare` — documented in the README but not implemented | every corpus command in the docs errored out |
| ROCm index built as `cu124rocm`, XPU index missing the `download.` host | both wheel installs 404'd |
| Assigning `DataConfig.dataset` left `binary_dir` alone | trained on the **previous** corpus's tokens |
| `DataConfig` `raw_dir`/`binary_dir` could not be overridden per corpus | every corpus shared one directory |
| Char-corpus reports decoded with GPT-2 BPE | `UnicodeEncodeError: 'charmap' codec can't encode '\ufffd'` on Windows consoles |
| `ByteTokenizer.decode` masked ids into bytes | the eos id (256) decoded to a NUL character instead of nothing |
| Windows `System Idle Process` (pid 0) in the process table | topped the "busiest processes" list forever, at a nonsense 200% CPU |
| `hub model --allow … --predict …` | the file patterns were dropped, so a 3.5 GB repo downloaded in full |
| `tokenizer(...)` assumed a `BatchEncoding` | `AttributeError` on any tokenizer that returns a plain dict |
| The GUI's next-token panel read a Tk variable from the worker thread | `RuntimeError: main thread is not in main loop` — the same trap as the earlier streaming bug |
| `PROJECT_ROOT` was `parent.parent.parent` unconditionally | an installed copy wrote checkpoints and corpora into `site-packages` |
| `/v1/completions` hardcoded `"logprobs": null` and `finish_reason: "length"` | clients could not score the model, and an eos stop was reported as a full-length one |
| `data tokens` sent the default corpus to `fetch_corpus` | `ValueError: source kind 'wikipedia' is handled by the data service`, so the one command that only reads text could not inspect the default corpus at all |
| A lab test called `lab.main(["--register-only"])` for real | it returns 1 when jupyterlab is absent, so all four CI jobs failed on a machine-dependent test while the local suite passed |
| The optimiser section documented $\beta_2 = 0.999$ | the code uses `beta2 = 0.95`; the README described Adam's textbook default, not this project's |
| The GUI text bar skipped a refresh when a count was still running | typing during a corpus count silently left the previous prompt's values on screen; now a stale result is dropped and the newer prompt is counted |
| The token table was printed by the CLI, so nothing else could use it | the counting lived in the view; it is now `build_inventory` in the data service, which the GUI reads |

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🔗 Links

- **Repository** — https://github.com/Gethubsathvik/ApexGPT
- **Dataset (default)** — [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.en`
- **Corpora** — [Tiny Shakespeare](https://github.com/karpathy/char-rnn) · [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) · [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) · [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)
- **Tokenizer** — GPT-2 BPE, vocab 50257

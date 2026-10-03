# 🧠 TinyLLM

> A GPT-style transformer language model built from scratch in PyTorch — dataset pipeline, training loop, CLI, desktop GUI, Jupyter Lab, and an optional HTTP inference service.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-225%20passing-success.svg)](tests)

---

## 📑 Contents

- [✨ What it does](#-what-it-does)
- [🖥️ Verified hardware](#%EF%B8%8F-verified-hardware)
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
- [🤖 Model](#-model)
- [🎓 Training](#-training)
- [💬 Inference](#-inference)
- [🖥️ GUI](#%EF%B8%8F-gui)
- [🌐 HTTP API](#-http-api)
- [🤖 Android](#-android)
- [🧪 Tests](#-tests)
- [📁 Layout](#-project-layout)
- [🐛 Bugs found and fixed](#-bugs-found-and-fixed)

---

## ✨ What it does

- **Scans the machine once** — CPU, cores, RAM, adapters, every PyTorch backend, and every requirement — then resolves and applies the settings TinyLLM will run with
- Runs on **CPU, NVIDIA CUDA, AMD ROCm, Intel XPU, Apple Metal, and Windows DirectML** — detected automatically, never silently
- Trains a **GPT from scratch** — no `transformers` model classes, every layer hand-written
- **123.8M parameters** at the requested 12 layers / 768 hidden / 12 heads
- Corpora: **tiny Shakespeare**, **Wikipedia**, **WikiText-2**, **TinyStories**, **OpenWebText**, any **Hugging Face** dataset, any **Kaggle** dataset, or a local file
- Next-token prediction with AdamW, warmup + cosine decay, and gradient clipping
- Samples with **temperature, top-k, top-p**, and repetition penalty
- Four front ends over one implementation: **CLI**, **Tkinter GUI**, **Jupyter Lab**, **HTTP service**
- **Automated tests** — 225 of them, covering causality, the KV cache, sampling, portability, corpus fetching, notebooks and the GUI

---

## 🖥️ Verified hardware

Everything below was measured on the machine this was built on.

| Item | Value |
|------|-------|
| CPU | **AMD Ryzen 3 7320U**, 4 cores / 8 threads |
| GPU | **AMD Radeon integrated graphics** — no NVIDIA GPU |
| CUDA | **Not available** — `nvidia-smi` and `nvcc` absent |
| Python | 3.14.7 (64-bit) |
| PyTorch | 2.14.1+cpu |
| RAM | ~8 GB (corpus is streamed to disk to stay inside it) |

`python -m tinyllm env` prints that same table for whatever machine it runs on,
plus the requirements and the resolved settings:

```
Machine spec
  CPU            : AMD Ryzen 3 7320U with Radeon Graphics
  cores          : 4 cores / 8 threads
  GPU            : AMD Radeon(TM) Graphics
  CUDA           : not available - no NVIDIA GPU or driver on this machine
  usable backend : cpu (AMD64 Family 23 Model 160 Stepping 0, AuthenticAMD)
  memory         : 7.8 GB RAM
  torch          : 2.14.1+cpu
  python         : 3.14.7 (CPython)
```

> ### ⚠️ Three deviations from the original spec
>
> **1. CUDA cannot be used here.** PyTorch CUDA kernels need an NVIDIA GPU and driver. This machine has an AMD iGPU, so `torch.cuda.is_available()` is `False` and training runs on CPU. **Every entry point still auto-selects the best available backend**, so the same code uses an NVIDIA GPU, an Intel GPU or Apple Metal with no edits — see [Hardware portability](#-hardware-portability).
>
> **2. The CPU is an AMD Ryzen 3, not an Intel i5.** All 8 logical threads are used.
>
> **3. On this CPU, AMP and gradient checkpointing are switched off automatically.** They are normally big wins on a GPU, but here they are a **66x penalty**:
>
> | Setting | Time/iter | Throughput |
> |---------|-----------|-----------|
> | bf16 autocast + gradient checkpointing | 195.1 s | 4 tok/s |
> | plain fp32, no checkpointing | **2.95 s** | **260 tok/s** |
>
> Zen2 has no native bf16, so autocast emulates it, and checkpointing's recompute blocks the fused attention path. This is detected, not hard-coded: newer CPUs with native bf16 keep AMP on. Force either back on with `--amp` / `--checkpointing`.

---

## 💻 Hardware portability

TinyLLM has no hard-coded CUDA. `core/device.py` probes the machine once and every
script follows the result.

| Backend | Detected via | Precision policy | Checkpointing |
|---------|---------------|------------------|---------------|
| `cuda` | `torch.cuda.is_available()` (NVIDIA **or** an AMD ROCm build) | fp16 autocast + `GradScaler` | on when VRAM < 12 GB |
| `xpu` | `torch.xpu.is_available()` | fp16 autocast + `GradScaler` | off |
| `mps` | `torch.backends.mps.is_available()` | fp16 autocast | off |
| `dml` | optional `torch-directml` package | fp32 (autocast is not honoured) | off |
| `cpu` | always present | bf16 autocast **only if native**, else fp32 | off |

```bash
python -m tinyllm setup --backend auto   # detect backend, print the exact pip plan
python -m tinyllm setup --install        # detect backend, then install
python -m tinyllm doctor                 # report every backend, RAM, VRAM, threads
```

`setup` picks the wheel index that matches your hardware (cu124, cu121, rocm6.2,
xpu, cpu, or the default Metal-capable wheel). Because that choice cannot be
expressed as one pinned version, **`torch` is deliberately unpinned** in
`requirements.txt` — see the comment at the top of that file.

| Machine | Command |
|---------|---------|
| NVIDIA (any CUDA) | `python -m tinyllm setup --install` |
| NVIDIA, pinned CUDA | `python -m tinyllm setup --backend cuda --cuda 121 --install` |
| AMD ROCm (Linux) | `python -m tinyllm setup --backend rocm --rocm 6.2 --install` |
| Intel GPU | `python -m tinyllm setup --backend xpu --install` |
| Apple Silicon | `python -m tinyllm setup --install` (default wheel has Metal) |
| AMD/Intel GPU on Windows | `python -m tinyllm setup --backend dml --install`, then `--device dml` |
| CPU only | `python -m tinyllm setup --backend cpu --install` |
| + Jupyter Lab | add `--notebook` |
| + HTTP API | add `--api` |

Device selection is automatic (`--device auto`, the default) but never silently
wrong: asking for a backend that is absent is a hard error, not a CPU fallback.

```bash
python -m tinyllm train --device cuda:1     # multi-GPU index form
python -m tinyllm train --device cpu        # force the CPU
python -m tinyllm train --device dml        # opt into DirectML
python -m tinyllm train --preset auto       # size the run to this machine
```

`--preset auto` reads the detected backend, its VRAM and the core count, then
picks a preset that actually fits instead of defaulting to something that will
page or thrash.

> **About AMD GPUs.** There is no CUDA path for them, so `doctor` and `env` say
> `cuda available: False` rather than pretending otherwise. PyTorch reaches AMD
> hardware through a **ROCm build** (which reports itself as `cuda`; `doctor`
> prints `rocm build:` to disambiguate) or, on Windows, through **DirectML**.
> DirectML has no fused attention kernel, is usually *slower* than a full CPU
> thread pool for training, and is therefore opt-in. For this machine's Radeon
> iGPU the honest answer is the CPU.

---

## 🔍 Environment scan

One command answers "can this machine run TinyLLM, and with what settings?" — the
same scan backs the CLI, the GUI, the API and the Jupyter kernel.

```bash
python -m tinyllm env                 # scan, report, and apply in this process
python -m tinyllm env --json          # machine-readable, for scripts and CI
python -m tinyllm env --check         # exit 1 if a required package is missing
python -m tinyllm env --export        # shell lines that carry the settings out
python -m tinyllm env --device cuda   # resolve settings for a specific backend
python -m tinyllm env --save env.json # keep the report next to your run
```

It prints the machine spec, every requirement parsed from `requirements*.txt`
with its installed version and whether that version satisfies the pin, the
resolved settings, and the exact next command to run.

The same scan is a library, so scripts and notebooks resolve settings instead of
guessing:

```python
from tinyllm.core.environment import bootstrap

report, settings = bootstrap()   # scan, then apply (threads + env vars)
print(report.to_text())         # or report.to_markdown() / report.to_json()
print(settings.device, settings.threads, settings.amp, settings.preset)
```

`apply_settings()` exports `TINYLLM_DEVICE`, `TINYLLM_THREADS`, `TINYLLM_PRESET`,
`TINYLLM_AMP`, `TINYLLM_CHECKPOINTING` and `OMP_NUM_THREADS`, so a notebook
kernel, a `serve` process and a second terminal all inherit the same
configuration instead of re-deriving it. `TINYLLM_DATASET` selects the corpus.

---

## 🚀 Quick start

```bash
git clone https://github.com/Gethubsathvik/TinyLLM.git
cd TinyLLM

python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

python -m tinyllm setup --install            # installs torch for YOUR hardware
python -m tinyllm env                        # scan this machine, apply its settings
python -m tinyllm doctor                     # verify the environment

python -m tinyllm data prepare --source shakespeare    # ~1 MB, ~4 s
python -m tinyllm train --dataset shakespeare --preset cpu-tiny
python -m tinyllm generate --prompt "ROMEO:" --max-new-tokens 200

python -m tinyllm lab --install              # Jupyter Lab, kernel preconfigured
```

The full-size route on a machine with room for it:

```bash
python -m tinyllm data prepare              # Wikipedia, ~15 min, ~1.3 GB
python -m tinyllm train --preset auto       # train, sized to this machine
python -m tinyllm gui                       # desktop GUI
```

All commands share one entry point:

```bash
python -m tinyllm            # list every command
python -m tinyllm train --help
```

---

## 🪟 Windows

<details open>
<summary><b>PowerShell</b></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m tinyllm setup --install
python -m tinyllm env
python -m tinyllm data prepare --source shakespeare
python -m tinyllm train --dataset shakespeare
python -m tinyllm gui
```

</details>

**If `Activate.ps1` is blocked** by the execution policy, use the interpreter directly — no activation needed:

```powershell
.\.venv\Scripts\python.exe -m tinyllm env
.\.venv\Scripts\python.exe -m tinyllm gui
```

Or unblock it once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Keep the corpus off the system drive.** By default data lands in `.\data`. If `C:` is tight, point it elsewhere:

```powershell
$env:TINYLLM_DATA_DIR = "C:\tinyllm_data"
```

<details>
<summary><b>GPU notes for Windows</b></summary>

`setup` detects `nvidia-smi` and installs the matching CUDA wheel automatically.
On a machine with only an AMD or Intel iGPU you will correctly get `cpu`,
because neither has a CUDA path in PyTorch. To try DirectML:

```powershell
python -m tinyllm setup --backend dml --install
python -m tinyllm env                      # dml available: True
python -m tinyllm train --device dml --preset smoke
```

</details>

---

## 🐧 Linux

```bash
sudo apt install python3-venv python3-pip     # Debian/Ubuntu
# Fedora: sudo dnf install python3 python3-pip

python3 -m venv .venv
source .venv/bin/activate

python -m tinyllm setup --install              # picks cu124 / rocm / cpu for you
python -m tinyllm env
python -m tinyllm data prepare --source shakespeare
python -m tinyllm train --dataset shakespeare
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
python -m tinyllm setup --backend cuda --cuda 124 --install

# AMD ROCm
python -m tinyllm setup --backend rocm --rocm 6.2 --install

# Intel
python -m tinyllm setup --backend xpu --install
```

For ROCm, PyTorch needs the matching `rocm-smi` libraries present; verify with
`rocm-smi` before trusting the detection. `python -m tinyllm env` then shows
`rocm build:` instead of a CUDA build.

</details>

---

## 🍎 macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m tinyllm setup --install
# Tk is bundled, but you may need:
xcode-select --install

python -m tinyllm env
python -m tinyllm data prepare --source shakespeare
python -m tinyllm train --dataset shakespeare
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
notebook-only model logic. `python -m tinyllm lab` installs the notebook
requirements, registers a kernel that starts with the device, thread count and
preset the scan resolved, validates the shipped notebooks, and starts the server.

```bash
python -m tinyllm lab --install     # jupyterlab + ipykernel (requirements-notebook.txt)
python -m tinyllm lab               # register the kernel and open Lab
python -m tinyllm lab --register-only   # write the kernel spec, don't launch
python -m tinyllm lab --check           # exit 1 if Lab isn't usable
python -m tinyllm lab --list            # notebooks, kernel path, kernel env
python -m tinyllm lab --no-browser --port 8890 --ip 0.0.0.0
```

The registered kernel spec carries the resolved environment:

```json
{
  "argv": [".../python.exe", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "TinyLLM (scanned environment)",
  "name": "tinyllm",
  "env": {"TINYLLM_DEVICE": "cpu", "OMP_NUM_THREADS": "8", "TINYLLM_PRESET": "cpu-tiny", ...}
}
```

### Notebooks

| Notebook | What it does |
|----------|--------------|
| `00_environment.ipynb` | The scan itself: machine spec, requirements, resolved settings, what each backend would do |
| `01_train.ipynb` | `Config` + `train()` — the same call the CLI makes — then the loss table and curves |
| `02_inference.ipynb` | `InferenceEngine`: one-shot generation, KV-cache comparison, temperature sweep |

They are generated from `tinyllm/tools/notebook_sources.py` so the cells stay
readable Python instead of escaped JSON:

```bash
python -m tinyllm.tools.notebook_sources      # regenerate notebooks/
```

`python -m tinyllm lab` refuses to be quiet about damage: it validates every
notebook (JSON, nbformat 4, kernel name, every code cell parses, no stored
outputs) and reports problems before launching.

Notebook deps live in [`requirements-notebook.txt`](requirements-notebook.txt);
nothing else in the project imports them.

---

## 📚 Corpora

`python -m tinyllm data sources` lists everything selectable. Every source ends
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
python -m tinyllm data sources                              # what is available
python -m tinyllm data prepare --source shakespeare         # tiny Shakespeare
python -m tinyllm data prepare --source wikitext --target-mb 50
python -m tinyllm data prepare --source tinystories --target-mb 500
python -m tinyllm data prepare --source hf:roneneldan/TinyStories
python -m tinyllm data prepare --source kaggle:user/dataset-slug
python -m tinyllm data prepare --source local:my_notes.txt

python -m tinyllm train --dataset shakespeare              # train on it
```

Each corpus gets its own directory (`data/raw/<key>/`, `data/binary/<key>/`), so
several can be built side by side. `TINYLLM_DATASET=shakespeare` sets the default
for every entry point.

**Hugging Face datasets are streamed, not downloaded.** `load_dataset(...,
streaming=True)` is cut off at `--target-mb`, so a 12 GB corpus is usable on a
laptop with 8 GB of RAM, and the raw text is cached so a second run tokenizes
offline.

**Tokenization** is GPT-2 byte-level BPE, vocab **50257**, round-trip verified
exact, written as flat `uint16` streams and split 80/20:

```
[tokenize] 338,025 tokens (0.34M), dtype=uint16
[split]    80/20 -> train 270,420 tok | val 67,605 tok
```

Useful flags: `--target-mb`, `--num-shards`, `--train-split`, `--keep-parquet`,
`--force` (re-tokenize from the cached text), `--redownload` (refetch), and
`--no-report`.

### Training scripts and models from Hugging Face

TinyLLM trains its own GPT rather than loading one, but the Hub is still the
place to go for text and for reference implementations:

- **Datasets** — `python -m tinyllm data prepare --source hf:<repo_id>` works with
  any Hub dataset that exposes a text column; `roneneldan/TinyStories` and
  `Salesforce/wikitext` are pre-registered above.
- **Reference training scripts** — `huggingface/transformers` examples
  (`examples/pytorch/language-modeling/run_clm.py`) and `huggingface/trl`'s
  `SFTTrainer` are the equivalents of `features/training/service.py` for
  pretrained models. This repo deliberately has no dependency on either.
- **Tokenizers** — `load_tokenizer()` uses the Hub's `gpt2` copy, caches it under
  `data/tokenizer/`, and verifies a local copy round-trips before trusting it.

---

## 🎭 Tiny Shakespeare in 5 minutes

The fastest way to see next-token prediction actually work, on any machine, with
no GPU:

```bash
python -m tinyllm data prepare --source shakespeare     # 1.1 MB -> 338,025 tokens
python -m tinyllm train --dataset shakespeare \
    --preset cpu-tiny --max-iters 400 --run-name gpt-shakespeare
python -m tinyllm generate --checkpoint models/runs/gpt-shakespeare/checkpoint.pt \
    --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
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

Sample output from the `cpu-tiny` (30.0M) model after 400 steps on this Ryzen —
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
tinyllm/
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
making it better. So TinyLLM ships exactly one deployable — the inference API
— and keeps every other boundary a clean in-process service interface.

---

## 🤖 Model

```bash
python -m tinyllm train --preset full --summary-only
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

## 🎓 Training

```bash
python -m tinyllm train --preset cpu-tiny --max-iters 600
python -m tinyllm train --show-history gpt-shakespeare
python -m tinyllm train --dataset shakespeare --resume models/runs/gpt-shakespeare/checkpoint.pt --max-iters 800
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
python -m tinyllm generate --prompt "The history of the city is"
python -m tinyllm generate --prompt "Once upon a time" \
    --temperature 0.7 --top-k 40 --top-p 0.95 --max-new-tokens 300
python -m tinyllm generate --interactive
```

Interactive commands: `:t <temp>`, `:k <top-k>`, `:p <top-p>`, `:n <tokens>`,
`:seed <n>`, `:quit`.

All four front ends share `models/sampling.py`, so they behave identically.

---

## 🖥️ GUI

```bash
python -m tinyllm gui
```

A dark-themed Tk window: prompt box, **real-time token-by-token streaming** on
a background thread (the UI never freezes), live sliders for temperature,
top-k, top-p, max tokens, seed and repetition penalty, a KV-cache toggle, a
**Stop** button mid-generation, and a tok/s readout.

> The 30M model produces word-like but incoherent text. That is the model, not
> the GUI — lower the temperature to `0.1` and output becomes repetitive, which
> confirms sampling is wired up correctly.

---

## 🌐 HTTP API

Optional, and the one part of TinyLLM that runs as its own process.

```bash
pip install -r requirements-api.txt
python -m tinyllm serve --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | liveness + model metadata |
| `/generate` | POST | one-shot generation, JSON in / JSON out |
| `/stream` | GET | server-sent events, one `data:` line per token |
| `/v1/completions` | POST | OpenAI-compatible alias for third-party clients |
| `/docs` | GET | interactive OpenAPI docs |

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The history of the city is","max_new_tokens":40,"temperature":0.8}'

curl -N "http://localhost:8000/stream?prompt=hello&max_new_tokens=20"
```

```
data: {"type": "meta", "parameters_m": 30.0, ...}
data: {"type": "token", "token": " List"}
data: {"type": "token", "token": " Faction"}
data: {"type": "done"}
```

The OpenAI-compatible alias returns the standard envelope, so clients written
against that API work unmodified:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllm","prompt":"hello","max_tokens":20}'
```

```json
{
  "id": "cmpl-1759000000",
  "object": "text_completion",
  "choices": [{"index": 0, "text": "...", "logprobs": null, "finish_reason": "length"}],
  "usage": {"prompt_tokens": 2, "completion_tokens": 20, "total_tokens": 22},
  "tinyllm": {"elapsed_s": 0.31, "device": "cuda:0", "...": "..."}
}
```

> `stream: true` is refused there with a `400` pointing at `/stream`: OpenAI
> streams token objects, TinyLLM streams bare text, and faking that shape would
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
python -m tinyllm setup --install          # Termux has no GPU, so this picks cpu
python -m tinyllm env
python -m tinyllm train --dataset shakespeare --preset smoke
```

Expect CPU-only training on a phone to be roughly **20-50x slower** than a laptop — and phone SoCs are usually ARM with far less memory bandwidth. Practical for the CLI, the test suite and inference; not for real training runs. tiny Shakespeare is the corpus to use here: 1.1 MB instead of 1.3 GB.

**2. A client for the HTTP API** — the supported route

Run `serve` on your PC, bind it to the LAN, and call it from any phone browser or app:

```bash
python -m tinyllm serve --host 0.0.0.0 --port 8000
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
  -d '{"model":"tinyllm","prompt":"hello","max_tokens":20}'
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

| File | Covers |
|------|--------|
| `tests/test_model.py` | architecture, causality, KV cache, sampling, LR schedule |
| `tests/test_data_and_inference.py` | tokenize/split/batch, engine streaming |
| `tests/test_data_sources.py` | corpus registry, URL/local/HF/Kaggle fetch, conversions, per-corpus paths |
| `tests/test_device.py` | backend probes, precision policy, presets, wheel indexes |
| `tests/test_environment.py` | the scan: spec, requirements, settings, shell export, CLI |
| `tests/test_lab.py` | kernel spec, notebook integrity, Lab CLI, a notebook executed in a real kernel |
| `tests/test_package.py` | every module imports, CLI wiring, layout |
| `tests/test_gui.py` | real Tk window, streaming, Stop button |
| `tests/test_api.py` | HTTP endpoints, SSE, validation, OpenAPI |

The tests that matter most are the ones that catch **silent** bugs: that a
causal mask leaks no future tokens, that cached decoding matches a full forward
pass, and that gradient checkpointing is bit-identical to the plain path.

The GUI, API and Jupyter suites skip themselves automatically when Tk, FastAPI
or jupyterlab is unavailable, so the suite passes headless and without the
notebook extras.

---

## 📁 Project layout

```
TinyLLM/
├── tinyllm/
│   ├── __main__.py            # single command dispatcher
│   ├── core/                   # config, paths, device, environment, seeding
│   ├── models/                 # M: GPT architecture, builder, sampling
│   ├── features/
│   │   ├── data/               # service.py + cli.py + sources.py
│   │   ├── training/           # service.py + cli.py
│   │   └── inference/          # service.py + cli.py + gui.py
│   ├── api/server.py           # optional FastAPI service
│   └── tools/                  # setup, env, doctor, lab, notebook_sources
├── notebooks/                  # generated .ipynb: environment, train, inference
├── tests/
├── models/runs/<name>/         # checkpoint.pt, history.json, loss_curves.png
├── data/raw/<corpus>/          # cached corpus text
├── data/binary/<corpus>/       # train.bin, val.bin (uint16)
├── requirements.txt
├── requirements-api.txt
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
| `python -m tinyllm data prepare` — documented in the README but not implemented | every corpus command in the docs errored out |
| ROCm index built as `cu124rocm`, XPU index missing the `download.` host | both wheel installs 404'd |
| Assigning `DataConfig.dataset` left `binary_dir` alone | trained on the **previous** corpus's tokens |
| `DataConfig` `raw_dir`/`binary_dir` could not be overridden per corpus | every corpus shared one directory |

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🔗 Links

- **Repository** — https://github.com/Gethubsathvik/TinyLLM
- **Dataset (default)** — [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.en`
- **Corpora** — [Tiny Shakespeare](https://github.com/karpathy/char-rnn) · [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) · [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) · [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)
- **Tokenizer** — GPT-2 BPE, vocab 50257
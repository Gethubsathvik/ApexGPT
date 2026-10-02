# TinyLLM

A GPT-style transformer language model built from scratch in PyTorch, with a
data pipeline, training loop, CLI inference, and a Tkinter desktop GUI.

---

## 1. Environment

### Verified hardware on this machine

| Item | Value |
|------|-------|
| CPU | **AMD Ryzen 3 7320U**, 4 cores / 8 threads |
| GPU | **AMD Radeon integrated graphics** (no NVIDIA GPU) |
| CUDA | **Not available** — `nvidia-smi` and `nvcc` are absent |
| Python | 3.14.7 (64-bit) |
| PyTorch | 2.14.1+cpu |

> **Two deviations from the original request, both forced by the hardware:**
>
> 1. **CUDA cannot be used.** PyTorch CUDA kernels require an NVIDIA GPU and
>    driver. This machine has only an AMD iGPU, so `torch.cuda.is_available()`
>    is `False` and training runs on CPU. Every script still auto-selects CUDA
>    when present, so the same code trains on an NVIDIA box unchanged.
> 2. **The CPU is an AMD Ryzen 3, not an Intel i5.** The 8 logical threads are
>    configured as `torch.set_num_threads(8)`.

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify the environment (versions, CUDA detection, autocast, gradient
checkpointing, tokenizer):

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

### Data location

The project ships with `data/`, but bulk corpus files are large. If the
project drive is nearly full, point the data root somewhere else:

```powershell
$env:TINYLLM_DATA_DIR = "C:\tinyllm_data"
```

`config.py` reads this variable; when unset it defaults to `./data`.

### Project structure

```
TinyLLM/
├── config.py                 # central config dataclasses + data root
├── model.py                  # GPT model, KV cache, model_summary()
├── requirements.txt
├── data/                     # default data root (see TINYLLM_DATA_DIR)
│   ├── raw/wiki_corpus.txt   # extracted plain text
│   ├── binary/train.bin      # uint16 token stream (80%)
│   ├── binary/val.bin        # uint16 token stream (20%)
│   └── tokenizer/            # GPT-2 BPE tokenizer
├── models/
│   ├── checkpoints/
│   └── runs/<name>/          # checkpoint.pt, history.json, loss_curves.png
├── scripts/
    ├── check_env.py          # environment + CUDA verification
    ├── prepare_data.py       # download, tokenize, split, sample report
    ├── smoke_test.py         # 21 fast correctness checks
    ├── test_gui.py           # 13 headless GUI checks
    ├── show_history.py       # print a run's loss table
    ├── train.py              # training loop
    ├── generate.py           # CLI text generation
    └── gui.py                # Tkinter desktop GUI
```

---

## 2. Dataset

```powershell
$env:TINYLLM_DATA_DIR = "C:\tinyllm_data"
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

* **Source**: `wikimedia/wikipedia` `20231101.en`, 2 parquet shards ≈ 772 MB raw
  (in the requested 500 MB–1 GB range), yielding **1.28 B characters**.
* **Tokenizer**: GPT-2 byte-level BPE, vocabulary **50257**, verified
  round-trip exact (`round-trip ok: True`).
* **Output**: flat `uint16` token streams — **242,087,052 train** and
  **60,521,763 validation** tokens (80/20 split).
* **Batching**: random `block_size` windows; `y` is `x` shifted left by one for
  next-token prediction.
* Parquet shards are deleted after extraction (`--keep-parquet` to retain).

Sample verification output:

```
RAW  : 'Anarchism is a political philosophy and movement that is skeptical of all...'
TOKENS: ['An', 'arch', 'ism', ' is', ' a', ' political', ' philosophy', ' and', ...]
round-trip ok: True

x = [321, 461, 11, 284, 6594, 683, 290, 4405, 262, 21752, 338, 4427, 25, 284, 3638, 257]
y = [461, 11, 284, 6594, 683, 290, 4405, 262, 21752, 338, 4427, 25, 284, 3638, 257, 41927]
y is x shifted left by 1: True
```

Useful flags: `--target-mb`, `--num-shards`, `--keep-parquet`, `--force`
(re-tokenize), `--redownload` (re-fetch the corpus).

---

## 3. Model

`model.py` implements the transformer from scratch: token/position embeddings,
causal multi-head attention (packed QKV, `scaled_dot_product_attention`), GELU
MLP, pre-LayerNorm residual blocks, final LayerNorm, and a weight-tied LM head.

```powershell
.\.venv\Scripts\python.exe model.py
.\.venv\Scripts\python.exe scripts\train.py --preset full --summary-only
```

### Requested configuration — 12 layers / 768 hidden / 12 heads

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
fp32 size on disk  : 0.50 GB
```

123.8M sits inside the requested 100–200M range.

Per-component breakdown:

```
wte (token emb)      38,597,376  (38.60M)
wpe (pos emb)           196,608  ( 0.20M)
attention            28,348,416  (28.35M)
mlp                  56,669,184  (56.67M)
layernorms               38,400  ( 0.04M)
per transformer block  7,087,872  ( 7.09M)
```

### Memory features

* **Mixed precision** — `torch.autocast` with fp16 + `GradScaler` on CUDA,
  bfloat16 on CPU.
* **Gradient checkpointing** — per-block, via
  `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Verified
  bit-identical to the un-checked forward pass.
* **Weight tying** — the LM head reuses the token embedding, saving ~38M params.

> **Both are enabled automatically on CUDA and disabled automatically on CPU.**
> On this machine they are a *66x penalty*, not a win. Measured with the
> `cpu-tiny` config (batch 4x192):
>
> | Setting | Time/iter | Throughput |
> |---------|-----------|-----------|
> | bf16 autocast + gradient checkpointing | 195.1 s | 4 tok/s |
> | plain fp32, no checkpointing | **2.95 s** | **260 tok/s** |
>
> The Ryzen 3 7320U is Zen2, which has no native bf16 support, so autocast
> emulates it; and checkpointing's recompute blocks the fused `SDPA` path.
> Both are standard wins on an NVIDIA GPU, where the code uses them by
> default. Force either on with `--amp` / `--checkpointing`.

### Presets

| Preset | Layers | d_model | Heads | Params | CPU speed | 4000 iters |
|--------|--------|---------|-------|--------|-----------|-----------|
| `full` | 12 | 768 | 12 | 123.8M | 55.5 s/it | **~62 h** |
| `medium` | 8 | 512 | 8 | ~50M | — | — |
| `cpu-tiny` | 6 | 384 | 6 | 30.0M | 3.0 s/it | 3.3 h |
| `smoke` | 2 | 128 | 4 | 6.8M | — | seconds |

Measured on this Ryzen 3 with batch 4x192 in fp32. Earlier numbers quoted
gradient checkpointing plus bf16 autocast, which turned out to be a 66x
penalty on this CPU (see the memory section below).

---

## 4. Training

```powershell
$env:TINYLLM_DATA_DIR = "C:\tinyllm_data"
.\.venv\Scripts\python.exe scripts\train.py --preset cpu-tiny --max-iters 600
```

The loop: forward pass → cross-entropy next-token loss → backward → gradient
clipping → AdamW step, with linear warmup and cosine decay, periodic validation,
checkpointing every 100 steps, tqdm progress, and matplotlib loss curves.

Optimizations applied: fused `scaled_dot_product_attention` with a packed QKV
projection, weight tying, `torch.set_num_threads(8)`, memory-mapped `uint16`
binaries, and a warmup/cosine LR schedule. Mixed precision and gradient
checkpointing are enabled on CUDA and disabled on CPU (see above).

Flags: `--max-iters`, `--epochs`, `--batch-size`, `--block-size`,
`--learning-rate`, `--eval-interval`, `--eval-iters`, `--checkpoint-interval`,
`--amp` / `--no-amp`, `--checkpointing` / `--no-checkpointing`,
`--resume <ckpt>`, `--summary-only`.

Outputs land in `models/runs/<run-name>/`: `checkpoint.pt`, `history.json`,
`loss_curves.png`.

### Honest note on "2-3 epochs"

The corpus is **242M training tokens**. One epoch at 768 tokens per step is
about **325,000 iterations** — roughly 15 days at the measured 4 s/iteration.
Training is therefore **step-budgeted, not epoch-budgeted**. On this CPU a
meaningful run is 600–2000 steps (tens of minutes to a few hours), which sees
well under 1% of the corpus. A model at that budget learns English word
structure and syntax but produces incoherent text. Genuine multi-epoch training
of the 123.8M `full` config requires an NVIDIA GPU.

### Completed run

`cpu-tiny` (6L / 384d / 6H, 30.0M params), batch 4x192, 600 steps, fp32:

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
elapsed: 44.3 min   (~220 tok/s)
```

Validation loss fell from **10.82** (ln 50257, i.e. random guessing) to **7.04**.
Artifacts in `models/runs/gpt-cpu-tiny/`: `checkpoint.pt` (344 MB),
`history.json`, `loss_curves.png`.

Sample output at this budget (T=0.8, top-k 40, top-p 0.95):

```
The history of the city is the the the a the the in the the the the, to the the
a and the the the. the for the. ...
```

This is the expected result for 30M parameters trained on 460k tokens (~0.19%
of one epoch). It has learned English word and sentence structure but not
coherence. Sample output from the GUI at a 12-token budget:

```
The capital city of
 the first of the new time of the United.
```

### Correctness checks

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py    # 21/21
.\.venv\Scripts\python.exe scripts\test_gui.py      # 13/13
```

`smoke_test.py` covers: parameter count, weight tying, forward/loss shapes, loss
≈ ln(vocab) at init, causal masking with no future leakage, backward + AdamW,
loss decrease, warmup/cosine LR schedule, gradient-checkpointing transparency,
validation loss, checkpoint save/load, generation shape, seeded reproducibility,
low-temperature determinism, KV-cache growth, cached-vs-full-forward agreement,
generation past the context window, KV-cache trimming, loss-curve plotting, and
GUI importability.

`test_gui.py` builds the real Tk window, streams a generation through the
worker thread, and checks the sliders, prompt, Stop button, clear button, and
status readout.

---

## 5. Inference and GUI

### CLI

```powershell
.\.venv\Scripts\python.exe scripts\generate.py --prompt "The history of the city is"
.\.venv\Scripts\python.exe scripts\generate.py --prompt "Once upon a time" `
    --temperature 0.7 --top-k 40 --top-p 0.95 --max-new-tokens 300
.\.venv\Scripts\python.exe scripts\generate.py --interactive
```

Interactive commands: `:t <temp>`, `:k <k>`, `:p <p>`, `:n <tokens>`,
`:seed <n>`, `:quit`.

### GUI

```powershell
.\.venv\Scripts\python.exe scripts\gui.py
```

A Tkinter window with a prompt box, real-time token-by-token streaming output
on a background thread (the UI never freezes), and live sliders for
temperature, top-k, top-p, max tokens, seed, and repetition penalty, plus a KV
cache toggle, Stop button, and a tok/s status readout.

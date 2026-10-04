# 💻 Hardware portability

Every backend ApexGPT probes for, what it costs, and what happens when one is absent.


# 💻 Hardware portability

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


---

Back to [the README](../README.md).


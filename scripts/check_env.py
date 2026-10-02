"""Verify the environment: versions, device, CUDA detection, CPU config.

Run:  python scripts/check_env.py
"""
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm


def cpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return platform.processor() or "unknown"


def gpu_names() -> list:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def main() -> None:
    print("=" * 66)
    print("TinyLLM environment check")
    print("=" * 66)
    print(f"Python        : {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"Platform      : {platform.platform()}")
    print(f"CPU           : {cpu_name()}")
    print(f"Logical cores : {os.cpu_count()}")

    print("-" * 66)
    print("Packages")
    for name in ("torch", "transformers", "datasets", "tokenizers",
                 "numpy", "matplotlib", "tqdm"):
        mod = __import__(name)
        print(f"  {name:<13} {getattr(mod, '__version__', 'unknown')}")

    print("-" * 66)
    print("CUDA detection")
    print(f"  torch.version.cuda built with : {torch.version.cuda}")
    print(f"  torch.cuda.is_available()      : {torch.cuda.is_available()}")
    print(f"  cuda.device_count()            : {torch.cuda.device_count()}")
    print(f"  backends.cudnn available       : {torch.backends.cudnn.is_available()}")
    print(f"  detected display adapters      : {gpu_names() or 'none'}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i} -> {props.name}, {props.total_memory / 1e9:.1f} GB, "
                  f"cc {props.major}.{props.minor}")
    else:
        print("  NOTE: No CUDA device. This machine has no NVIDIA GPU,")
        print("        so PyTorch CUDA kernels are unavailable. Training")
        print("        will run on CPU. All model/training code still")
        print("        auto-selects CUDA when it becomes present.")

    print("-" * 66)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"torch threads  : {torch.get_num_threads()} / interop {torch.get_num_interop_threads()}")
    print(f"selected device: {device}")

    print("-" * 66)
    print("Smoke tests")
    x = torch.randn(4, 4, device=device)
    y = (x @ x.transpose(-2, -1)).sum()
    print(f"  tensor matmul on {device}: ok (shape {tuple(x.shape)}, finite {torch.isfinite(y).item()})")

    # AMP availability
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        _ = (x @ x.transpose(-2, -1)).sum()
    print(f"  autocast {amp_dtype} ({device.type}) : ok")

    # Gradient checkpointing API
    from torch.utils.checkpoint import checkpoint
    lin = torch.nn.Linear(4, 4)
    ck = checkpoint(lin, x, use_reentrant=False)
    print(f"  gradient checkpointing: ok (shape {tuple(ck.shape)})")

    # BPE tokenizer
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        ids = tok("Hello, world!").input_ids
        print(f"  GPT-2 tokenizer: ok (vocab {len(tok)}, 'Hello, world!' -> {ids})")
    except Exception as e:
        print(f"  GPT-2 tokenizer: FAILED ({e})")

    print("-" * 66)
    print(f"numpy {np.__version__} sanity: {np.arange(5).sum()}")
    with tqdm(total=3, desc="tqdm", leave=False) as bar:
        for _ in range(3):
            bar.update(1)
    print("=" * 66)
    print("RESULT: environment is usable.")


if __name__ == "__main__":
    main()
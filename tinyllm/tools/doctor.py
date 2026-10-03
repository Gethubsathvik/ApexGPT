"""Environment and hardware verification."""
from __future__ import annotations

import argparse
import os
import platform
import sys

from ..core.environment import cpu_name
from ..core.paths import describe as describe_paths


def display_adapters() -> list[str]:
    """Best-effort graphics adapter list on any platform."""
    from ..core.device import display_adapters as _detect
    return _detect()


PACKAGES = ("torch", "transformers", "datasets", "tokenizers",
            "numpy", "matplotlib", "tqdm")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tinyllm doctor",
                                 description="Verify the TinyLLM environment")
    ap.parse_args(argv)

    print("=" * 66)
    print("TinyLLM environment check")
    print("=" * 66)
    print(f"Python        : {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"Platform      : {platform.platform()}")
    print(f"Machine       : {platform.machine()}")
    print(f"CPU           : {cpu_name()}")
    print(f"Logical cores : {os.cpu_count()}")

    print("-" * 66)
    print("Packages")
    ok = True
    for name in PACKAGES:
        try:
            mod = __import__(name)
            print(f"  {name:<13} {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            print(f"  {name:<13} MISSING")
            ok = False

    print("-" * 66)
    print("Accelerator detection")
    import torch
    from ..core.device import (available_backends, cuda_build_version,
                               dml_available, mps_available, rocm_build,
                               total_ram_gb, xpu_available)

    print(f"  cuda available : {torch.cuda.is_available()}")
    print(f"  cuda build     : {cuda_build_version()}")
    print(f"  rocm build     : {rocm_build() or '-'}")
    print(f"  mps available  : {mps_available()}")
    print(f"  xpu available  : {xpu_available()}")
    print(f"  dml available  : {dml_available()} (opt-in: --device dml)")
    print(f"  usable backends: {available_backends() or ['cpu']}")
    print(f"  display adapters: {display_adapters() or 'none'}")
    print(f"  system RAM      : {total_ram_gb():.1f} GB")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i} -> {props.name}, "
                  f"{props.total_memory / 1e9:.1f} GB, cc {props.major}.{props.minor}")

    print("-" * 66)
    from ..core.device import configure_threads, get_device
    device = get_device()
    print(f"torch threads   : {configure_threads()}")
    print(f"selected device : {device}")

    from ..core.device import profile
    hw = profile()
    print(f"profile         : {hw.describe()}")
    print(f"suggested preset: {hw.recommend_preset()}")

    print("-" * 66)
    print("Smoke tests")
    x = torch.randn(4, 4, device=device)
    y = (x @ x.transpose(-2, -1)).sum()
    print(f"  tensor matmul on {device}: ok "
          f"(shape {tuple(x.shape)}, finite {torch.isfinite(y).item()})")

    from ..core.device import autocast_dtype, use_amp_by_default
    dtype = autocast_dtype(device)
    try:
        with torch.autocast(device_type=device.type, dtype=dtype):
            _ = (x @ x.transpose(-2, -1)).sum()
        print(f"  autocast {dtype} on {device.type}: ok")
    except Exception as exc:
        print(f"  autocast {dtype} on {device.type}: unavailable ({exc})")
    print(f"  amp recommended  : {use_amp_by_default(device)}")

    from torch.utils.checkpoint import checkpoint
    layer = torch.nn.Linear(4, 4)
    print(f"  gradient checkpointing: ok "
          f"(shape {tuple(checkpoint(layer, x, use_reentrant=False).shape)})")

    try:
        from ..features.data.service import load_tokenizer
        tok = load_tokenizer()
        ids = tok("Hello, world!", add_special_tokens=False)["input_ids"]
        print(f"  GPT-2 tokenizer: ok (vocab {tok.vocab_size}, ids {ids[:6]}...)")
    except Exception as exc:
        print(f"  GPT-2 tokenizer: FAILED ({exc})")
        ok = False

    print("-" * 66)
    print(describe_paths())

    print("=" * 66)
    print("RESULT:", "environment is usable" if ok else "environment has problems")
    print("Full scan (requirements, settings, next steps): python -m tinyllm env")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""Model construction and reporting."""
from __future__ import annotations

import torch

from ..core.config import ModelConfig
from .gpt import GPT


def build_model(cfg: ModelConfig) -> GPT:
    """Instantiate a GPT from a validated :class:`ModelConfig`."""
    cfg.validate()
    return GPT(cfg)


def count_parameters(model: GPT) -> dict:
    total = model.num_parameters()
    non_emb = model.num_parameters(non_embedding=True)
    return {
        "total": total,
        "embeddings": total - non_emb,
        "transformer": non_emb,
    }


def estimate_flops(cfg: ModelConfig) -> dict:
    """Analytic forward-pass FLOPs; matmuls dominate."""
    per_layer = 12 * cfg.n_embd ** 2 + 2 * cfg.block_size * cfg.n_embd
    per_token = 2 * cfg.n_layer * per_layer + 2 * cfg.vocab_size * cfg.n_embd
    return {
        "per_token": per_token,
        "per_sequence": per_token * cfg.block_size,
    }


def summary_lines(model: GPT) -> list[str]:
    cfg = model.cfg
    counts = count_parameters(model)
    flops = estimate_flops(cfg)
    return [
        "=" * 68,
        "ApexGPT - GPT model summary",
        "=" * 68,
        f"  layers (n_layer)   : {cfg.n_layer}",
        f"  hidden size        : {cfg.n_embd}",
        f"  attention heads    : {cfg.n_head}  (head_dim={cfg.n_embd // cfg.n_head})",
        f"  context/block size : {cfg.block_size}",
        f"  vocab size         : {cfg.vocab_size}",
        f"  dropout            : {cfg.dropout}",
        f"  weight tying       : yes (lm_head shares wte)",
        "-" * 68,
        f"  total parameters   : {counts['total']:,}  ({counts['total'] / 1e6:.1f}M)",
        f"  embedding params   : {counts['embeddings']:,}  ({counts['embeddings'] / 1e6:.1f}M)",
        f"  transformer params : {counts['transformer']:,}  ({counts['transformer'] / 1e6:.1f}M)",
        f"  fp32 size on disk  : {counts['total'] * 4 / 1e9:.2f} GB",
        "-" * 68,
        f"  fwd FLOPs / token  : {flops['per_token'] / 1e6:.1f}M",
        f"  fwd FLOPs / seq    : {flops['per_sequence'] / 1e9:.1f}G",
        "=" * 68,
    ]


def print_summary(model: GPT) -> None:
    print("\n".join(summary_lines(model)))

    breakdown = model.parameter_breakdown()
    groups = {
        "wte (token emb)": lambda k: k.startswith("transformer.wte"),
        "wpe (pos emb)": lambda k: k.startswith("transformer.wpe"),
        "attention": lambda k: ".attn." in k,
        "mlp": lambda k: ".mlp." in k,
        "layernorms": lambda k: ".ln_" in k,
    }
    print("\nPer-component parameter breakdown:")
    for name, match in groups.items():
        keys = [k for k in breakdown if match(k)]
        if not keys:
            continue
        sub = sum(breakdown[k] for k in keys)
        print(f"  {name:<18} {sub:>12,}  ({sub / 1e6:6.2f}M)  [{len(keys)} tensors]")

    first = sum(v for k, v in breakdown.items() if k.startswith("transformer.h.0."))
    print(f"  {'per transformer bk':<18} {first:>12,}  ({first / 1e6:6.2f}M)")

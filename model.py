"""GPT-style transformer language model, implemented from scratch.

Components
----------
* Token + position embeddings
* Causal self-attention (multi-head, masked) with optional KV cache
* MLP / feed-forward block (GELU, 4x expansion)
* Pre-LayerNorm residual blocks
* Final LayerNorm + tied LM head for next-token prediction

Memory features
---------------
* Gradient checkpointing per transformer block
* Mixed precision via torch.autocast (fp16 on CUDA, bf16 on CPU)
* Weight-tied output head (saves ~38M params at 768 dim / 50257 vocab)
"""
from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig


# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional inference KV cache."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        # all three projections packed into one matrix for speed
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # B,H,T,D
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        Tq = T   # query length stays T; k/v may be longer with a cache
        prefix = 0
        if kv_cache is not None:
            if kv_cache["k"] is not None:
                k = torch.cat((kv_cache["k"], k), dim=2)
                v = torch.cat((kv_cache["v"], v), dim=2)
            # keep the cache within the position-embedding window
            if k.size(2) > self.causal_mask.size(-1):
                k = k[:, :, -self.causal_mask.size(-1):]
                v = v[:, :, -self.causal_mask.size(-1):]
            kv_cache["k"], kv_cache["v"] = k, v
            prefix = k.size(2) - Tq

        if prefix == 0:
            # fresh forward over the whole window: top-left causal mask
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # cached continuation: query j (absolute position prefix+j) may only
            # attend to keys 0..prefix+j, so the mask is bottom-right aligned
            L = k.size(2)
            allow = torch.ones(Tq, L, dtype=torch.bool, device=q.device).tril(
                diagonal=prefix)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=allow.view(1, 1, Tq, L),
                dropout_p=self.dropout if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.resid_dropout(self.c_proj(y))


# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Feed-forward network: n_embd -> 4*n_embd -> n_embd."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# --------------------------------------------------------------------------- #
class Block(nn.Module):
    """Pre-LayerNorm transformer block: attention then MLP, both residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


# --------------------------------------------------------------------------- #
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
            "wpe": nn.Embedding(cfg.block_size, cfg.n_embd),
            "drop": nn.Dropout(cfg.dropout),
            "h": nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            "ln_f": nn.LayerNorm(cfg.n_embd),
        })
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight   # weight tying

        self.apply(self._init_weights)
        # scaled init on residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self.gradient_checkpointing = False

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    # -- persistence -------------------------------------------------------- #
    def save(self, path, optimizer=None, step=0, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_config": asdict(self.cfg),
            "model_state": self.state_dict(),
            "step": step,
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path, map_location="cpu", weights_only=False):
        payload = torch.load(path, map_location=map_location, weights_only=weights_only)
        model = cls(ModelConfig(**payload["model_config"]))
        model.load_state_dict(payload["model_state"])
        return model, payload

    # -- plumbing ----------------------------------------------------------- #
    def set_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable and self.training

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()
            n -= self.transformer.wpe.weight.numel()
        return n

    def parameter_breakdown(self) -> dict:
        out = {}
        for name, p in self.named_parameters():
            out[name] = p.numel()
        return out

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"sequence {T} > block size {self.cfg.block_size}"

        # with a KV cache, positions continue after the cached prefix
        offset = 0
        if kv_caches is not None and kv_caches[0]["k"] is not None:
            offset = kv_caches[0]["k"].size(2)
        offset = max(0, min(offset, self.cfg.block_size - T))   # keep pos ids in range
        pos = torch.arange(offset, offset + T, device=idx.device)
        x = self.transformer["drop"](
            self.transformer["wte"](idx) + self.transformer["wpe"](pos)
        )

        for i, block in enumerate(self.transformer["h"]):
            cache = None if kv_caches is None else kv_caches[i]
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cache, use_reentrant=False)
            else:
                x = block(x, kv_cache=cache)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, eos_id=None, kv_caches=None, generator=None):
        """Sample `max_new_tokens` continuations. Used by the CLI and GUI."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_idx_to_remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_idx_to_remove[..., 0] = False
                remove = sorted_idx_to_remove.scatter(-1, sorted_idx, sorted_idx_to_remove)
                logits = logits.masked_fill(remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return idx

    def new_kv_caches(self, batch_size, device, dtype):
        return [{"k": None, "v": None} for _ in self.transformer["h"]]


# --------------------------------------------------------------------------- #
def build_model(cfg: ModelConfig) -> GPT:
    return GPT(cfg)


def model_summary(model: GPT) -> str:
    cfg = model.cfg
    total = model.num_parameters()
    non_emb = model.num_parameters(non_embedding=True)
    emb = total - non_emb

    # analytical FLOPs per token (fwd+bwd ~= 3x fwd, matmuls dominate)
    hd = cfg.n_embd
    per_layer = 12 * cfg.n_embd ** 2 + 2 * cfg.block_size * cfg.n_embd
    flops_fwd = 2 * cfg.n_layer * per_layer + 2 * cfg.vocab_size * cfg.n_embd

    lines = [
        "=" * 68,
        "TinyLLM - GPT model summary",
        "=" * 68,
        f"  layers (n_layer)   : {cfg.n_layer}",
        f"  hidden size        : {cfg.n_embd}",
        f"  attention heads    : {cfg.n_head}  (head_dim={hd // cfg.n_head})",
        f"  context/block size : {cfg.block_size}",
        f"  vocab size         : {cfg.vocab_size}",
        f"  dropout            : {cfg.dropout}",
        f"  weight tying       : yes (lm_head shares wte)",
        "-" * 68,
        f"  total parameters   : {total:,}  ({total / 1e6:.1f}M)",
        f"  embedding params   : {emb:,}  ({emb / 1e6:.1f}M)",
        f"  transformer params : {non_emb:,}  ({non_emb / 1e6:.1f}M)",
        f"  fp32 size on disk  : {total * 4 / 1e9:.2f} GB",
        "-" * 68,
        f"  fwd FLOPs / token  : {flops_fwd / 1e6:.1f}M",
        f"  fwd FLOPs / seq    : {flops_fwd * cfg.block_size / 1e9:.1f}G",
        "=" * 68,
    ]

    print("\n".join(lines))

    breakdown = model.parameter_breakdown()
    groups = {
        "wte (token emb)": [k for k in breakdown if k.startswith("transformer.wte")],
        "wpe (pos emb)": [k for k in breakdown if k.startswith("transformer.wpe")],
        "attention": [k for k in breakdown if ".attn." in k],
        "mlp": [k for k in breakdown if ".mlp." in k],
        "layernorms": [k for k in breakdown if ".ln_" in k or k == "transformer.ln_f.weight"],
    }
    print("\nPer-component parameter breakdown:")
    for gname, keys in groups.items():
        if not keys:
            continue
        sub = sum(breakdown[k] for k in keys)
        print(f"  {gname:<18} {sub:>12,}  ({sub / 1e6:6.2f}M)  [{len(keys)} tensors]")

    per_layer = sum(v for k, v in breakdown.items() if k.startswith("transformer.h.0."))
    print(f"  {'per transformer bk':<18} {per_layer:>12,}  ({per_layer / 1e6:6.2f}M)")

    print("\n" + str(model))
    return "\n".join(lines)

if __name__ == "__main__":
    from config import Config
    cfg = Config().model
    m = build_model(cfg)
    print(model_summary(m))

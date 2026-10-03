"""The GPT architecture: causal self-attention, MLP, blocks, and the model.

The module/parameter names here are part of the checkpoint format, so they
must stay stable for saved weights to keep loading.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..core.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an optional inference KV cache.

    With no cache this is a plain top-left-causal forward pass. With a cache
    the new queries sit after a prefix, so the mask must be bottom-right
    aligned instead - otherwise later queries attend to future keys.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.max_context = cfg.block_size

        # Q, K and V share one packed projection for speed
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None) -> torch.Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # B,H,T,D
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        prefix = 0
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat((kv_cache["k"], k), dim=2)
                v = torch.cat((kv_cache["v"], v), dim=2)
            # keep the cache inside the position-embedding window
            if k.size(2) > self.max_context:
                k = k[:, :, -self.max_context:]
                v = v[:, :, -self.max_context:]
            kv_cache["k"], kv_cache["v"] = k, v
            prefix = k.size(2) - T

        if prefix == 0:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # query j has absolute position prefix+j and may attend to
            # keys 0..prefix+j
            L = k.size(2)
            allow = torch.ones(T, L, dtype=torch.bool, device=q.device).tril(
                diagonal=prefix)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allow.view(1, 1, T, L),
                dropout_p=self.dropout if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network: n_embd -> 4*n_embd -> n_embd."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: attention then MLP, both residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer trained for next-token prediction."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
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
        # GPT-2 scales residual-output projections by 1/sqrt(2 * n_layer)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)

        self._gradient_checkpointing = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    # -- memory policy ------------------------------------------------------ #
    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        """Record the policy.

        The flag is stored rather than combined with ``self.training`` at set
        time: doing that made the call silently do nothing if it happened
        before ``.train()``, which depended on call order alone.
        """
        self._gradient_checkpointing = bool(enable)

    @property
    def gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing and self.training

    # -- persistence --------------------------------------------------------- #
    def save(self, path, optimizer=None, step: int = 0, extra: dict | None = None) -> None:
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
    def load(cls, path, map_location="cpu", weights_only: bool = False):
        payload = torch.load(path, map_location=map_location, weights_only=weights_only)
        model = cls(ModelConfig(**payload["model_config"]))
        model.load_state_dict(payload["model_state"])
        return model, payload

    # -- introspection -------------------------------------------------------- #
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()
            n -= self.transformer.wpe.weight.numel()
        return n

    def parameter_breakdown(self) -> dict:
        return {name: p.numel() for name, p in self.named_parameters()}

    # -- forward --------------------------------------------------------------- #
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                kv_caches: list | None = None):
        B, T = idx.size()
        if T > self.cfg.block_size:
            raise ValueError(f"sequence {T} exceeds block size {self.cfg.block_size}")

        # with a KV cache the positions continue past the cached prefix
        offset = 0
        if kv_caches is not None and kv_caches[0]["k"] is not None:
            offset = kv_caches[0]["k"].size(2)
        offset = max(0, min(offset, self.cfg.block_size - T))
        pos = torch.arange(offset, offset + T, device=idx.device)

        x = self.transformer["drop"](
            self.transformer["wte"](idx) + self.transformer["wpe"](pos)
        )

        checkpointing = self.gradient_checkpointing
        for i, block in enumerate(self.transformer["h"]):
            cache = None if kv_caches is None else kv_caches[i]
            if checkpointing:
                x = checkpoint(block, x, cache, use_reentrant=False)
            else:
                x = block(x, kv_cache=cache)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def new_kv_caches(self) -> list:
        """One empty cache entry per transformer block."""
        return [{"k": None, "v": None} for _ in self.transformer["h"]]

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 eos_id: int | None = None, kv_caches: list | None = None,
                 generator=None, repetition_penalty: float = 1.0,
                 stop_on_eos: bool = True) -> torch.Tensor:
        """Sample continuations one token at a time.

        Sampling is delegated to :mod:`apexgpt.models.sampling` so the CLI,
        the GUI and the HTTP API all behave identically.
        """
        from .sampling import sample_next_token

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond, kv_caches=kv_caches)
            next_id = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated=idx[0],
                generator=generator,
            )
            idx = torch.cat((idx, next_id), dim=1)
            if stop_on_eos and eos_id is not None and next_id.item() == eos_id:
                break
        return idx

"""Inference service: checkpoint discovery, loading, and streaming generation.

One implementation backs the CLI, the GUI and the HTTP API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ...core.device import get_device
from ...core.paths import CHECKPOINTS_DIR, RUNS_DIR
from ...models.gpt import GPT


def find_checkpoint(explicit: str | Path | None = None) -> Path:
    """Locate a checkpoint, most recent real run by default.

    Runs whose directory starts with ``_`` are test artifacts and are skipped,
    so a throwaway smoke run cannot shadow the checkpoint you actually trained.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    default = CHECKPOINTS_DIR / "checkpoint.pt"
    if default.exists():
        return default

    runs = [p for p in RUNS_DIR.glob("*/checkpoint.pt")
            if not p.parent.name.startswith(("_", "."))]
    if runs:
        return max(runs, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        "no checkpoint found -- train one first: python -m tinyllm train")


def load_tokenizer():
    from ..data.service import load_tokenizer as _load
    return _load()


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    seed: int | None = None
    repetition_penalty: float = 1.0
    use_cache: bool = True
    stop_on_eos: bool = True

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


@dataclass
class InferenceEngine:
    """Loads a checkpoint once and serves generation requests from it."""
    device: str = "auto"
    checkpoint: Path | None = None
    model: GPT | None = None
    payload: dict = field(default_factory=dict)
    tokenizer: object | None = None

    def load(self, checkpoint: str | Path | None = None) -> "InferenceEngine":
        torch_device = get_device(self.device)
        path = find_checkpoint(checkpoint or self.checkpoint)
        model, payload = GPT.load(path, map_location=str(torch_device))
        model.eval()
        self.model = model
        self.payload = payload
        self.checkpoint = path
        self.tokenizer = load_tokenizer()
        self._device = torch_device
        return self

    @property
    def device_obj(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("engine not loaded; call load() first")
        return next(self.model.parameters()).device

    def metadata(self) -> dict:
        if self.model is None:
            return {}
        cfg = self.model.cfg
        extra = self.payload.get("extra") or {}
        return {
            "checkpoint": str(self.checkpoint),
            "parameters": self.model.num_parameters(),
            "parameters_m": round(self.model.num_parameters() / 1e6, 1),
            "n_layer": cfg.n_layer,
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "step": self.payload.get("step", 0),
            "val_loss": extra.get("val_loss"),
            "device": str(self.device_obj),
        }

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        device = self.device_obj
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        limit = self.model.cfg.block_size
        if ids.shape[1] > limit:
            ids = ids[:, -limit:]
        return ids

    def generate_text(self, req: GenerationRequest) -> str:
        """Generate a full continuation and return the decoded text."""
        req.validate()
        pieces = list(self.stream(req))
        return "".join(pieces)

    def stream(self, req: GenerationRequest):
        """Yield decoded text incrementally, one token at a time."""
        req.validate()
        model, tokenizer = self.model, self.tokenizer

        if req.seed is not None:
            torch.manual_seed(int(req.seed))

        ids = self._encode_prompt(req.prompt)
        kv = model.new_kv_caches() if req.use_cache else None

        from ...models.sampling import sample_next_token

        for _ in range(req.max_new_tokens):
            idx_cond = ids[:, -model.cfg.block_size:]
            with torch.no_grad():
                logits, _ = model(idx_cond, kv_caches=kv)
                next_id = sample_next_token(
                    logits[:, -1, :],
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                    generated=ids[0],
                )
            ids = torch.cat((ids, next_id), dim=1)
            token_id = next_id.item()
            yield tokenizer.decode([token_id])
            if req.stop_on_eos and token_id == tokenizer.eos_token_id:
                return
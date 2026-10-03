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
        "no checkpoint found -- train one first: python -m apexgpt train")


def load_tokenizer(kind: str = "gpt2"):
    """Build the tokenizer named by ``kind`` ("gpt2" or "char")."""
    from ..data.tokenizers import load_tokenizer as _load
    return _load(kind)


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
class Prediction:
    """One candidate continuation, with the model's confidence in it."""

    rank: int
    token_id: int
    text: str
    probability: float
    logprob: float

    @property
    def label(self) -> str:
        """The token as something readable, never an empty string.

        A space is shown as ``\\u0020`` and a newline as ``\\n``, because a
        column of blanks is impossible to read and cannot be copy-pasted.
        """
        shown = (self.text.replace(" ", "\\u0020").replace("\n", "\\n")
                 .replace("\r", "\\r").replace("\t", "\\t"))
        return shown if shown else repr(self.text)

    def as_dict(self) -> dict:
        return {"rank": self.rank, "token_id": self.token_id, "text": self.text,
                "probability": round(self.probability, 6),
                "logprob": round(self.logprob, 6)}


@dataclass
class Step:
    """One generated token, with what the model thought of it."""

    text: str
    token_id: int
    logprob: float
    finished: bool = False
    stop_reason: str = ""

    def as_dict(self) -> dict:
        return {"text": self.text, "token_id": self.token_id,
                "logprob": round(self.logprob, 6),
                "stop_reason": self.stop_reason or None}


@dataclass
class InferenceEngine:
    """Loads a checkpoint once and serves generation requests from it."""
    device: str = "auto"
    checkpoint: Path | None = None
    model: GPT | None = None
    payload: dict = field(default_factory=dict)
    tokenizer: object | None = None
    tokenizer_kind: str | None = None

    def load(self, checkpoint: str | Path | None = None) -> "InferenceEngine":
        torch_device = get_device(self.device)
        path = find_checkpoint(checkpoint or self.checkpoint)
        model, payload = GPT.load(path, map_location=str(torch_device))
        model.eval()
        self.model = model
        self.payload = payload
        self.checkpoint = path
        # The checkpoint says which tokenizer its tokens were made with. An
        # explicit kind wins; older checkpoints have neither and were GPT-2.
        self.tokenizer_kind = (self.tokenizer_kind
                               or (payload.get("extra") or {}).get("tokenizer")
                               or "gpt2")
        recorded = (payload.get("extra") or {}).get("vocab_size")
        if recorded is not None and int(recorded) != int(model.cfg.vocab_size):
            raise RuntimeError(
                f"{path.name} was saved with a {recorded}-id vocabulary but its "
                f"model has {model.cfg.vocab_size}; refusing to decode")
        self.tokenizer = load_tokenizer(self.tokenizer_kind)
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
            "tokenizer": self.tokenizer_kind,
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
        return "".join(step.text for step in self.stream_steps(req))

    def generate_with_details(self, req: GenerationRequest) -> tuple[str, dict]:
        """``(text, details)`` where details carry the logprobs and stop reason.

        ``details`` is the OpenAI-shaped ``logprobs`` payload plus ``finish_reason``,
        which is what the HTTP service returns and what a caller scoring a model
        needs.
        """
        steps = list(self.stream_steps(req))
        text = "".join(step.text for step in steps)
        finish = steps[-1].stop_reason if steps and steps[-1].finished else "length"
        return text, {
            "logprobs": {
                "tokens": [step.text for step in steps],
                "token_ids": [step.token_id for step in steps],
                "token_logprobs": [round(step.logprob, 6) for step in steps],
                "text_offset": [0],
            },
            "finish_reason": finish,
            "generated_tokens": len(steps),
        }

    def predict_next(self, prompt: str, top_k: int = 10,
                     temperature: float = 1.0) -> list[Prediction]:
        """Rank what the model expects to come next, without sampling anything.

        This is the autoregressive objective made visible: the loss the model
        was trained on *is* the negative log-probability of the token that
        actually followed. Returning the distribution shows that number.
        """
        predictions, _ = self.predict_next_with_entropy(prompt, top_k, temperature)
        return predictions

    def predict_next_with_entropy(self, prompt: str, top_k: int = 10,
                                  temperature: float = 1.0
                                  ) -> tuple[list[Prediction], float]:
        """``predict_next`` plus the entropy of the whole next-token distribution.

        Entropy is in nats: ``ln(vocab_size)`` is a uniform model and ``0`` is a
        perfectly confident one, which makes it comparable across vocabularies.
        """
        if self.model is None:
            raise RuntimeError("engine not loaded; call load() first")
        from ...models.sampling import next_token_distribution

        ids = self._encode_prompt(prompt)
        with torch.no_grad():
            logits, _ = self.model(ids[:, -self.model.cfg.block_size:])
        token_ids, probs, entropy = next_token_distribution(
            logits, top_k=top_k, temperature=temperature)
        out = []
        for rank, (token_id, prob) in enumerate(
                zip(token_ids.tolist(), probs.tolist()), start=1):
            prob = float(prob)
            out.append(Prediction(
                rank=rank,
                token_id=int(token_id),
                text=self.tokenizer.decode([int(token_id)]),
                probability=prob,
                logprob=float(torch.tensor(max(prob, 1e-12)).log()),
            ))
        return out, entropy

    def predict_next_text(self, prompt: str, temperature: float = 1.0) -> str:
        """The single most likely next piece of text - the next word."""
        best = self.predict_next(prompt, top_k=1, temperature=temperature)
        return best[0].text if best else ""

    def stream(self, req: GenerationRequest):
        """Yield decoded text incrementally, one token at a time."""
        for step in self.stream_steps(req):
            yield step.text

    def stream_steps(self, req: GenerationRequest):
        """Yield a :class:`Step` per generated token: text, id, logprob, stop.

        ``stream`` is this minus the bookkeeping, which keeps the HTTP API's
        ``logprobs`` field honest without duplicating the decoding loop.
        """
        req.validate()
        model, tokenizer = self.model, self.tokenizer

        if req.seed is not None:
            torch.manual_seed(int(req.seed))

        ids = self._encode_prompt(req.prompt)
        kv = model.new_kv_caches() if req.use_cache else None

        from ...models.sampling import sample_next_token_with_logprob

        for produced in range(req.max_new_tokens):
            idx_cond = ids[:, -model.cfg.block_size:]
            with torch.no_grad():
                logits, _ = model(idx_cond, kv_caches=kv)
                next_id, logprob = sample_next_token_with_logprob(
                    logits[:, -1, :],
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                    generated=ids[0],
                )
            ids = torch.cat((ids, next_id), dim=1)
            token_id = next_id.item()
            stop_reason = ""
            if req.stop_on_eos and token_id == tokenizer.eos_token_id:
                stop_reason = "eos"
            elif produced == req.max_new_tokens - 1:
                stop_reason = "length"
            yield Step(text=tokenizer.decode([token_id]), token_id=token_id,
                       logprob=float(logprob), finished=bool(stop_reason),
                       stop_reason=stop_reason)
            if stop_reason == "eos":
                return
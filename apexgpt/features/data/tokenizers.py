"""Which tokenizer a corpus is built with, and the byte-level one.

Two tokenizers ship:

``gpt2``  GPT-2 byte-level BPE, 50,257 ids. The default: matches every modern
         model, and the right choice when the corpus is large.
``char``  Raw UTF-8 bytes, 257 ids (0-255 plus ``<|endoftext|>``). Lossless,
         no training, no download, and for a *small* corpus on a *small* model
         it is strictly better: a 30M model on tiny Shakespeare spends 19.3M
         parameters on the GPT-2 embedding table and only 10.6M on the
         transformer. Dropping the vocabulary to 257 gives most of those
         parameters back to the layers that actually predict.

The choice is a property of the **corpus**, not of a run, so it is written next
to the token binaries as ``tokenizer.json`` and read back by training and
inference. That is what stops a char-trained checkpoint being decoded with
GPT-2 BPE and producing garbage.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ...core.paths import TOKENIZER_DIR

#: Name of the descriptor written beside train.bin / val.bin.
SPEC_FILENAME = "tokenizer.json"

GPT2_VOCAB_SIZE = 50257
BYTE_VOCAB_SIZE = 257          # 256 byte values + <|endoftext|>
BYTE_EOS_ID = 256


@dataclass(frozen=True)
class TokenizerSpec:
    """Everything needed to rebuild a tokenizer for a token file."""
    kind: str = "gpt2"
    name: str = "gpt2"
    vocab_size: int = GPT2_VOCAB_SIZE
    eos_token_id: int = 50256

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TokenizerSpec":
        return cls(kind=payload["kind"], name=payload.get("name", payload["kind"]),
                   vocab_size=int(payload["vocab_size"]),
                   eos_token_id=int(payload.get("eos_token_id", 0)))


GPT2 = TokenizerSpec(kind="gpt2", name="gpt2", vocab_size=GPT2_VOCAB_SIZE,
                     eos_token_id=50256)
BYTE = TokenizerSpec(kind="char", name="byte-level", vocab_size=BYTE_VOCAB_SIZE,
                     eos_token_id=BYTE_EOS_ID)

SPECS: dict[str, TokenizerSpec] = {"gpt2": GPT2, "char": BYTE}
ALIASES = {"byte": "char", "bytes": "char", "char-level": "char",
           "bpe": "gpt2", "gpt-2": "gpt2"}


class ByteTokenizer:
    """One id per UTF-8 byte, with the call surface the pipeline expects.

    Mirrors the parts of the Hugging Face tokenizer API that ApexGPT uses:
    ``__call__`` over one string or a list, ``decode``, ``vocab_size`` and
    ``eos_token_id``.
    """

    kind = "char"
    name = "byte-level"
    vocab_size = BYTE_VOCAB_SIZE
    eos_token_id = BYTE_EOS_ID

    def __init__(self, spec: TokenizerSpec = BYTE):
        self.spec = spec

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        # ids >= 256 are specials (256 is <|endoftext|>); masking them into
        # bytes would turn eos into a NUL character
        values = [int(i) for i in ids if int(i) < 256]
        if skip_special_tokens:
            values = [v for v in values if v < 256]
        # errors="replace" rather than "strict": a truncated generation must
        # render as text, not raise
        return bytes(values).decode("utf-8", errors="replace")

    def batch_decode(self, sequences, skip_special_tokens: bool = False) -> list[str]:
        return [self.decode(seq, skip_special_tokens) for seq in sequences]

    def __call__(self, text, add_special_tokens: bool = False,
                 return_tensors: str | None = None, **kwargs) -> dict:
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        encoded = [self.encode(t) for t in texts]
        if return_tensors == "pt":
            import torch
            width = max((len(e) for e in encoded), default=0)
            padded = torch.zeros(len(encoded), width, dtype=torch.long)
            for row, ids in enumerate(encoded):
                if ids:
                    padded[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            return {"input_ids": padded}
        return {"input_ids": encoded[0] if single else encoded}

    def __len__(self) -> int:
        return self.vocab_size


def spec_for(kind: str | None) -> TokenizerSpec:
    """Resolve a tokenizer name or alias to its spec."""
    name = (kind or "gpt2").strip().lower()
    name = ALIASES.get(name, name)
    if name not in SPECS:
        raise ValueError(f"unknown tokenizer {kind!r}. Known: "
                         + ", ".join(sorted(SPECS)) + " (aliases: "
                         + ", ".join(sorted(ALIASES)) + ")")
    return SPECS[name]


def load_tokenizer(kind: str | None = "gpt2",
                   tokenizer_dir: Path = TOKENIZER_DIR):
    """Build the tokenizer named by ``kind``.

    For ``gpt2`` this is the hub copy, cached under ``tokenizer_dir`` and
    verified: a local copy that reloads with an empty vocabulary is ignored,
    because that silently produces a zero-token corpus.
    """
    spec = spec_for(kind)
    if spec.kind == "char":
        return ByteTokenizer(spec)

    from transformers import GPT2TokenizerFast

    probe = "Hello, world! The quick brown fox."
    tokenizer_dir = Path(tokenizer_dir)

    if tokenizer_dir.exists() and any(tokenizer_dir.iterdir()):
        try:
            local = GPT2TokenizerFast.from_pretrained(str(tokenizer_dir))
            if local(probe, add_special_tokens=False)["input_ids"]:
                return local
            print("[tokenizer] local copy yields zero tokens, ignoring it")
        except Exception as exc:
            print(f"[tokenizer] local copy unusable ({exc}), falling back to hub")

    print("[tokenizer] loading gpt2 tokenizer from hub (cached after first run)")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = 1_000_000

    try:
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(tokenizer_dir))
        if not GPT2TokenizerFast.from_pretrained(str(tokenizer_dir))(
                probe, add_special_tokens=False)["input_ids"]:
            print("[tokenizer] WARNING: save/reload produced an empty tokenizer; "
                  "keeping the in-memory copy")
            import shutil
            shutil.rmtree(tokenizer_dir, ignore_errors=True)
    except Exception as exc:
        print(f"[tokenizer] save failed ({exc}); using in-memory copy")

    return tok


# --------------------------------------------------------------------------- #
# the spec that travels with the token files
# --------------------------------------------------------------------------- #
def write_spec(binary_dir: Path | str, spec: TokenizerSpec) -> Path:
    """Record the tokenizer beside ``train.bin``."""
    directory = Path(binary_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SPEC_FILENAME
    path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    return path


def read_spec(binary_dir: Path | str) -> TokenizerSpec | None:
    """Read the recorded tokenizer, or ``None`` when the corpus predates this."""
    path = Path(binary_dir) / SPEC_FILENAME
    if not path.exists():
        return None
    try:
        return TokenizerSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def resolve_corpus_spec(binary_dir: Path | str,
                        fallback: str = "gpt2") -> TokenizerSpec:
    """The corpus's tokenizer, defaulting to GPT-2 for older builds."""
    return read_spec(binary_dir) or spec_for(fallback)
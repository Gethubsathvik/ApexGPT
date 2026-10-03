"""Data service: download, tokenize, split, and batch.

All dataset behaviour lives here so the CLI, tests and any future server share
one implementation. Corpus *sources* live in :mod:`.sources`; this module turns
whichever one was selected into ``train.bin`` / ``val.bin``.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from ...core.config import DataConfig
from ...core.paths import TOKENIZER_DIR
from .sources import Corpus, fetch_corpus, legacy_corpus, resolve
from .tokenizers import (TokenizerSpec, load_tokenizer, read_spec,
                         resolve_corpus_spec, spec_for, write_spec)

WIKI_REPO_ID = "wikimedia/wikipedia"
WIKI_REPO_CONFIG = "20231101.en"


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
def corpus_tokenizer(cfg: DataConfig) -> TokenizerSpec:
    """The tokenizer a corpus is already built with, honouring ``--tokenizer``."""
    recorded = read_spec(cfg.binary_dir)
    if recorded is not None and recorded.kind == (cfg.tokenizer or "gpt2"):
        return recorded
    if cfg.tokenizer:
        return spec_for(cfg.tokenizer)
    return recorded or spec_for("gpt2")


def build_tokenizer(cfg: DataConfig, progress=print):
    """Load the tokenizer for ``cfg``, and record it next to the tokens.

    The descriptor is what training and inference read back, so a corpus can
    never be tokenized one way and decoded another.
    """
    spec = corpus_tokenizer(cfg)
    tokenizer = load_tokenizer(spec.kind)
    progress(f"[tokenize] tokenizer {spec.kind} "
             f"({spec.name}, vocab {tokenizer.vocab_size})")
    write_spec(cfg.binary_dir, spec)
    return tokenizer


# --------------------------------------------------------------------------- #
# download + extract
# --------------------------------------------------------------------------- #
def download_wikipedia(target_bytes: int, max_shards: int, raw_dir: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download, list_repo_files

    raw_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        f for f in list_repo_files(WIKI_REPO_ID, repo_type="dataset")
        if f.startswith(f"{WIKI_REPO_CONFIG}/train")
    )

    print(f"[download] target ~{target_bytes / 1e6:.0f} MB, "
          f"fetching up to {max_shards} shard(s)")

    downloaded: list[Path] = []
    total = 0
    for fname in files:
        if len(downloaded) >= max_shards or total >= target_bytes:
            break
        local = hf_hub_download(repo_id=WIKI_REPO_ID, filename=fname,
                               repo_type="dataset", local_dir=str(raw_dir))
        size = Path(local).stat().st_size
        total += size
        downloaded.append(Path(local))
        print(f"  + {Path(fname).name}  {size / 1e6:7.1f} MB  (total {total / 1e6:.0f} MB)")

    if not downloaded:
        raise RuntimeError("no wikipedia shards were downloaded")
    return downloaded


def extract_corpus(shards: list[Path], corpus_path: Path) -> int:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with corpus_path.open("w", encoding="utf-8") as out:
        for shard in shards:
            reader = pq.ParquetFile(shard)
            for batch in reader.iter_batches(batch_size=2000, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    if not text:
                        continue
                    out.write(text.replace("\r\n", "\n").strip())
                    out.write("\n\n")
                    written += len(text)
            del reader
    print(f"[extract] {written / 1e6:.1f} M characters -> {corpus_path}")
    return written


# --------------------------------------------------------------------------- #
# tokenize + split
# --------------------------------------------------------------------------- #
def tokenize_corpus(corpus_path: Path, tokenizer, out_dir: Path,
                    train_split: float = 0.8, num_workers: int = 1,
                    chunk_chars: int = 2_000_000) -> tuple[int, int]:
    """Tokenize the corpus and write ``train.bin`` / ``val.bin`` as uint16.

    Tokens stream to one temporary file as they are produced, then that file is
    split by block copy. Collecting every id in a list first would need roughly
    twice peak RAM for no benefit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out_dir / "train.bin", out_dir / "val.bin"
    tmp_all = out_dir / "all.tmp.bin"
    batch_chunks = max(4, num_workers * 2)
    total_bytes = corpus_path.stat().st_size

    total_tokens = 0
    with corpus_path.open("r", encoding="utf-8") as fh, tmp_all.open("wb") as sink, \
            tqdm(total=total_bytes, desc="[tokenize] chars", unit="B",
                 unit_scale=True, unit_divisor=1_000_000) as bar:
        while True:
            chunks: list[str] = []
            chars = 0
            while len(chunks) < batch_chunks:
                buf = fh.read(chunk_chars)
                if not buf:
                    break
                chunks.append(buf)
                chars += len(buf)
            if not chunks:
                break

            for ids in tokenizer(chunks, add_special_tokens=False)["input_ids"]:
                arr = np.asarray(ids, dtype=np.uint16)
                if arr.size and int(arr.max()) >= 65536:
                    raise ValueError("token id exceeds the uint16 range")
                sink.write(arr.tobytes())
                total_tokens += arr.size
            bar.update(chars)

    print(f"[tokenize] {total_tokens:,} tokens ({total_tokens / 1e6:.2f}M), dtype=uint16")

    n_train = _split_binary(tmp_all, train_path, val_path, total_tokens, train_split)
    tmp_all.unlink(missing_ok=True)

    n_val = total_tokens - n_train
    pct = int(round(train_split * 100))
    print(f"[split]    {pct}/{100 - pct} -> train {n_train:,} tok | val {n_val:,} tok")
    return n_train, n_val


def _split_binary(src: Path, train_path: Path, val_path: Path,
                  total_tokens: int, train_split: float) -> int:
    """Split a flat uint16 file into train/val. Returns the train token count."""
    split_at = int(total_tokens * train_split)
    block = 1 << 20
    consumed = 0
    n_train = 0
    with src.open("rb") as fh, \
            train_path.open("wb") as f_train, val_path.open("wb") as f_val:
        while consumed < total_tokens:
            raw = fh.read(block * 2)
            if not raw:
                break
            take = min(len(raw) // 2, total_tokens - consumed)
            arr = np.frombuffer(raw[: take * 2], dtype=np.uint16)
            head = min(arr.size, split_at - consumed)
            if head > 0:
                f_train.write(arr[:head].tobytes())
                n_train += head
            if arr.size - head > 0:
                f_val.write(arr[head:].tobytes())
            consumed += arr.size
    return n_train


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #
class TokenBatcher:
    """Random ``block_size`` windows over the packed token binaries.

    The memmaps are opened once and reused. Re-opening a 461 MB file on every
    batch was the single biggest waste in the training loop.
    """

    def __init__(self, binary_dir: Path, device: str = "cpu",
                 batch_size: int = 8, block_size: int = 256,
                 val_batch_size: int | None = None):
        self.binary_dir = Path(binary_dir)
        self.device = device
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.block_size = block_size
        self._maps: dict[str, np.memmap] = {}

    def _memmap(self, split: str) -> np.memmap:
        if split not in self._maps:
            path = self.binary_dir / f"{split}.bin"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found -- run: python -m apexgpt data prepare")
            self._maps[split] = np.memmap(path, dtype=np.uint16, mode="r")
        return self._maps[split]

    def get_batch(self, split: str = "train", batch_size: int | None = None):
        """Return ``(x, y)`` where ``y`` is ``x`` shifted left by one."""
        batch_size = batch_size or (
            self.val_batch_size if split == "val" else self.batch_size)
        data = self._memmap(split)

        span = self.block_size + 1
        if len(data) <= span:
            raise ValueError(
                f"{split}.bin holds {len(data)} tokens, need more than {span}")

        high = len(data) - span
        ix = torch.randint(high, (batch_size,))
        rows_x = np.stack([data[i: i + self.block_size].astype(np.int64) for i in ix])
        rows_y = np.stack([data[i + 1: i + 1 + self.block_size].astype(np.int64) for i in ix])
        x = torch.from_numpy(rows_x)
        y = torch.from_numpy(rows_y)

        # pinning only helps CUDA transfers and errors without an accelerator
        if torch.cuda.is_available():
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def close(self) -> None:
        self._maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def count_distinct_tokens(path: Path, vocab_size: int, chunk: int = 1 << 22) -> int:
    """Count distinct token ids without sorting the whole stream.

    ``np.unique`` sorts all 242M values, which is slow and memory hungry for a
    number only used for a log line. A presence bitmap over the vocabulary is
    exact and linear.
    """
    seen = np.zeros(vocab_size, dtype=bool)
    data = np.memmap(path, dtype=np.uint16, mode="r")
    for start in range(0, len(data), chunk):
        seen[np.unique(data[start:start + chunk])] = True
    return int(seen.sum())


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def prepare(cfg: DataConfig, target_mb: int = 1000, num_shards: int = 2,
            keep_parquet: bool = False, redownload: bool = False,
            force: bool = False, num_workers: int = 1,
            source: str | None = None, tokenizer: str | None = None,
            progress=print) -> dict:
    """Run the full data pipeline. Returns a small report dict.

    ``source`` selects the corpus: a registry key (``shakespeare``, ``wikitext``,
    ``tinystories``, ``openwebtext``), or ``hf:``/``kaggle:``/``local:``/``url:``
    prefixed form. Defaults to ``cfg.dataset``.

    ``tokenizer`` selects ``gpt2`` or ``char``; the choice is recorded beside the
    token binaries and read back by training and inference.
    """
    corpus_spec: Corpus = resolve(source or cfg.dataset)
    cfg.select_dataset(corpus_spec.key)
    if tokenizer:
        cfg.tokenizer = tokenizer
    if cfg.is_ready() and not force:
        recorded = resolve_corpus_spec(cfg.binary_dir, cfg.tokenizer)
        progress(f"[skip]  {cfg.train_path} and {cfg.val_path} already built "
                 f"with the {recorded.kind} tokenizer (use --force to rebuild)")
        return {"skipped": True, "dataset": cfg.dataset,
                "corpus": str(cfg.corpus_path), "tokenizer": recorded.kind}

    token = build_tokenizer(cfg, progress)
    cfg.vocab_size = int(getattr(token, "vocab_size", cfg.vocab_size))
    corpus_path = cfg.corpus_path

    if force:
        # --force only invalidates token binaries; rebuilding the corpus is
        # expensive, so that needs --redownload.
        for path in (cfg.train_path, cfg.val_path):
            path.unlink(missing_ok=True)
    if redownload and corpus_path.exists():
        progress(f"[reset] removing corpus {corpus_path}")
        corpus_path.unlink()

    t0 = time.time()
    progress(f"[data]    source '{corpus_spec.key}' ({corpus_spec.title})")

    cached = legacy_corpus(cfg.raw_dir, cfg.dataset)
    if cached is not None and not redownload:
        progress(f"[migrate] reusing the old cache file {cached}")
        cached.rename(corpus_path)

    if corpus_spec.kind == "wikipedia":
        _build_wikipedia(cfg, corpus_path, target_mb, num_shards,
                         keep_parquet, redownload, progress)
    else:
        fetch_corpus(corpus_spec, corpus_path, cfg.raw_dir,
                     target_mb=target_mb, progress=progress)

    n_train, n_val = tokenize_corpus(corpus_path, token, cfg.binary_dir,
                                     train_split=cfg.train_split,
                                     num_workers=num_workers)
    progress(f"[done]  data prep in {time.time() - t0:.0f}s")
    return {"skipped": False, "train_tokens": n_train, "val_tokens": n_val,
            "dataset": cfg.dataset, "corpus": str(corpus_path),
            "tokenizer": cfg.tokenizer or "gpt2"}


def _build_wikipedia(cfg: DataConfig, corpus_path: Path, target_mb: int,
                     num_shards: int, keep_parquet: bool, redownload: bool,
                     progress) -> None:
    """The original parquet-shard path, kept because it is the default corpus."""
    ready = corpus_path.exists() and corpus_path.stat().st_size > 0
    if ready and not redownload:
        progress(f"[skip]  reusing existing corpus {corpus_path} "
                 f"({corpus_path.stat().st_size / 1e6:.0f} MB chars)")
        return

    shards = download_wikipedia(int(target_mb * 1e6), num_shards, cfg.raw_dir)
    extract_corpus(shards, corpus_path)
    if not keep_parquet:
        freed = sum(p.stat().st_size for p in shards)
        for p in shards:
            p.unlink(missing_ok=True)
        shutil.rmtree(cfg.raw_dir / ".cache", ignore_errors=True)
        progress(f"[cleanup] removed parquet shards, freed {freed / 1e6:.0f} MB "
                 f"(pass --keep-parquet to retain)")

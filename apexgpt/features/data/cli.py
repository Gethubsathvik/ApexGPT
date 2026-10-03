"""Data feature CLI: list corpora, build one, print a tokenization report."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ...core.config import Config
from ...core.seeding import set_seed
from ...core.text import console_safe
from .service import TokenBatcher, count_distinct_tokens, prepare
from .sources import DEFAULT_SOURCE, REGISTRY, describe_sources, resolve
from .tokenizers import load_tokenizer, resolve_corpus_spec


def report(cfg, tokenizer, n_windows: int = 5) -> None:
    """Print human-readable verification of the tokenized corpus."""
    from ..data.tokenizers import resolve_corpus_spec

    spec = resolve_corpus_spec(cfg.binary_dir, cfg.tokenizer)
    train = np.memmap(cfg.train_path, dtype=np.uint16, mode="r")
    print("\n" + "=" * 70)
    print(f"TOKENIZED SAMPLE OUTPUT ({spec.kind} tokenizer, "
          f"vocab {spec.vocab_size}, corpus {cfg.dataset})")
    print("=" * 70)

    corpus_path = cfg.corpus_path
    if corpus_path.exists():
        raw = corpus_path.read_text(encoding="utf-8")[:400].strip()
        print("\n-- Round-trip of raw corpus text --")
        print(f"RAW  : {console_safe(raw[:200])!r}")
        ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
        print(f"IDS  : {list(ids)[:24]}{' ...' if len(ids) > 24 else ''}  "
              f"({len(ids)} tokens)")
        print(f"TOKENS: {[console_safe(tokenizer.decode([i])) for i in list(ids)[:12]]}")
        back = tokenizer.decode(ids)
        print(f"DECODED: {console_safe(back[:200])!r}")
        print("round-trip ok: ", back.strip() == raw.strip())

    print("\n-- Token statistics --")
    print(f"tokenizer       : {spec.kind} ({spec.name})")
    print(f"vocab size       : {tokenizer.vocab_size}")
    print(f"eos token id     : {tokenizer.eos_token_id}")
    print(f"train tokens     : {len(train):,}")
    print(f"val tokens       : {len(np.memmap(cfg.val_path, dtype=np.uint16, mode='r')):,}")
    print(f"distinct ids seen: {count_distinct_tokens(cfg.train_path, tokenizer.vocab_size):,}")

    window = min(128, len(train) - 1)
    print(f"\n-- {n_windows} random {window}-token windows decoded --")
    rng = np.random.default_rng(0)
    for i in range(n_windows):
        start = int(rng.integers(0, len(train) - window))
        chunk = train[start: start + window]
        text = tokenizer.decode(chunk.tolist()).replace("\n", " ")
        print(f"[{i}] ids[:8]={chunk[:8].tolist()} -> {console_safe(text[:150])!r}")

    print("\n-- Next-token-prediction pair (x, y) --")
    with TokenBatcher(cfg.binary_dir, batch_size=1, block_size=16) as batcher:
        x, y = batcher.get_batch("train")
    print(f"x = {x[0].tolist()}")
    print(f"y = {y[0].tolist()}")
    print("y is x shifted left by 1:", y[0].tolist()[:-1] == x[0].tolist()[1:])
    print(f"decode(x) = {console_safe(tokenizer.decode(x[0].tolist()))!r}")
    print(f"decode(y) = {console_safe(tokenizer.decode(y[0].tolist()))!r}")
    print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="apexgpt data",
        description="Prepare ApexGPT training data from Wikipedia, tiny "
                    "Shakespeare, a Hugging Face dataset, Kaggle, or a local file")
    ap.add_argument("command", nargs="?", default="prepare",
                    choices=["prepare", "sources"],
                    help="'prepare' (default) builds the corpus; 'sources' lists "
                         "what is available")
    ap.add_argument("--source", default=None,
                    help="corpus: " + ", ".join(sorted(REGISTRY))
                         + ", or hf:<repo_id> / kaggle:<slug> / local:<path> / url:<link>")
    ap.add_argument("--dataset", default=None,
                    help=f"corpus for the data directories (default {DEFAULT_SOURCE})")
    ap.add_argument("--tokenizer", default=None, choices=["gpt2", "char"],
                    help="gpt2: 50,257 ids, best for big corpora. "
                         "char: 257 ids (one per byte), best for small corpora "
                         "and small models - see the README comparison")
    ap.add_argument("--target-mb", type=int, default=1000,
                    help="raw download budget in MB (500-1000 recommended)")
    ap.add_argument("--num-shards", type=int, default=2,
                    help="max parquet shards to download (each ~420 MB)")
    ap.add_argument("--keep-parquet", action="store_true",
                    help="keep downloaded parquet shards after extraction")
    ap.add_argument("--redownload", action="store_true",
                    help="re-download the corpus even if the text cache exists")
    ap.add_argument("--num-workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--force", action="store_true",
                    help="re-tokenize from the cached corpus (add --redownload to refetch)")
    ap.add_argument("--train-split", type=float, default=0.8,
                    help="fraction of tokens used for training")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the tokenization sample report")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "sources":
        print("=" * 70)
        print("\n".join(describe_sources()))
        print("=" * 70)
        return 0

    try:
        corpus = resolve(args.source or args.dataset or DEFAULT_SOURCE)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    cfg = Config()
    set_seed(cfg.train.seed)
    cfg.data.train_split = args.train_split
    cfg.data.select_dataset(corpus.key)
    if args.tokenizer:
        cfg.data.tokenizer = args.tokenizer

    result = prepare(cfg.data, target_mb=args.target_mb,
                     num_shards=args.num_shards,
                     keep_parquet=args.keep_parquet,
                     redownload=args.redownload,
                     force=args.force,
                     num_workers=args.num_workers,
                     source=corpus.key,
                     tokenizer=args.tokenizer)

    if not cfg.data.is_ready():
        print("[error] token binaries are missing after prepare", flush=True)
        return 1

    if not args.no_report:
        spec = resolve_corpus_spec(cfg.data.binary_dir, cfg.data.tokenizer)
        report(cfg.data, load_tokenizer(spec.kind))

    if result.get("skipped"):
        print("\nNothing to do: this corpus is already built.")
    else:
        print(f"\nCorpus '{corpus.key}' ready: {result['train_tokens']:,} train "
              f"tokens, {result['val_tokens']:,} val tokens")
        print(f"  binaries: {cfg.data.binary_dir}")
        print(f"  train it: python -m apexgpt train --dataset {corpus.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
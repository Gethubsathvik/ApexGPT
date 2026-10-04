"""Data feature CLI: list corpora, build one, inspect its tokens, report."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from ...core.config import Config
from ...core.seeding import set_seed
from ...core.text import console_safe
from .service import (TokenBatcher, build_inventory, count_distinct_tokens,
                      prepare)
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


def token_table(corpus_path, tokenizer, top: int = 0, limit: int = 0,
                show_text: int = 0, predict: str = "", top_k: int = 8) -> None:
    """Print every token id with its value: text, count and share of the corpus.

    This is the vocabulary as the model actually receives it - no abstraction
    between the file on disk and the integers the network is fed. The counting
    itself lives in :mod:`.service`, so the GUI shows the same numbers.
    """
    inventory = build_inventory(corpus_path, tokenizer)
    total = inventory.total
    counts = inventory.counts

    print("=" * 70)
    print(f"TOKEN TABLE - {inventory.path.name}")
    print("=" * 70)
    print(f"tokenizer      : {getattr(tokenizer, 'kind', 'gpt2')}")
    print(f"vocabulary     : {tokenizer.vocab_size:,} ids"
          f" (eos = {tokenizer.eos_token_id})")
    print(f"corpus         : {inventory.characters:,} characters")
    print(f"tokens         : {total:,}")
    print(f"distinct ids   : {inventory.distinct:,} "
          f"({100.0 * inventory.distinct / tokenizer.vocab_size:.1f}% of the vocabulary)")
    if inventory.characters:
        print(f"compression    : {total / inventory.characters:.2f} tokens per character")
    print("=" * 70)
    print(f"{'id':>7}  {'token':<14} {'count':>10} {'share':>8}  rank")
    print("-" * 70)

    ordered = counts.most_common() if top else sorted(counts.items())
    if top:
        ordered = ordered[:top]
    rows = ordered[:limit] if limit else ordered
    for rank, (token_id, count) in enumerate(rows, start=1):
        value = inventory.value_of(token_id, tokenizer)
        print(f"{token_id:>7}  {value.label:<14} {count:>10,} "
              f"{100.0 * count / total:>7.3f}%  #{rank}")

    if not rows:
        print("(the corpus produced no tokens)")
    if limit and len(ordered) > limit:
        print(f"... {len(ordered) - limit:,} more ids; use --limit 0 to see them all")
    if top and len(counts) > top:
        shown_top = sum(count for _, count in ordered)
        print(f"-- the {len(ordered)} most frequent of {len(counts):,} ids: "
              f"{100.0 * shown_top / total:.1f}% of the corpus; "
              f"use --top 0 for every id")

    if show_text:
        start = max(0, show_text)
        end = min(total, show_text + 24)
        if end > start:
            window = inventory.ids[start:end]
            print("\n-- a slice of the corpus, id by id --")
            print(f"ids [{start}:{end}] = {[int(i) for i in window]}")
            print("      " + " | ".join(
                f"{int(i)}:{tokenizer.decode([int(i)])!r}" for i in window))

    if predict:
        prompt_ids = [int(i) for i in
                      tokenizer(predict, add_special_tokens=False)["input_ids"]]
        print("\n-- prompt, id by id --")
        print(f"text : {console_safe(predict)!r}")
        print(f"ids  : {prompt_ids}")
        _print_successors(inventory, tokenizer, prompt_ids, top_k)


def _print_successors(inventory, tokenizer, prompt_ids, top_k: int) -> None:
    """Rank what follows the prompt's last id, from the corpus itself.

    Counts of single tokens cannot say what comes *next* - that needs pairs. This
    counts bigrams, which is a real predictor and costs one pass over the corpus:
    the successor of every id with the number of times it was seen there. A
    trained checkpoint does better, and ``apexgpt generate --predict`` prints the
    model's own ranking; this one needs no model at all.
    """
    print("\n-- next token, from bigram counts in this corpus --")
    if not prompt_ids:
        print("(empty prompt)")
        return
    last = prompt_ids[-1]
    following = inventory.successors.get(last)
    print(f"last id : {last} = {tokenizer.decode([last])!r}")
    if not following:
        print("no successor was observed for this id")
        return
    print(f"seen {sum(following.values()):,} times in this corpus")
    for rank, (token_id, count, probability) in enumerate(
            inventory.successors_of(last, top_k), start=1):
        print(f"   {rank}. id={token_id:<7} count={count:>7,} "
              f"p={probability:>7.3%}  "
              f"{tokenizer.decode([token_id])!r}")


def _tokens_command(args, corpus, cfg) -> int:
    """``data tokens``: the id table for a corpus, built or already on disk."""
    from .sources import fetch_corpus

    if args.source and corpus.kind in ("local", "text-url", "hf-stream", "kaggle"):
        # a local file or a URL needs no index: tokenize it where it lies
        corpus_path = corpus_path_for(corpus, cfg.data.raw_dir)
        one_off = True
    else:
        one_off = False
        if not cfg.data.corpus_path.exists():
            if corpus.kind == "wikipedia":
                print("[error] the Wikipedia corpus is not built. This command "
                      "reads text, it does not download 772 MB of parquet:\n"
                      "         python -m apexgpt data prepare --source wikipedia "
                      "--target-mb 5\n"
                      "         python -m apexgpt data tokens --source wikipedia",
                      file=sys.stderr)
                return 1
            print(f"[tokens]  fetching {corpus.key}")
            fetch_corpus(corpus, cfg.data.corpus_path, cfg.data.raw_dir,
                         target_mb=max(1, args.target_mb))
        corpus_path = cfg.data.corpus_path

    if not corpus_path.exists():
        print(f"[error] no corpus text at {corpus_path}", file=sys.stderr)
        return 1

    tokenizer = load_tokenizer(cfg.data.tokenizer)
    print(f"[tokens]  reading {corpus_path}")
    token_table(corpus_path, tokenizer, top=args.top, limit=args.limit,
                show_text=args.slice_at, predict=args.predict,
                top_k=args.top_k)
    if one_off:
        print(f"\nbuild it for training:  python -m apexgpt data prepare "
              f"--source {args.source} --tokenizer {cfg.data.tokenizer}")
    print(f"train a model on it:  python -m apexgpt train "
          f"--dataset {corpus.key} --tokenizer {cfg.data.tokenizer}")
    print("model's own ranking:  python -m apexgpt generate --predict 8")
    return 0


def corpus_path_for(corpus, raw_dir):
    """Where a one-off ``local:``/``url:`` corpus is read from.

    A local plain-text file is read where it lies: copying it into the cache
    would put every ``local:`` spec on the same ``<key>.txt`` path, so the
    second file read would silently report the first one's tokens. Formats
    that need converting, and remote sources, get a cache file named after
    the source instead of after the corpus key, which all of them share
    (``local``, ``url``, ``kaggle``).
    """
    from pathlib import Path

    from .sources import fetch_corpus

    if corpus.kind == "local":
        source = Path(corpus.locator).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"local corpus not found: {source}")
        if source.suffix.lower() in ("", ".txt", ".md"):
            return source

    target = Path(raw_dir) / f"{source_slug(corpus)}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    return fetch_corpus(corpus, target, raw_dir, target_mb=50)


def source_slug(corpus) -> str:
    """A filesystem-safe cache name that identifies the source, not the kind.

    A file name on its own is not an identity: ``a/train.csv`` and
    ``b/train.csv`` are different corpora. The directory a source sits in goes
    into the name too, and for a URL the host, so two one-off corpora cannot
    land on one cache file.
    """
    import re
    from pathlib import Path
    from urllib.parse import urlsplit

    locator = corpus.locator or corpus.key
    if corpus.kind == "text-url":
        parts = urlsplit(locator)
        trail = [part for part in parts.path.strip("/").split("/") if part]
        head = [parts.netloc] if parts.netloc else []
    else:
        trail = [part for part in Path(locator).parts
                 if part not in (".", "..", "\\", "/") and not part.endswith(":")]
        head = []
    if not trail:
        return corpus.key

    stem = Path(trail[-1]).stem or trail[-1]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join([*head, *trail[-2:-1], stem]))
    return slug.strip("-.")[-64:] or corpus.key


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="apexgpt data",
        description="Prepare ApexGPT training data from Wikipedia, tiny "
                    "Shakespeare, a Hugging Face dataset, Kaggle, or a local file")
    ap.add_argument("command", nargs="?", default="prepare",
                     choices=["prepare", "sources", "tokens"],
                     help="'prepare' (default) builds the corpus; 'sources' lists "
                          "what is available; 'tokens' prints every token id with "
                          "its value and predicts the next one")
    ap.add_argument("--source", default=None,
                    help="corpus: " + ", ".join(sorted(REGISTRY))
                         + ", or hf:<repo_id> / kaggle:<slug> / local:<path> / url:<link>")
    ap.add_argument("--dataset", default=None,
                     help=f"corpus for the data directories "
                          f"(default {DEFAULT_SOURCE}, or shakespeare for 'tokens')")
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
    # 'tokens' only
    ap.add_argument("--top", type=int, default=0,
                    help="tokens: show only the N most frequent ids (0 = all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="tokens: show at most N rows (0 = all ids)")
    ap.add_argument("--slice-at", type=int, default=0, metavar="CHAR",
                    help="tokens: print the ids of a 24-token slice at this "
                         "character offset")
    ap.add_argument("--predict", default="", metavar="TEXT",
                    help="tokens: rank the next token after TEXT, from bigram "
                         "counts in this corpus")
    ap.add_argument("--top-k", type=int, default=8,
                    help="tokens: how many candidate successors to rank")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "sources":
        print("=" * 70)
        print("\n".join(describe_sources()))
        print("=" * 70)
        return 0

    # 'tokens' inspects one corpus, so it defaults to the smallest one rather
    # than to the 772 MB Wikipedia default
    fallback = "shakespeare" if args.command == "tokens" else DEFAULT_SOURCE
    try:
        corpus = resolve(args.source or args.dataset or fallback)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    cfg = Config()
    cfg.data.train_split = args.train_split
    cfg.data.select_dataset(corpus.key)
    if args.tokenizer:
        cfg.data.tokenizer = args.tokenizer

    if args.command == "tokens":
        return _tokens_command(args, corpus, cfg)

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

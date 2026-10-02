"""Dataset preparation for TinyLLM.

Pipeline
--------
1. Download a Wikipedia (2023-11-01, en) parquet subset from the Hugging Face
   hub until the target byte budget is reached (default ~1 GB raw).
2. Extract the plain `text` column and stream it into a single UTF-8 corpus.
3. Tokenize the corpus with GPT-2 byte-level BPE and pack the ids into flat
   uint16 binaries.
4. Split 80/20 into train.bin / val.bin.
5. Print a human-readable sample of the tokenized output for verification.

Outputs (under data/):
    raw/           downloaded parquet shards
    corpus.txt     extracted plain text
    binary/train.bin, val.bin   uint16 token streams
    tokenizer/     saved GPT-2 tokenizer

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --target-mb 800 --num-workers 8
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from config import Config, DATA_DIR, set_seed

REPO_ID = "wikimedia/wikipedia"
REPO_CONFIG = "20231101.en"


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def download_wikipedia(target_bytes: int, max_shards: int, raw_dir: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in _repo_files() if f.startswith(f"{REPO_CONFIG}/train")]
    files.sort()

    target_mb = target_bytes / 1e6
    print(f"[download] target ~{target_mb:.0f} MB, fetching up to {max_shards} shard(s)")

    downloaded: list[Path] = []
    total = 0
    for fname in files:
        if len(downloaded) >= max_shards or total >= target_bytes:
            break
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        size = Path(local).stat().st_size
        total += size
        downloaded.append(Path(local))
        print(f"  + {Path(fname).name}  {size / 1e6:7.1f} MB  (total {total / 1e6:.0f} MB)")

    if not downloaded:
        raise RuntimeError("no shards downloaded")
    return downloaded


def _repo_files() -> list[str]:
    from huggingface_hub import list_repo_files
    return list_repo_files(REPO_ID, repo_type="dataset")


# --------------------------------------------------------------------------- #
# extract text
# --------------------------------------------------------------------------- #
def extract_corpus(shards: list[Path], corpus_path: Path) -> int:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with corpus_path.open("w", encoding="utf-8") as out:
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    if not text:
                        continue
                    out.write(text.replace("\r\n", "\n").strip())
                    out.write("\n\n")
                    written += len(text)
            del pf
    print(f"[extract] {written / 1e6:.1f} M characters -> {corpus_path}")
    return written


# --------------------------------------------------------------------------- #
# tokenize + pack
# --------------------------------------------------------------------------- #
def get_tokenizer(tok_dir: Path, model_max_len: int = 1_000_000):
    """Load GPT-2 BPE, saving a local copy only if it round-trips correctly.

    transformers>=5 `save_pretrained` can silently write an incomplete
    tokenizer; reloading it yields an empty vocabulary. So verify before
    trusting the local copy and otherwise fall back to the hub copy.
    """
    from transformers import GPT2TokenizerFast

    probe = "Hello, world! The quick brown fox."

    if tok_dir.exists() and any(tok_dir.iterdir()):
        try:
            local = GPT2TokenizerFast.from_pretrained(str(tok_dir))
            if local(probe, add_special_tokens=False)["input_ids"]:
                print(f"[tokenizer] loaded verified local copy from {tok_dir}")
                return local
            print("[tokenizer] local copy decodes to zero tokens, ignoring it")
        except Exception as e:
            print(f"[tokenizer] local copy unusable ({e}), falling back to hub")

    print("[tokenizer] loading gpt2 tokenizer from hub (cached after first run) ...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = model_max_len

    try:
        tok_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(tok_dir))
        check = GPT2TokenizerFast.from_pretrained(str(tok_dir))
        ok = check(probe, add_special_tokens=False)["input_ids"]
        if ok:
            print(f"[tokenizer] saved and verified at {tok_dir}")
        else:
            print("[tokenizer] WARNING: save/reload produced an empty tokenizer; "
                  "will keep using the in-memory hub copy")
            shutil.rmtree(tok_dir, ignore_errors=True)
    except Exception as e:
        print(f"[tokenizer] save failed ({e}); using in-memory copy")

    return tok


def tokenize_corpus(corpus_path: Path, tok, out_dir: Path, num_workers: int):
    """Fast BPE encoding, packed into a flat uint16 numpy array.

    Uses the Rust batch encoder (many chunks per call) so encoding is
    parallelized instead of one 2M-char string at a time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_chars = 2_000_000
    batch_chunks = max(4, num_workers * 2)

    pieces: list[np.ndarray] = []
    total_tokens = 0
    total_chars = corpus_path.stat().st_size

    with corpus_path.open("r", encoding="utf-8") as fh, \
            tqdm(total=total_chars, desc="[tokenize] chars", unit="B",
                 unit_scale=True, unit_divisor=1_000_000) as bar:
        while True:
            batch: list[str] = []
            chars_in_batch = 0
            while len(batch) < batch_chunks:
                buf = fh.read(chunk_chars)
                if not buf:
                    break
                batch.append(buf)
                chars_in_batch += len(buf)
            if not batch:
                break

            encs = tok(batch, add_special_tokens=False)["input_ids"]
            for ids in encs:
                arr = np.asarray(ids, dtype=np.uint16)
                pieces.append(arr)
                total_tokens += arr.size
            bar.update(chars_in_batch)

    ids = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint16)
    assert ids.size == 0 or ids.max() < 65536, "token id exceeds uint16 range"
    del pieces

    print(f"[tokenize] {total_tokens:,} tokens ({total_tokens / 1e6:.2f}M), "
          f"dtype={ids.dtype}")

    split_idx = int(len(ids) * 0.8)
    train, val = ids[:split_idx], ids[split_idx:]
    train.tofile(out_dir / "train.bin")
    val.tofile(out_dir / "val.bin")
    print(f"[split]    80/20 -> train {len(train):,} tok | val {len(val):,} tok")
    return train, val


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def get_batch(split: str, batch_size: int, block_size: int,
              binary_dir: Path, device: str = "cpu"):
    """Sample (x, y) where y = x shifted by one, for next-token prediction."""
    path = binary_dir / f"{split}.bin"
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.from_numpy(np.stack([data[i: i + block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1: i + 1 + block_size].astype(np.int64) for i in ix]))
    # pinning is only useful for CUDA transfer; it errors without an accelerator
    pin = torch.cuda.is_available()
    x = x.pin_memory() if pin else x
    y = y.pin_memory() if pin else y
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# --------------------------------------------------------------------------- #
# sample report
# --------------------------------------------------------------------------- #
def show_samples(tok, binary_dir: Path, corpus_path: Path, n: int = 5) -> None:
    train = np.memmap(binary_dir / "train.bin", dtype=np.uint16, mode="r")
    print("\n" + "=" * 70)
    print("TOKENIZED SAMPLE OUTPUT (GPT-2 BPE, vocab 50257)")
    print("=" * 70)

    print("\n-- Round-trip of raw corpus text --")
    raw = corpus_path.read_text(encoding="utf-8")[:400].strip()
    print(f"RAW  : {raw[:200]!r}")
    ids = tok(raw, add_special_tokens=False)["input_ids"]
    print(f"IDS  : {ids[:24]}{' ...' if len(ids) > 24 else ''}  ({len(ids)} tokens)")
    print(f"TOKENS: {[tok.decode([i]) for i in ids[:12]]}")
    back = tok.decode(ids)
    print(f"DECODED: {back[:200]!r}")
    print(f"round-trip ok: {back.strip() == raw.strip()}")

    print("\n-- Token id / string statistics --")
    print(f"vocab size      : {tok.vocab_size}")
    print(f"eos token id    : {tok.eos_token_id}")
    print(f"train token count: {len(train):,}")
    print(f"distinct ids seen: {len(np.unique(train)):,}")

    print(f"\n-- {n} random {128}-token windows decoded --")
    rng = np.random.default_rng(0)
    for i in range(n):
        s = int(rng.integers(0, len(train) - 128))
        win = train[s: s + 128]
        text = tok.decode(win.tolist()).replace("\n", " ")
        print(f"[{i}] ids[:8]={win[:8].tolist()} -> {text[:150]!r}")

    print("\n-- Next-token-prediction pair (x, y) --")
    x, y = get_batch("train", 1, 16, binary_dir)
    print(f"x = {x[0].tolist()}")
    print(f"y = {y[0].tolist()}")
    print("y is x shifted left by 1 (verify above):", y[0].tolist()[:-1] == x[0].tolist()[1:])
    print(f"decode(x) = {tok.decode(x[0].tolist())!r}")
    print(f"decode(y) = {tok.decode(y[0].tolist())!r}")
    print("=" * 70)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare TinyLLM training data")
    ap.add_argument("--target-mb", type=int, default=1000,
                    help="raw download budget in MB (500-1000 recommended)")
    ap.add_argument("--num-shards", type=int, default=2,
                    help="max parquet shards to download (each ~420 MB)")
    ap.add_argument("--keep-parquet", action="store_true",
                    help="keep downloaded parquet shards after extraction")
    ap.add_argument("--redownload", action="store_true",
                    help="re-download the corpus even if wiki_corpus.txt already exists")
    ap.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--force", action="store_true", help="re-download and re-tokenize")
    args = ap.parse_args()

    cfg = Config()
    set_seed(cfg.train.seed)
    dc = cfg.data

    corpus_path = dc.raw_dir / "wiki_corpus.txt"
    train_bin, val_bin = dc.binary_dir / "train.bin", dc.binary_dir / "val.bin"

    if args.force:
        # --force only invalidates the token binaries; the corpus is expensive
        # to rebuild, so it is kept unless --redownload is given.
        for p in (train_bin, val_bin):
            if p.exists():
                p.unlink()
    if args.redownload and corpus_path.exists():
        print(f"[reset] removing corpus {corpus_path}")
        corpus_path.unlink()

    bins_ready = (train_bin.exists() and val_bin.exists()
                  and train_bin.stat().st_size > 0
                  and val_bin.stat().st_size > 0)
    corpus_ready = corpus_path.exists() and corpus_path.stat().st_size > 0

    if not bins_ready or args.force:
        t0 = time.time()
        tok = get_tokenizer(DATA_DIR / "tokenizer")

        if corpus_ready and not args.redownload:
            print(f"[skip]  reusing existing corpus {corpus_path} "
                  f"({corpus_path.stat().st_size / 1e6:.0f} MB chars)")
        else:
            shards = download_wikipedia(int(args.target_mb * 1e6), args.num_shards, dc.raw_dir)
            extract_corpus(shards, corpus_path)
            if not args.keep_parquet:
                freed = sum(p.stat().st_size for p in shards)
                for p in shards:
                    p.unlink(missing_ok=True)
                shutil.rmtree(dc.raw_dir / ".cache", ignore_errors=True)
                print(f"[cleanup] removed parquet shards, freed {freed / 1e6:.0f} MB "
                      f"(pass --keep-parquet to retain)")

        tokenize_corpus(corpus_path, tok, dc.binary_dir, args.num_workers)
        print(f"[done]  data prep in {time.time() - t0:.0f}s")
    else:
        print(f"[skip]  {train_bin} and {val_bin} already built (use --force to rebuild)")
        tok = get_tokenizer(DATA_DIR / "tokenizer")

    show_samples(tok, dc.binary_dir, corpus_path)

    print("\nCorpus sample (first 500 chars):")
    print(corpus_path.read_text(encoding="utf-8")[:500])
    print("\nLoader shape check:")
    x, y = get_batch("train", dc.batch_size, dc.block_size, dc.binary_dir)
    xv, yv = get_batch("val", dc.batch_size, dc.block_size, dc.binary_dir)
    print(f"  train batch x{tuple(x.shape)} y{tuple(y.shape)} | "
          f"val batch x{tuple(xv.shape)} y{tuple(yv.shape)}")


if __name__ == "__main__":
    main()
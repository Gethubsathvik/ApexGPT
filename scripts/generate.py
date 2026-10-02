"""Inference / text generation for a trained TinyLLM model.

Supports temperature, top-k and top-p (nucleus) sampling, KV caching, and
token-by-token streaming output.

Usage:
    python scripts/generate.py --prompt "The meaning of life is"
    python scripts/generate.py --checkpoint models/runs/gpt-cpu-tiny/checkpoint.pt
    python scripts/generate.py --interactive
    python scripts/generate.py --prompt "Once upon a time" --temperature 0.7 --top-k 40 --top-p 0.95
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import CHECKPOINTS_DIR, DATA_DIR, RUNS_DIR
from model import GPT


def find_checkpoint(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"checkpoint not found: {p}")
        return p
    if (CHECKPOINTS_DIR / "checkpoint.pt").exists():
        return CHECKPOINTS_DIR / "checkpoint.pt"
    runs = sorted(RUNS_DIR.glob("*/checkpoint.pt"), key=lambda p: p.stat().st_mtime)
    if runs:
        return runs[-1]
    raise SystemExit("no checkpoint found -- run scripts/train.py first")


def load_model(path: Path, device: str):
    model, payload = GPT.load(path, map_location=device)
    model.eval()
    return model, payload


def get_tokenizer():
    from transformers import GPT2TokenizerFast
    tok_dir = DATA_DIR / "tokenizer"
    if tok_dir.exists() and any(tok_dir.iterdir()):
        try:
            t = GPT2TokenizerFast.from_pretrained(str(tok_dir))
            if t("probe", add_special_tokens=False)["input_ids"]:
                return t
        except Exception:
            pass
    t = GPT2TokenizerFast.from_pretrained("gpt2")
    t.model_max_length = 1_000_000
    return t


@torch.no_grad()
def generate_stream(model, tokenizer, prompt: str, max_new_tokens: int,
                    temperature: float, top_k: int | None, top_p: float | None,
                    device: str, seed: int | None = None, stop_at_eos: bool = True,
                    use_cache: bool = True, quiet: bool = False):
    """Yield decoded strings incrementally, one token at a time."""
    if seed is not None:
        torch.manual_seed(seed)

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    if ids.shape[1] > model.cfg.block_size:
        ids = ids[:, -model.cfg.block_size:]

    kv_caches = model.new_kv_caches(ids.size(0), device, torch.float32) if use_cache else None
    t0 = time.time()
    produced = 0

    for _ in range(max_new_tokens):
        idx_cond = ids[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond, kv_caches=kv_caches)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cdf = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove_sorted = cdf - torch.softmax(sorted_logits, dim=-1) > top_p
            remove_sorted[..., 0] = False
            logits = logits.masked_fill(
                remove_sorted.scatter(-1, sorted_idx, remove_sorted), float("-inf")
            )

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat((ids, next_id), dim=1)
        produced += 1

        text = tokenizer.decode(next_id[0].tolist())
        yield text

        if stop_at_eos and next_id.item() == tokenizer.eos_token_id:
            break

    dt = time.time() - t0
    if not quiet and dt > 0:
        print(f"\n[gen] {produced} tokens in {dt:.2f}s ({produced / dt:.2f} tok/s)")


def run_once(model, tokenizer, args, device):
    print(f"\nprompt      : {args.prompt!r}")
    print(f"params      : temperature={args.temperature} top_k={args.top_k} "
          f"top_p={args.top_p} max_new_tokens={args.max_new_tokens}")
    print("-" * 70)
    print(args.prompt, end="", flush=True)
    for piece in generate_stream(model, tokenizer, args.prompt, args.max_new_tokens,
                                 args.temperature, args.top_k, args.top_p, device,
                                 args.seed, not args.no_stop, not args.no_cache):
        print(piece, end="", flush=True)
    print("\n" + "-" * 70)


def run_interactive(model, tokenizer, args, device):
    print("\nInteractive mode. Commands: :t <temp> :k <k> :p <p> :n <tokens> :seed <n> :quit")
    prompt = ""
    while True:
        try:
            line = input(f"\n>>> {prompt}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith(":"):
            parts = line[1:].split()
            cmd, val = parts[0], (parts[1] if len(parts) > 1 else None)
            if cmd in ("q", "quit", "exit"):
                break
            try:
                if cmd == "t": args.temperature = float(val)
                elif cmd == "k": args.top_k = int(val)
                elif cmd == "p": args.top_p = float(val)
                elif cmd == "n": args.max_new_tokens = int(val)
                elif cmd == "seed": args.seed = int(val)
                else: print("unknown command")
            except ValueError:
                print("bad value")
            continue
        print(end="", flush=True)
        for piece in generate_stream(model, tokenizer, line, args.max_new_tokens,
                                     args.temperature, args.top_k, args.top_p,
                                     device, args.seed, not args.no_stop, not args.no_cache):
            print(piece, end="", flush=True)
        print()


def main():
    ap = argparse.ArgumentParser(description="Generate text with a trained TinyLLM model")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--prompt", default="The history of the city is")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="disable KV caching")
    ap.add_argument("--no-stop", action="store_true", help="ignore eos token")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_checkpoint(args.checkpoint)
    model, payload = load_model(ckpt, device)
    tokenizer = get_tokenizer()

    print("=" * 70)
    print("TinyLLM inference")
    print("=" * 70)
    print(f"checkpoint : {ckpt}")
    print(f"device     : {device}")
    print(f"parameters : {model.num_parameters():,} ({model.num_parameters() / 1e6:.1f}M)")
    print(f"config     : L={model.cfg.n_layer} d={model.cfg.n_embd} "
          f"H={model.cfg.n_head} ctx={model.cfg.block_size} vocab={model.cfg.vocab_size}")
    if "step" in payload:
        print(f"step       : {payload['step']}")
    if payload.get("extra", {}).get("val_loss") is not None:
        print(f"val loss   : {payload['extra']['val_loss']:.4f}")
    print("=" * 70)

    if args.interactive:
        run_interactive(model, tokenizer, args, device)
    else:
        run_once(model, tokenizer, args, device)


if __name__ == "__main__":
    main()

"""Inference feature CLI: single-shot and interactive generation."""
from __future__ import annotations

import argparse
import math
import sys

from ...core.text import console_safe
from .service import GenerationRequest, InferenceEngine, Prediction


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="apexgpt generate",
                                 description="Generate text with a trained ApexGPT model")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--prompt", default="The history of the city is")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:N | mps | xpu | dml")
    ap.add_argument("--tokenizer", default=None, choices=["gpt2", "char"],
                    help="override the tokenizer recorded in the checkpoint")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--predict", type=int, default=0, metavar="K",
                    help="print the K most likely next tokens before generating")
    ap.add_argument("--next-word", action="store_true",
                    help="print only the single most likely next word")
    ap.add_argument("--no-cache", action="store_true", help="disable KV caching")
    ap.add_argument("--no-stop", action="store_true", help="ignore the eos token")
    return ap


def _header(engine: InferenceEngine) -> None:
    meta = engine.metadata()
    print("=" * 70)
    print("ApexGPT inference")
    print("=" * 70)
    print(f"checkpoint : {meta['checkpoint']}")
    print(f"device     : {meta['device']}")
    print(f"tokenizer  : {meta['tokenizer']} (vocab {meta['vocab_size']})")
    print(f"parameters : {meta['parameters']:,} ({meta['parameters_m']}M)")
    print(f"config     : L={meta['n_layer']} d={meta['n_embd']} "
          f"H={meta['n_head']} ctx={meta['block_size']} vocab={meta['vocab_size']}")
    print(f"step       : {meta['step']}")
    if meta["val_loss"] is not None:
        print(f"val loss   : {meta['val_loss']:.4f}")
    print("=" * 70)


def _print_predictions(engine: InferenceEngine, prompt: str, top_k: int,
                       temperature: float) -> None:
    """The next-token distribution, ranked, with probabilities."""
    try:
        rows, entropy = engine.predict_next_with_entropy(prompt, top_k, temperature)
    except RuntimeError as exc:
        print(f"[predict] {exc}", file=sys.stderr)
        return
    if not rows:
        print("[predict] the model returned nothing")
        return
    print(f"[predict] next token after {prompt!r}")
    width = max(len(r.label) for r in rows)
    for row in rows:
        bar = "#" * int(round(row.probability * 40))
        print(f"   {row.rank:>2}. id={row.token_id:<6} p={row.probability:>7.4f} "
              f"logp={row.logprob:>7.3f}  {row.label:<{width}}  {bar}")
    share = sum(r.probability for r in rows)
    vocab = engine.metadata().get("vocab_size") or 0
    print(f"        top-{len(rows)} hold {share:.4f} of the mass; entropy "
          f"{entropy:.3f} nats (uniform would be "
          f"{math.log(vocab):.3f} over {vocab:,} tokens)"
          if vocab else
          f"        top-{len(rows)} hold {share:.4f} of the mass; "
          f"entropy {entropy:.3f} nats")


def _run_once(engine: InferenceEngine, args) -> None:
    req = GenerationRequest(
        prompt=args.prompt, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, seed=args.seed,
        use_cache=not args.no_cache, stop_on_eos=not args.no_stop,
    )
    print(f"\nprompt : {req.prompt!r}")
    print(f"params : temperature={req.temperature} top_k={req.top_k} "
          f"top_p={req.top_p} max_new_tokens={req.max_new_tokens}")
    if args.predict > 0:
        _print_predictions(engine, req.prompt, args.predict, 1.0)
    if args.next_word:
        word = Prediction(1, 0, engine.predict_next_text(req.prompt), 0.0, 0.0)
        print(f"[next]   most likely next word: {word.label}")
    print("-" * 70)
    print(console_safe(req.prompt), end="", flush=True)
    import time
    t0 = time.time()
    count = 0
    for piece in engine.stream(req):
        print(console_safe(piece), end="", flush=True)
        count += 1
    dt = time.time() - t0
    print(f"\n[gen] {count} tokens in {dt:.2f}s ({count / max(dt, 1e-9):.2f} tok/s)")
    print("-" * 70)


def _run_interactive(engine: InferenceEngine, args) -> None:
    print("\nInteractive mode. Commands: :t <temp> :k <k> :p <p> :n <tokens> "
          ":seed <n> :pred [k] :quit")
    while True:
        try:
            line = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith(":"):
            parts = line[1:].split()
            cmd, val = parts[0], (parts[1] if len(parts) > 1 else None)
            if cmd in ("q", "quit", "exit"):
                return
            try:
                if cmd == "t": args.temperature = float(val)
                elif cmd == "k": args.top_k = int(val)
                elif cmd == "p": args.top_p = float(val)
                elif cmd == "n": args.max_new_tokens = int(val)
                elif cmd == "seed": args.seed = int(val)
                else:
                    print("unknown command")
            except (ValueError, TypeError):
                print("bad value")
            continue
        try:
            _print_predictions(engine, line, args.predict or 5, 1.0)
            req = GenerationRequest(
                prompt=line, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k or None,
                top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                seed=args.seed, use_cache=not args.no_cache,
                stop_on_eos=not args.no_stop)
            for piece in engine.stream(req):
                print(console_safe(piece), end="", flush=True)
            print()
        except ValueError as exc:
            print(f"[error] {exc}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        engine = InferenceEngine(device=args.device, tokenizer_kind=args.tokenizer)
        engine.load(args.checkpoint)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    _header(engine)
    if args.interactive:
        _run_interactive(engine, args)
    else:
        _run_once(engine, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
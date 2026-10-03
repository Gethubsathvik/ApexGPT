"""Inference feature CLI: single-shot and interactive generation."""
from __future__ import annotations

import argparse
import sys

from .service import GenerationRequest, InferenceEngine


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tinyllm generate",
                                 description="Generate text with a trained TinyLLM model")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--prompt", default="The history of the city is")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="disable KV caching")
    ap.add_argument("--no-stop", action="store_true", help="ignore the eos token")
    return ap


def _header(engine: InferenceEngine) -> None:
    meta = engine.metadata()
    print("=" * 70)
    print("TinyLLM inference")
    print("=" * 70)
    print(f"checkpoint : {meta['checkpoint']}")
    print(f"device     : {meta['device']}")
    print(f"parameters : {meta['parameters']:,} ({meta['parameters_m']}M)")
    print(f"config     : L={meta['n_layer']} d={meta['n_embd']} "
          f"H={meta['n_head']} ctx={meta['block_size']} vocab={meta['vocab_size']}")
    print(f"step       : {meta['step']}")
    if meta["val_loss"] is not None:
        print(f"val loss   : {meta['val_loss']:.4f}")
    print("=" * 70)


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
    print("-" * 70)
    print(req.prompt, end="", flush=True)
    import time
    t0 = time.time()
    count = 0
    for piece in engine.stream(req):
        print(piece, end="", flush=True)
        count += 1
    dt = time.time() - t0
    print(f"\n[gen] {count} tokens in {dt:.2f}s ({count / max(dt, 1e-9):.2f} tok/s)")
    print("-" * 70)


def _run_interactive(engine: InferenceEngine, args) -> None:
    print("\nInteractive mode. Commands: :t <temp> :k <k> :p <p> :n <tokens> "
          ":seed <n> :quit")
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
            req = GenerationRequest(
                prompt=line, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k or None,
                top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                seed=args.seed, use_cache=not args.no_cache,
                stop_on_eos=not args.no_stop)
            for piece in engine.stream(req):
                print(piece, end="", flush=True)
            print()
        except ValueError as exc:
            print(f"[error] {exc}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        engine = InferenceEngine(device=args.device).load(args.checkpoint)
    except (FileNotFoundError, RuntimeError) as exc:
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
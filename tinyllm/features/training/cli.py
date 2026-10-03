"""Training feature CLI."""
from __future__ import annotations

import argparse
import sys

from ...core.config import Config
from ...core.paths import RUNS_DIR
from ...core.seeding import set_seed
from .service import PRESETS, show_history, train


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tinyllm train",
                                 description="Train the TinyLLM GPT model")
    ap.add_argument("--preset", default="auto",
                    help="'auto' picks a preset that fits this machine, or name "
                         "one explicitly: " + ", ".join(sorted(PRESETS)))
    ap.add_argument("--max-iters", type=int, default=None,
                    help="step budget (overrides --epochs)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="derive the step count from epochs x corpus size")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--eval-iters", type=int, default=None)
    ap.add_argument("--checkpoint-interval", type=int, default=None)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--dataset", default=None,
                    help="corpus to train on (default: the TINYLLM_DATASET env "
                         "var, else wikipedia). Build it first with: "
                         "python -m tinyllm data prepare --source <name>")
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:N | mps | xpu | dml")
    ap.add_argument("--num-threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--amp", dest="use_amp", action="store_true", default=None,
                    help="force mixed precision on (default: CUDA only)")
    ap.add_argument("--no-amp", dest="use_amp", action="store_false")
    ap.add_argument("--checkpointing", dest="use_checkpointing",
                    action="store_true", default=None,
                    help="force gradient checkpointing on (default: CUDA only)")
    ap.add_argument("--no-checkpointing", dest="use_checkpointing", action="store_false")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--summary-only", action="store_true",
                    help="print the model summary and exit")
    ap.add_argument("--show-history", metavar="RUN", default=None,
                    help="print the loss table for a run directory and exit")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.show_history:
        return show_history(RUNS_DIR / args.show_history)

    cfg = Config()
    set_seed(cfg.train.seed)

    if args.dataset:
        cfg.data.select_dataset(args.dataset)
    print(f"[data]    corpus '{cfg.data.dataset}' -> {cfg.data.binary_dir}")

    # 'auto' sizes the run to the detected hardware
    from ...core.device import profile
    hardware = profile()
    if args.preset == "auto":
        args.preset = hardware.recommend_preset()
        print(f"[auto]    {hardware.describe()}")
        print(f"[auto]    selected preset '{args.preset}'")

    if args.max_iters is not None and args.epochs is not None:
        print("[warn] both --max-iters and --epochs given; --max-iters wins")
        args.epochs = None

    try:
        train(cfg,
              preset=args.preset,
              max_iters=args.max_iters,
              epochs=args.epochs,
              batch_size=args.batch_size,
              block_size=args.block_size,
              learning_rate=args.learning_rate,
              eval_interval=args.eval_interval,
              eval_iters=args.eval_iters,
              checkpoint_interval=args.checkpoint_interval,
              log_interval=args.log_interval,
              run_name=args.run_name,
              use_amp=args.use_amp,
              use_checkpointing=args.use_checkpointing,
              resume=args.resume,
              device_spec=args.device,
              num_threads=args.num_threads,
              seed=args.seed,
              summary_only=args.summary_only)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
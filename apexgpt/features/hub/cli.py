"""Hub feature CLI: Hugging Face models and datasets, Kaggle datasets."""
from __future__ import annotations

import argparse
import sys

from ...core.text import console_safe
from .service import (DATASET_REPO, MODEL_REPO, HubError, check, download_dataset,
                      download_kaggle_dataset, download_model, environment,
                      list_files, run_model)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="apexgpt hub",
        description="Hugging Face and Kaggle: download models and datasets, "
                    "and run a pretrained model")
    sub = ap.add_subparsers(dest="action", required=True)

    sub.add_parser("check", help="what this machine can do with the Hub")

    files = sub.add_parser("files", help="list the files in a repository")
    files.add_argument("repo_id", help="e.g. gpt2 or Salesforce/wikitext")
    files.add_argument("--dataset", action="store_true",
                       help="treat repo_id as a dataset repo (default: model)")
    files.add_argument("--revision", default=None)

    model = sub.add_parser("model", help="download a model repo, optionally run it")
    model.add_argument("repo_id")
    model.add_argument("--revision", default=None)
    model.add_argument("--dest", default=None, help="directory to download into")
    model.add_argument("--allow", nargs="*", default=None,
                       help="glob patterns, e.g. '*.json' '*.safetensors'")
    model.add_argument("--predict", default=None, metavar="PROMPT",
                       help="download, then print what the model predicts next")
    model.add_argument("--top-k", type=int, default=10)
    model.add_argument("--max-new-tokens", type=int, default=0,
                       help="also generate this many tokens after the prompt")
    model.add_argument("--temperature", type=float, default=0.8)
    model.add_argument("--device", default="auto",
                       help="auto | cpu | cuda | cuda:N | mps | xpu | dml")

    dataset = sub.add_parser("dataset", help="download a dataset repo's files")
    dataset.add_argument("repo_id")
    dataset.add_argument("--revision", default=None)
    dataset.add_argument("--dest", default=None)
    dataset.add_argument("--allow", nargs="*", default=None)

    kaggle = sub.add_parser("kaggle", help="download a Kaggle dataset")
    kaggle.add_argument("slug", help="owner/dataset-slug")
    kaggle.add_argument("--dest", default=None)
    return ap


def _print_predictions(rows, tokenizer_note: str = "") -> None:
    width = max((len(r.text or repr(r.text)) for r in rows), default=1)
    for row in rows:
        shown = (row.text or repr(row.text)).replace(" ", "\\u0020") \
                                            .replace("\n", "\\n")
        bar = "#" * int(round(row.probability * 40))
        print(f"   {row.rank:>2}. id={row.token_id:<8} p={row.probability:>9.4g} "
              f"{shown:<{width}}  {bar}")
    print(f"        {tokenizer_note}" if tokenizer_note else "")


def _run_check() -> int:
    env = environment()
    print("=" * 70)
    print("ApexGPT hub")
    print("=" * 70)
    print(f"  models   : {env['model_dir']}")
    print(f"  datasets : {env['dataset_dir']}")
    print()
    for label, ok, detail in check():
        print(f"  [{'ok' if ok else '--'}] {label:<42} {detail}")
    print()
    print("Reference training scripts (not vendored here):")
    print("  transformers  examples/pytorch/language-modeling/run_clm.py"
          "  -> features/training/service.py")
    print("  trl           SFTTrainer                                  "
          "-> features/training/service.py")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "check":
            return _run_check()

        if args.action == "files":
            repo_type = DATASET_REPO if args.dataset else MODEL_REPO
            files = list_files(args.repo_id, repo_type=repo_type,
                               revision=args.revision)
            total = sum(f.size_bytes or 0 for f in files)
            print(f"{args.repo_id} ({repo_type}) - {len(files)} files, "
                  f"{total / 1e6:.1f} MB")
            for f in files:
                size = "" if f.size_mb is None else f"  {f.size_mb:>9.2f} MB"
                print(f"  {f.name}{size}")
            return 0

        if args.action == "model":
            if args.predict is not None:
                rows, text = run_model(
                    args.repo_id, args.predict, top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, device=args.device,
                    revision=args.revision, allow_patterns=args.allow)
                print(f"\nprompt : {args.predict!r}")
                print(f"[next]  {args.repo_id} predicts:")
                _print_predictions(rows)
                if text:
                    print("-" * 70)
                    print(console_safe(args.predict + text))
                    print("-" * 70)
            else:
                path = download_model(args.repo_id, revision=args.revision,
                                      allow_patterns=args.allow, dest=args.dest)
                print(f"[hub]     downloaded to {path}")
                print("           (config, tokenizer and weights only; --allow "
                      "changes that)")
                print(f"           run it:  python -m apexgpt hub model "
                      f"{args.repo_id} --predict \"your prompt\"")
            return 0

        if args.action == "dataset":
            path = download_dataset(args.repo_id, revision=args.revision,
                                    allow_patterns=args.allow, dest=args.dest)
            print(f"[hub]     downloaded to {path}")
            print(f"           turn it into a corpus:  "
                  f"python -m apexgpt data prepare --source local:<file>")
            return 0

        if args.action == "kaggle":
            path = download_kaggle_dataset(args.slug, dest=args.dest)
            print(f"[kaggle]  downloaded to {path}")
            print(f"           train on it:  python -m apexgpt data prepare "
                  f"--source kaggle:{args.slug}")
            return 0
    except HubError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
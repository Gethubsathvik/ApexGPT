"""``python -m tinyllm env`` - scan the machine, collect the requirements, apply the settings.

One command that answers "can this machine run TinyLLM, and with what settings?":

    python -m tinyllm env                 # scan, report, and apply in this process
    python -m tinyllm env --json          # machine-readable report
    python -m tinyllm env --export        # shell lines that carry the settings out
    python -m tinyllm env --check         # exit 1 if a required package is missing
    python -m tinyllm env --device cuda   # resolve settings for a specific backend
"""
from __future__ import annotations

import argparse
import sys

from ..core.environment import (OPTIONAL_GROUPS, apply_settings, scan,
                                shell_exports, write_report)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tinyllm env",
        description="Detect this machine's hardware, collect the requirements, "
                    "and resolve the settings TinyLLM will run with")
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:N | mps | xpu | dml")
    ap.add_argument("--json", action="store_true",
                    help="print the report as JSON and nothing else")
    ap.add_argument("--export", action="store_true",
                    help="print shell export lines for the resolved settings")
    ap.add_argument("--shell", default="auto",
                    choices=["auto", "bash", "powershell", "cmd"],
                    help="shell dialect for --export (default: auto-detect)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when a required package is missing or outdated")
    ap.add_argument("--groups", default="core",
                    help="requirement groups to collect, comma separated: "
                         + ",".join(("core",) + OPTIONAL_GROUPS))
    ap.add_argument("--save", metavar="FILE", default=None,
                    help="also write the JSON report to FILE")
    ap.add_argument("--no-apply", action="store_true",
                    help="report only; do not configure threads or export vars")
    return ap


def _next_steps(report) -> list[str]:
    acc = report.accelerator
    steps = []
    if not acc.backends:
        steps.append("No GPU backend: training will use the CPU. On an NVIDIA box, "
                     "run 'python -m tinyllm setup --install' to get the CUDA wheel.")
        if acc.adapters:
            steps.append("An adapter is present but unused - AMD/Intel GPUs need ROCm "
                         "(Linux) or 'python -m tinyllm setup --backend dml --install' "
                         "(Windows, experimental).")
    if not report.ok:
        steps.append("Install what is missing: python -m tinyllm setup --install")
    steps += [
        "Verify the install:      python -m tinyllm doctor",
        "Build the corpus:        python -m tinyllm data prepare",
        f"Train ({report.settings.preset} preset): python -m tinyllm train",
        "Open Jupyter Lab:        python -m tinyllm lab",
    ]
    return steps


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip()) or ("core",)

    try:
        report = scan(device=args.device, groups=groups)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(report.to_json())
    else:
        print(report.to_text())
        for step in _next_steps(report):
            print(f"  -> {step}")
        print()

    if not args.no_apply:
        apply_settings(report.settings)

    if args.export:
        for line in shell_exports(report.settings, args.shell):
            print(line)

    if args.save:
        path = write_report(report, args.save)
        print(f"[saved] environment report -> {path}")

    if args.check and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

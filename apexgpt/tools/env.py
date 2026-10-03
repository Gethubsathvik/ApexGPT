"""``python -m apexgpt env`` - scan the machine, collect the requirements, apply the settings.

One command that answers "can this machine run ApexGPT, and with what settings?":

    python -m apexgpt env                 # scan, report, and apply in this process
    python -m apexgpt env --json          # machine-readable report
    python -m apexgpt env --export        # shell lines that carry the settings out
    python -m apexgpt env --check         # exit 1 if a required package is missing
    python -m apexgpt env --device cuda   # resolve settings for a specific backend
    python -m apexgpt env --threads 4 --save-settings   # pin, and keep the choice

Automatic by default, customisable when the automatic answer is wrong: any
setting can be overridden for this run with a flag, for the session with a
``APEXGPT_SET_*`` variable, or for good with ``--save-settings``, which writes
``apexgpt.settings.json``.
"""
from __future__ import annotations

import argparse
import sys

from ..core.environment import (DEFAULT_MAX_CPU_PERCENT, DEFAULT_RESERVE_RAM_GB,
                                OPTIONAL_GROUPS, SETTING_KEYS, SettingsOverrides,
                                apply_settings, load_overrides, save_overrides,
                                scan, shell_exports, write_report)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="apexgpt env",
        description="Detect this machine's hardware, collect the requirements, "
                    "and resolve the settings ApexGPT will run with")
    ap.add_argument("--device", default=None,
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

    tuning = ap.add_argument_group(
        "overrides",
        "customise the automatic settings; each one wins over the measurement. "
        f"Defaults: reserve_ram_gb={DEFAULT_RESERVE_RAM_GB}, "
        f"max_cpu_percent={DEFAULT_MAX_CPU_PERCENT}. Environment equivalents are "
        + ", ".join(sorted(set(SETTING_KEYS.values()))))
    tuning.add_argument("--threads", type=int, default=None,
                        help="torch CPU threads (default: every core)")
    tuning.add_argument("--preset", default=None,
                        help="training preset to recommend (default: sized to fit)")
    tuning.add_argument("--batch-size", type=int, default=None)
    tuning.add_argument("--block-size", type=int, default=None)
    tuning.add_argument("--tokenizer", default=None,
                        help="gpt2 | char - which tokenizer the corpus is built with")
    tuning.add_argument("--amp", dest="amp", action="store_true", default=None,
                        help="force mixed precision on")
    tuning.add_argument("--no-amp", dest="amp", action="store_false")
    tuning.add_argument("--checkpointing", dest="checkpointing",
                        action="store_true", default=None)
    tuning.add_argument("--no-checkpointing", dest="checkpointing",
                        action="store_false")
    tuning.add_argument("--reserve-ram-gb", type=float, default=None,
                        help="RAM left for the OS when adjusting the batch")
    tuning.add_argument("--max-cpu-percent", type=float, default=None,
                        help="CPU utilisation above which a thread is handed back")
    tuning.add_argument("--save-settings", action="store_true",
                        help="write these overrides to apexgpt.settings.json")
    tuning.add_argument("--reset-settings", action="store_true",
                        help="clear apexgpt.settings.json and start from auto")

    ap.add_argument("--no-apply", action="store_true",
                    help="report only; do not configure threads or export vars")
    ap.add_argument("--no-load-scan", action="store_true",
                    help="skip the live CPU/RAM/process reading")
    return ap


def overrides_from_args(args) -> SettingsOverrides:
    """Merge the settings file, the ``APEXGPT_SET_*`` variables and the flags."""
    base = load_overrides()
    origin = dict(base.origin)
    values = base.active()
    for key in SettingsOverrides.KEYS:
        flag = getattr(args, key, None)
        if flag is not None:
            values[key] = flag
            origin[key] = "flag"
    return SettingsOverrides(origin=origin, **values)


def _next_steps(report) -> list[str]:
    acc = report.accelerator
    s = report.settings
    steps = []
    if not acc.backends:
        steps.append("No GPU backend: training will use the CPU. On an NVIDIA box, "
                     "run 'python -m apexgpt setup --install' to get the CUDA wheel.")
        if acc.adapters:
            steps.append("An adapter is present but unused - AMD/Intel GPUs need ROCm "
                         "(Linux) or 'python -m apexgpt setup --backend dml --install' "
                         "(Windows, experimental).")
    if not report.ok:
        steps.append("Install what is missing: python -m apexgpt setup --install")
    if s.notes:
        steps.append("Settings were adjusted for the current load - see the notes above")
    steps += [
        "Verify the install:      python -m apexgpt doctor",
        f"Build a corpus:          python -m apexgpt data prepare --source shakespeare",
        f"Train ({s.preset} preset): python -m apexgpt train --preset {s.preset}",
        "Open Jupyter Lab:        python -m apexgpt lab",
        "Change these settings:   python -m apexgpt env --threads 4 --save-settings",
    ]
    return steps


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.reset_settings:
        from ..core.environment import SETTINGS_FILE
        SETTINGS_FILE.unlink(missing_ok=True)
        print(f"[reset]  removed {SETTINGS_FILE.name}; everything is automatic again")

    overrides = overrides_from_args(args)
    if args.save_settings:
        path = save_overrides(overrides)
        print(f"[saved]  overrides -> {path}")

    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip()) or ("core",)

    try:
        report = scan(device=args.device, groups=groups, overrides=overrides,
                      measure_load=not args.no_load_scan)
    except (RuntimeError, ValueError) as exc:
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
"""TinyLLM command line entry point.

    python -m tinyllm env                  # scan hardware, collect requirements
    python -m tinyllm setup                # install dependencies for this machine
    python -m tinyllm doctor               # verify the environment
    python -m tinyllm data prepare
    python -m tinyllm train --preset cpu-tiny
    python -m tinyllm generate --prompt "hello"
    python -m tinyllm gui
    python -m tinyllm lab                  # Jupyter Lab
    python -m tinyllm serve --port 8000
"""
from __future__ import annotations

import argparse
import sys

COMMANDS = {
    "env":      ("tinyllm.tools.env", "scan this machine and apply its settings"),
    "setup":    ("tinyllm.tools.setup", "install dependencies for this machine"),
    "doctor":   ("tinyllm.tools.doctor", "verify the environment"),
    "data":     ("tinyllm.features.data.cli",
                 "build a corpus: shakespeare, wikipedia, Hugging Face, Kaggle"),
    "train":    ("tinyllm.features.training.cli", "train the model"),
    "generate": ("tinyllm.features.inference.cli", "generate text from a checkpoint"),
    "gui":      ("tinyllm.features.inference.gui", "desktop GUI"),
    "lab":      ("tinyllm.tools.lab", "open Jupyter Lab with a configured kernel"),
    "serve":    ("tinyllm.api.server", "run the HTTP inference API"),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("TinyLLM\n")
        print("usage: python -m tinyllm <command> [options]\n")
        print("commands:")
        for name, (_, help_text) in COMMANDS.items():
            print(f"  {name:<10} {help_text}")
        print("\nrun 'python -m tinyllm <command> --help' for command options")
        return 0

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"unknown command: {command}", file=sys.stderr)
        print(f"available: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2

    module_name, _ = COMMANDS[command]
    module = __import__(module_name, fromlist=["main"])
    return module.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
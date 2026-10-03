"""ApexGPT command line entry point.

    python -m apexgpt env                  # scan hardware, collect requirements
    python -m apexgpt setup                # install dependencies for this machine
    python -m apexgpt doctor               # verify the environment
    python -m apexgpt hub check            # Hugging Face / Kaggle capability
    python -m apexgpt data prepare
    python -m apexgpt train --preset cpu-tiny
    python -m apexgpt generate --prompt "hello"
    python -m apexgpt generate --prompt "hello" --predict 10
    python -m apexgpt gui
    python -m apexgpt lab                  # Jupyter Lab
    python -m apexgpt serve --port 8000
"""
from __future__ import annotations

import argparse
import sys

COMMANDS = {
    "env":      ("apexgpt.tools.env", "scan this machine and apply its settings"),
    "setup":    ("apexgpt.tools.setup", "install dependencies for this machine"),
    "doctor":   ("apexgpt.tools.doctor", "verify the environment"),
    "data":     ("apexgpt.features.data.cli",
                 "build a corpus, or list and inspect its tokens"),
    "hub":      ("apexgpt.features.hub.cli",
                 "download Hugging Face / Kaggle models and datasets, run one"),
    "train":    ("apexgpt.features.training.cli", "train the model"),
    "generate": ("apexgpt.features.inference.cli", "generate text from a checkpoint"),
    "gui":      ("apexgpt.features.inference.gui", "desktop GUI"),
    "lab":      ("apexgpt.tools.lab", "open Jupyter Lab with a configured kernel"),
    "serve":    ("apexgpt.api.server", "run the HTTP inference API"),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("ApexGPT\n")
        print("usage: python -m apexgpt <command> [options]\n")
        print("commands:")
        for name, (_, help_text) in COMMANDS.items():
            print(f"  {name:<10} {help_text}")
        print("\nrun 'python -m apexgpt <command> --help' for command options")
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
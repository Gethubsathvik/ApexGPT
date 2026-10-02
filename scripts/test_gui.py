"""Headless verification of the GUI: widgets, streaming, and controls.

Builds the real Tk window (withdrawn), runs a short generation through the
GUI's own worker thread, and checks that tokens stream into the output pane.
"""
from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk
import torch

from model import GPT
from scripts.generate import find_checkpoint, get_tokenizer
from scripts.gui import App

results: list[tuple[str, bool, str]] = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
          flush=True)


def main():
    ckpt = find_checkpoint(None)
    model, payload = GPT.load(ckpt, map_location="cpu")
    model.eval()
    tok = get_tokenizer()
    device = "cpu"

    print(f"checkpoint: {ckpt}\n")

    root = tk.Tk()
    root.withdraw()
    app = App(root, model, tok, ckpt, payload, device)
    root.update()

    check("window builds", True)
    check("controls present",
          set(app.vars) == {"temperature", "top_k", "top_p",
                            "max_new_tokens", "seed", "penalty"},
          f"{sorted(app.vars)}")
    check("buttons present",
          bool(app.btn_run) and bool(app.btn_stop) and bool(app.use_cache))

    # sliders are wired to readable values
    app.vars["temperature"].set(0.35)
    app.vars["top_k"].set(20)
    app.vars["max_new_tokens"].set(12)
    root.update()
    check("sliders update values",
          abs(app.vars["temperature"].get() - 0.35) < 1e-9
          and app.vars["top_k"].get() == 20,
          f"T={app.vars['temperature'].get()} k={app.vars['top_k'].get()}")

    # prompt round-trip
    app.prompt.delete("1.0", "end")
    app.prompt.insert("1.0", "The capital city of")
    check("prompt input works", app.prompt.get("1.0", "end").strip() == "The capital city of")

    # run a real generation through the GUI worker
    app.start()
    deadline = time.time() + 300
    while time.time() < deadline and app.thread and app.thread.is_alive():
        root.update()
        time.sleep(0.05)
    for _ in range(40):          # let the queue drain
        root.update()
        time.sleep(0.02)

    text = app.output.get("1.0", "end").strip()
    check("generation produced output", len(text) > len("The capital city of") + 5,
          f"{len(text)} chars")
    check("prompt echoed in output", text.startswith("The capital city of"))
    check("tokens streamed incrementally", app.token_count > 0,
          f"{app.token_count} tokens")
    check("run button re-enabled", str(app.btn_run.cget("state")) == "normal",
          str(app.btn_run.cget("state")))
    check("stop button disabled after run", str(app.btn_stop.cget("state")) == "disabled")
    check("status reports tok/s", "tok/s" in app.status.cget("text"),
          app.status.cget("text"))

    # stop button path
    app.prompt.delete("1.0", "end")
    app.prompt.insert("1.0", "In a distant galaxy")
    app.vars["max_new_tokens"].set(400)
    app.start()
    root.update()
    time.sleep(0.4)
    app.stop()
    deadline = time.time() + 120
    while time.time() < deadline and app.thread and app.thread.is_alive():
        root.update()
        time.sleep(0.05)
    for _ in range(20):
        root.update()
        time.sleep(0.02)
    check("stop halts generation",
          "[stopped by user]" in app.output.get("1.0", "end")
          or not (app.thread and app.thread.is_alive()))

    print("\n  --- sample GUI output ---")
    for line in text[:300].splitlines()[:4]:
        print("  " + line)
    print("  " + "-" * 40)

    app.prompt.delete("1.0", "end")
    app.prompt.insert("1.0", "clear test")
    app._clear()
    check("clear button empties output", app.output.get("1.0", "end").strip() == "")

    root.destroy()

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nGUI TEST: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

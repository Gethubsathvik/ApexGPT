"""Tkinter desktop GUI for the trained model.

The view owns no model logic: generation runs through
:class:`~apexgpt.features.inference.service.InferenceEngine` on a background
thread, so the window stays responsive and the sampling behaviour is identical
to the CLI and the HTTP API.
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk

from ...core import describe_device, get_device
from .service import GenerationRequest, InferenceEngine

BG = "#1e1e2e"
FG = "#e6e6f0"
ACCENT = "#89b4fa"
MUTED = "#a6adc8"
OK = "#a6e3a1"
ERR = "#f38ba8"
PANEL = "#181825"
INPUT_BG = "#313244"
OUTPUT_BG = "#11111b"


def format_token_values(report: dict, max_rows: int = 12) -> str:
    """Render the token table and the next-word ranking for the text bar.

    A pure function of the report, so the formatting can be checked without a
    window and so every view prints the same numbers.
    """
    corpus = report.get("corpus") or {}
    ids = report.get("token_ids") or []
    rows = report.get("tokens") or []
    has_counts = bool(corpus.get("path"))

    if has_counts:
        header = (f"{len(ids)} id(s)  |  {Path(corpus['path']).name}: "
                  f"{corpus['tokens']:,} tokens, {corpus['distinct']:,} distinct"
                  + ("" if corpus.get("ready", True) else "  (counting)"))
    else:
        header = f"{len(ids)} id(s)  |  no corpus on disk: ids only"

    lines = [header]
    for row in rows[:max_rows]:
        if has_counts:
            lines.append(f"  {row['token_id']:>6}  {row['label']:<16} "
                         f"{row['count']:>9,}  {row['share'] * 100:>6.3f}%")
        else:
            lines.append(f"  {row['token_id']:>6}  {row['label']}")
    if len(rows) > max_rows:
        lines.append(f"  ... {len(rows) - max_rows} more id(s)")

    predictions = report.get("predictions") or []
    if predictions:
        ranked = "  ".join(f"{p['label']} {p['probability']:.1%}"
                           for p in predictions[:5])
        entropy = report.get("entropy")
        suffix = f"   entropy {entropy:.2f} nats" if entropy is not None else ""
        lines.append(f"next:  {ranked}{suffix}")
    return "\n".join(lines) + "\n"


class ApexGPTApp:
    """Main window. All generation happens on a worker thread."""

    def __init__(self, root: tk.Tk, engine: InferenceEngine):
        self.root = root
        self.engine = engine
        self.events: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None
        self.token_thread: threading.Thread | None = None
        self.token_after: str | None = None
        self._tokens_wanted = ""
        self._tokens_done = ""

        self._build()
        self._describe()
        self._set_tokens("the token values of the prompt appear here\n")
        self.root.after(60, self._drain)
        self.root.after(400, self._refresh_tokens)

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        r = self.root
        r.title("ApexGPT - Language Model Playground")
        r.geometry("1120x860")
        r.configure(bg=BG)
        r.minsize(920, 680)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Head.TLabel", background=BG, foreground=ACCENT,
                        font=("Segoe UI", 15, "bold"))
        style.configure("Stat.TLabel", background=BG, foreground=OK, font=("Consolas", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Horizontal.TScale", background=BG)

        head = ttk.Frame(r, padding=(14, 10, 14, 0))
        head.pack(fill="x")
        ttk.Label(head, text="ApexGPT", style="Head.TLabel").pack(side="left")
        self.meta_label = ttk.Label(head, text="", style="Muted.TLabel")
        self.meta_label.pack(side="left", padx=12)

        ctrl = ttk.Frame(r, padding=(14, 10, 14, 6))
        ctrl.pack(fill="x")
        ctrl.columnconfigure(0, weight=1)

        self.prompt = tk.Text(ctrl, height=2, bg=INPUT_BG, fg=FG,
                              insertbackground=FG, relief="flat", font=("Consolas", 11),
                              wrap="word", insertwidth=2, borderwidth=0,
                              padx=8, pady=6)
        self.prompt.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.prompt.insert("1.0", "The history of the city is")
        self.prompt.bind("<<Modified>>", self._prompt_changed)

        # The token values of whatever is in the prompt bar, and what the model
        # expects to put after it: the same table `apexgpt data tokens` prints.
        self.token_bar = tk.Text(ctrl, height=7, bg=PANEL, fg=FG, relief="flat",
                                 font=("Consolas", 9), wrap="none", padx=8,
                                 pady=6, insertbackground=FG, state="disabled",
                                 borderwidth=0)
        self.token_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.token_bar.tag_configure("head", foreground=ACCENT,
                                     font=("Consolas", 9, "bold"))
        self.token_bar.tag_configure("count", foreground=MUTED)
        self.token_bar.tag_configure("next", foreground=OK)

        grid = ttk.Frame(ctrl)
        grid.grid(row=2, column=0, sticky="ew")
        for col in (1, 3, 5):
            grid.columnconfigure(col, weight=1)

        self.vars: dict[str, tk.DoubleVar] = {}
        self.vars["temperature"] = self._slider(grid, 0, "temperature", 0.0, 2.0, 0.8, 0.05, "%.2f")
        self.vars["top_k"] = self._slider(grid, 2, "top-k", 0, 200, 50, 1, "%d")
        self.vars["top_p"] = self._slider(grid, 4, "top-p", 0.0, 1.0, 0.95, 0.01, "%.2f")
        self.vars["max_new_tokens"] = self._slider(grid, 0, "max tokens", 10, 1024, 200, 10, "%d")
        self.vars["seed"] = self._slider(grid, 2, "seed", 0, 9999, 0, 1, "%d")
        self.vars["penalty"] = self._slider(grid, 4, "repetition penalty", 1.0, 2.0, 1.0, 0.05, "%.2f")

        bar = ttk.Frame(ctrl, padding=(0, 10, 0, 0))
        bar.grid(row=3, column=0, sticky="ew")
        self.btn_run = ttk.Button(bar, text="Generate", style="Run.TButton", command=self.start)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left")
        self.use_cache = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="KV cache", variable=self.use_cache).pack(side="left", padx=14)
        self.predict_next = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show next token",
                        variable=self.predict_next).pack(side="left")
        ttk.Button(bar, text="Quit", command=self.root.destroy).pack(side="right")
        self.status = ttk.Label(bar, text="ready", style="Stat.TLabel")
        self.status.pack(side="right", padx=10)

        out = ttk.Frame(r, padding=(14, 4, 14, 12))
        out.pack(fill="both", expand=True)
        self.output = tk.Text(out, bg=OUTPUT_BG, fg=FG, relief="flat",
                              font=("Consolas", 11), wrap="word", padx=10, pady=8,
                              insertbackground=FG, state="disabled",
                              spacing1=2, spacing3=2)
        scroll = ttk.Scrollbar(out, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)
        self.output.tag_configure("prompt", foreground=ACCENT)
        self.output.tag_configure("meta", foreground=MUTED, font=("Consolas", 9))
        self.output.tag_configure("error", foreground=ERR)

    def _slider(self, parent, col, label, lo, hi, value, _step, fmt):
        frame = ttk.Frame(parent, padding=(0, 0, 14, 0))
        frame.grid(row=0, column=col, sticky="ew")
        head = ttk.Frame(frame)
        head.pack(fill="x")
        ttk.Label(head, text=label, font=("Segoe UI", 9)).pack(side="left")
        readout = ttk.Label(head, text=fmt % value, style="Muted.TLabel", width=6)
        readout.pack(side="right")

        var = tk.DoubleVar(value=value)
        var.trace_add("write", lambda *_: readout.configure(text=fmt % var.get()))
        ttk.Scale(frame, from_=lo, to=hi, variable=var, orient="horizontal").pack(fill="x")
        return var

    def _describe(self) -> None:
        try:
            meta = self.engine.metadata()
            self.meta_label.configure(
                text=f"{meta['parameters_m']}M params  |  L={meta['n_layer']} "
                     f"d={meta['n_embd']} H={meta['n_head']} ctx={meta['block_size']}  |  "
                     f"{meta['device']}")
        except Exception:
            self.meta_label.configure(text="")

    # -------------------------------------------------------------- behaviour
    def clear(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        self.status.configure(text="ready")

    def _echo(self, text: str, tag: str | None = None) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text, tag)
        self.output.see("end")
        self.output.configure(state="disabled")

    # ------------------------------------------------- token values text bar
    def _set_tokens(self, text: str) -> None:
        """Write the token bar. Only ever called on the UI thread."""
        self.token_bar.configure(state="normal")
        self.token_bar.delete("1.0", "end")
        for line in text.rstrip("\n").splitlines() or [""]:
            if line.startswith("next:"):
                self.token_bar.insert("end", line + "\n", "next")
            elif line.startswith("  ") and not line.strip().startswith("..."):
                self.token_bar.insert("end", line + "\n", "count")
            else:
                self.token_bar.insert("end", line + "\n", "head")
        self.token_bar.configure(state="disabled")

    def _prompt_changed(self, _event=None) -> None:
        if not self.prompt.edit_modified():
            return
        self.prompt.edit_modified(False)
        self._schedule_tokens()

    def _schedule_tokens(self) -> None:
        """Debounce: a keystroke must not start a forward pass."""
        if self.token_after is not None:
            self.root.after_cancel(self.token_after)
        self.token_after = self.root.after(400, self._refresh_tokens)

    def _refresh_tokens(self) -> None:
        prompt = self.prompt.get("1.0", "end").strip()
        self._tokens_wanted = prompt
        if not prompt:
            self._set_tokens("type a prompt to see its token values\n")
            return
        if self.token_thread and self.token_thread.is_alive():
            # Counting a big corpus takes seconds; the result is checked against
            # the prompt again before it is shown, so this is not a dropped
            # update - the newer prompt is picked up the moment the worker ends.
            return
        self._set_tokens("counting the token values...\n")
        self.token_thread = threading.Thread(target=self._token_worker,
                                             args=(prompt,), daemon=True)
        self.token_thread.start()

    def _token_worker(self, prompt: str) -> None:
        try:
            report = self.engine.token_values(prompt, top_k=5)
            self.events.put(("tokens-for", prompt))
            self.events.put(("tokens", format_token_values(report)))
        except Exception as exc:
            self.events.put(("tokens-for", prompt))
            self.events.put(("tokens",
                             f"[token values] {type(exc).__name__}: {exc}\n"))

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            self.status.configure(text="enter a prompt", style="Muted.TLabel")
            return

        seed = int(self.vars["seed"].get())
        request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=int(self.vars["max_new_tokens"].get()),
            temperature=self.vars["temperature"].get(),
            top_k=int(self.vars["top_k"].get()) or None,
            top_p=self.vars["top_p"].get(),
            repetition_penalty=self.vars["penalty"].get(),
            seed=seed if seed > 0 else None,
            # Tcl variables must be read on the UI thread, never in the worker
            use_cache=self.use_cache.get(),
        )
        try:
            request.validate()
        except ValueError as exc:
            self.status.configure(text=str(exc), style="Muted.TLabel")
            return

        self.stop_flag.clear()
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        self._echo(prompt + "\n", "prompt")

        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status.configure(text="generating...")

        # Tcl variables are read here, on the UI thread, and passed to the
        # worker as plain values: touching them from the worker raises
        # "main thread is not in main loop"
        show_prediction = bool(self.predict_next.get())
        self.thread = threading.Thread(target=self._worker,
                                       args=(request, show_prediction),
                                       daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag.set()
        self.status.configure(text="stopping...")

    def _worker(self, request: GenerationRequest, show_prediction: bool = True) -> None:
        import time
        t0 = time.time()
        count = 0
        try:
            if show_prediction:
                # what the model expects before it writes anything; the same
                # table `generate --predict` prints
                rows, entropy = self.engine.predict_next_with_entropy(
                    request.prompt, top_k=5, temperature=1.0)
                vocab = self.engine.metadata().get("vocab_size") or 0
                lines = ["next token: " + "  ".join(
                    f"{r.label} {r.probability:.1%}" for r in rows)]
                lines.append(f"top-{len(rows)} of {vocab:,} tokens, "
                             f"entropy {entropy:.2f} nats "
                             f"(uniform {math.log(vocab):.2f})" if vocab > 1
                             else f"top-{len(rows)}, entropy {entropy:.2f} nats")
                self.events.put(("meta", "\n".join(lines) + "\n"))
            for piece in self.engine.stream(request):
                if self.stop_flag.is_set():
                    self.events.put(("meta", "\n[stopped by user]\n"))
                    break
                count += 1
                self.events.put(("text", piece))
            dt = time.time() - t0
            self.events.put(("done", f"{count} tokens in {dt:.2f}s "
                                     f"({count / max(dt, 1e-9):.2f} tok/s)"))
        except Exception as exc:
            self.events.put(("error", f"\n[error] {type(exc).__name__}: {exc}\n"))
        finally:
            self.events.put(("idle", ""))

    def _drain(self) -> None:
        idle = False
        summary = ""
        while True:
            try:
                kind, text = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "idle":
                idle = True
                continue
            if kind == "tokens-for":
                self._tokens_done = text
                continue
            if kind == "tokens":
                if self._tokens_done != self._tokens_wanted:
                    # the prompt moved on while this was being counted
                    self.token_after = self.root.after(10, self._refresh_tokens)
                else:
                    self._set_tokens(text)
                continue
            if kind == "done":
                summary = text.strip()
            self._echo(text, kind if kind in ("prompt", "meta", "error") else None)
            if kind == "error":
                self.status.configure(text="error", style="Muted.TLabel")

        if idle:
            self.btn_run.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            if not self.status.cget("text").startswith(("error", "stopping")):
                self.status.configure(text=summary or "ready")

        self.root.after(60, self._drain)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="apexgpt gui",
                                 description="ApexGPT desktop GUI")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    try:
        engine = InferenceEngine(device=args.device).load(args.checkpoint)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    root = tk.Tk()
    ApexGPTApp(root, engine)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

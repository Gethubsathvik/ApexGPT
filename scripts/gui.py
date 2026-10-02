"""Tkinter desktop GUI for interacting with the trained TinyLLM model.

Real-time token-by-token generation on a background thread so the window
stays responsive, with live controls for temperature, top-k, top-p, token
count, seed and repetition penalty.

Usage:
    python scripts/gui.py
    python scripts/gui.py --checkpoint models/runs/gpt-cpu-tiny/checkpoint.pt
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import CHECKPOINTS_DIR, RUNS_DIR
from model import GPT
from scripts.generate import get_tokenizer

import tkinter as tk
from tkinter import ttk

BG = "#1e1e2e"
FG = "#e6e6f0"
ACCENT = "#89b4fa"
MUTED = "#a6adc8"
OK = "#a6e3a1"
ERR = "#f38ba8"
PANEL = "#181825"


def find_checkpoint(explicit=None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    if (CHECKPOINTS_DIR / "checkpoint.pt").exists():
        return CHECKPOINTS_DIR / "checkpoint.pt"
    runs = sorted(RUNS_DIR.glob("*/checkpoint.pt"), key=lambda p: p.stat().st_mtime)
    if runs:
        return runs[-1]
    raise SystemExit("no checkpoint found -- train a model first: python scripts/train.py")


class App:
    def __init__(self, root: tk.Tk, model, tokenizer, ckpt_path, payload, device):
        self.root = root
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.ckpt = ckpt_path
        self.payload = payload
        self.q: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None
        self.t0 = 0.0
        self.token_count = 0

        self._build()
        self.root.after(60, self._drain)

    # ------------------------------------------------------------------ UI
    def _build(self):
        r = self.root
        r.title("TinyLLM - Language Model Playground")
        r.geometry("1080x760")
        r.configure(bg=BG)
        r.minsize(880, 620)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Head.TLabel", background=BG, foreground=ACCENT,
                        font=("Segoe UI", 15, "bold"))
        style.configure("Stat.TLabel", background=BG, foreground=OK, font=("Consolas", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Vertical.TScale", background=BG)

        # header
        head = ttk.Frame(r, padding=(14, 10, 14, 0))
        head.pack(fill="x")
        ttk.Label(head, text="TinyLLM", style="Head.TLabel").pack(side="left")
        cfg = self.model.cfg
        ttk.Label(
            head,
            text=f"{self.model.num_parameters() / 1e6:.1f}M params  |  "
                 f"L={cfg.n_layer} d={cfg.n_embd} H={cfg.n_head} ctx={cfg.block_size}  |  {self.device}",
            style="Muted.TLabel",
        ).pack(side="left", padx=12)

        # controls
        ctrl = ttk.Frame(r, padding=(14, 10, 14, 6))
        ctrl.pack(fill="x")

        self.prompt = tk.Text(ctrl, height=3, bg="#313244", fg=FG, insertbackground=FG,
                              relief="flat", font=("Consolas", 11), wrap="word",
                              insertwidth=2, borderwidth=0, padx=8, pady=6)
        self.prompt.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, 8))
        self.prompt.insert("1.0", "The history of the city is")
        ctrl.columnconfigure(0, weight=1)

        self.vars = {}
        ctl = ttk.Frame(ctrl)
        ctl.grid(row=1, column=0, columnspan=6, sticky="ew")
        ctl.columnconfigure(1, weight=1)
        ctl.columnconfigure(3, weight=1)
        ctl.columnconfigure(5, weight=1)

        self.vars["temperature"] = self._slider(ctl, 0, "temperature", 0.0, 2.0, 0.8, 0.05, "%.2f")
        self.vars["top_k"] = self._slider(ctl, 2, "top-k", 0, 200, 50, 1, "%d")
        self.vars["top_p"] = self._slider(ctl, 4, "top-p", 0.0, 1.0, 0.95, 0.01, "%.2f")
        self.vars["max_new_tokens"] = self._slider(ctl, 0, "max tokens", 10, 1024, 200, 10, "%d")
        self.vars["seed"] = self._slider(ctl, 2, "seed", 0, 9999, 0, 1, "%d")
        self.vars["penalty"] = self._slider(ctl, 4, "repetition penalty", 1.0, 2.0, 1.0, 0.05, "%.2f")

        # buttons
        bar = ttk.Frame(ctrl, padding=(0, 10, 0, 0))
        bar.grid(row=2, column=0, columnspan=6, sticky="ew")

        self.btn_run = ttk.Button(bar, text="Generate", style="Run.TButton", command=self.start)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="Clear", command=self._clear).pack(side="left")
        self.use_cache = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="KV cache", variable=self.use_cache).pack(side="left", padx=14)
        ttk.Button(bar, text="Quit", command=self.root.destroy).pack(side="right")

        self.status = ttk.Label(bar, text="ready", style="Stat.TLabel")
        self.status.pack(side="right", padx=10)

        # output
        out = ttk.Frame(r, padding=(14, 4, 14, 12))
        out.pack(fill="both", expand=True)
        self.output = tk.Text(out, bg="#11111b", fg=FG, relief="flat",
                              font=("Consolas", 11), wrap="word", padx=10, pady=8,
                              insertbackground=FG, state="disabled", spacing1=2, spacing3=2)
        sb = ttk.Scrollbar(out, command=self.output.yview)
        self.output.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)
        self.output.tag_configure("prompt", foreground=ACCENT)
        self.output.tag_configure("meta", foreground=MUTED, font=("Consolas", 9))
        self.output.tag_configure("error", foreground=ERR)

    def _slider(self, parent, col, label, lo, hi, val, step, fmt):
        f = ttk.Frame(parent, padding=(0, 0, 14, 0))
        f.grid(row=0, column=col, sticky="ew")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text=label, font=("Segoe UI", 9)).pack(side="left")
        var = tk.DoubleVar(value=val)
        value_lbl = ttk.Label(row, text=fmt % val, style="Muted.TLabel", width=6)
        value_lbl.pack(side="right")

        var.trace_add("write", lambda *_: value_lbl.configure(text=fmt % var.get()))
        sc = ttk.Scale(f, from_=lo, to=hi, variable=var, orient="horizontal")
        sc.pack(fill="x")
        return var

    # ----------------------------------------------------------- generation
    def _clear(self):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        self.status.configure(text="ready")

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            self.status.configure(text="enter a prompt", style="Muted.TLabel")
            return

        p = {k: v.get() for k, v in self.vars.items()}
        p["use_cache"] = self.use_cache.get()   # Tcl vars must be read on the UI thread
        self.stop_flag.clear()
        self.token_count = 0
        self.t0 = time.time()

        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", prompt, "prompt")
        self.output.insert("end", "\n")
        self.output.configure(state="disabled")

        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status.configure(text="generating...")

        self.thread = threading.Thread(
            target=self._worker, args=(prompt, p), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        self.status.configure(text="stopping...")

    def _worker(self, prompt, p):
        q = self.q
        try:
            seed_val = int(p["seed"])
            if seed_val > 0:
                torch.manual_seed(seed_val)

            ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.device)
            if ids.shape[1] > self.model.cfg.block_size:
                ids = ids[:, -self.model.cfg.block_size:]

            use_cache = p["use_cache"]
            kv = self.model.new_kv_caches(1, self.device, torch.float32) if use_cache else None
            temperature = max(p["temperature"], 1e-6)
            top_k = int(p["top_k"]) or None
            top_p = p["top_p"]
            penalty = p["penalty"]

            for _ in range(int(p["max_new_tokens"])):
                if self.stop_flag.is_set():
                    q.put(("meta", "\n[stopped by user]\n"))
                    break

                with torch.no_grad():
                    idx_cond = ids[:, -self.model.cfg.block_size:]
                    logits, _ = self.model(idx_cond, kv_caches=kv)
                    logits = logits[:, -1, :] / temperature

                    if penalty and penalty != 1.0:
                        # repetition penalty over generated context
                        for tok_id in torch.unique(ids[0, -256:]):
                            if logits[0, tok_id] > 0:
                                logits[0, tok_id] /= penalty
                            else:
                                logits[0, tok_id] *= penalty

                    if top_k:
                        k = min(top_k, logits.size(-1))
                        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
                        logits = logits.masked_fill(logits < kth, float("-inf"))

                    if top_p is not None and top_p < 1.0:
                        sl, si = torch.sort(logits, descending=True, dim=-1)
                        sm = torch.softmax(sl, dim=-1)
                        drop = (sm.cumsum(dim=-1) - sm) > top_p
                        drop[..., 0] = False
                        logits = logits.masked_fill(drop.scatter(-1, si, drop), float("-inf"))

                    probs = torch.softmax(logits, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)

                ids = torch.cat((ids, nxt), dim=1)
                self.token_count += 1

                if nxt.item() == self.tok.eos_token_id:
                    q.put(("meta", "\n[eos]\n"))
                    break
                q.put(("text", self.tok.decode(nxt[0].tolist())))

            dt = time.time() - self.t0
            q.put(("done", f"\n{self.token_count} tokens in {dt:.2f}s "
                           f"({self.token_count / max(dt, 1e-9):.2f} tok/s)"))
        except Exception as e:  # surface errors in the UI
            q.put(("error", f"\n[error] {type(e).__name__}: {e}\n"))
        finally:
            q.put(("idle", ""))

    def _drain(self):
        done = False
        summary = ""
        while True:
            try:
                kind, text = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "idle":
                done = True
                continue
            if kind == "done":
                summary = text.strip()
            self.output.configure(state="normal")
            tag = kind if kind in ("prompt", "meta", "error") else None
            self.output.insert("end", text, tag)
            self.output.see("end")
            self.output.configure(state="disabled")
            if kind == "error":
                self.status.configure(text="error", style="Muted.TLabel")
        if done:
            self.btn_run.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            if not self.status.cget("text").startswith(("error", "stopping")):
                self.status.configure(text=summary or "ready")
        self.root.after(60, self._drain)


def main():
    ap = argparse.ArgumentParser(description="TinyLLM desktop GUI")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_checkpoint(args.checkpoint)
    model, payload = GPT.load(ckpt, map_location=device)
    model.eval()
    tok = get_tokenizer()

    root = tk.Tk()
    App(root, model, tok, ckpt, payload, device)
    root.mainloop()


if __name__ == "__main__":
    main()

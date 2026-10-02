"""Training loop for the TinyLLM GPT model.

Features
--------
* next-token-prediction loss (cross entropy over shifted targets)
* AdamW with decoupled weight decay, linear warmup + cosine decay
* mixed precision (fp16 on CUDA, bf16 on CPU) via torch.autocast
* gradient clipping
* gradient checkpointing to cut activation memory
* periodic validation loss, checkpointing every N steps, resume support
* tqdm progress bars and matplotlib loss curves

Usage:
    python scripts/train.py
    python scripts/train.py --preset cpu-tiny
    python scripts/train.py --max-iters 4000 --batch-size 8 --block-size 256
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import Config, RUNS_DIR, ModelConfig, set_seed
from model import GPT, build_model, model_summary
from scripts.prepare_data import get_batch

PRESETS = {
    # requested full configuration (12L / 768d / 12H) -- for a real GPU
    "full":  dict(n_layer=12, n_head=12, n_embd=768, block_size=256,
                  batch_size=8,  max_iters=4000, dropout=0.1),
    "medium": dict(n_layer=8,  n_head=8,  n_embd=512, block_size=256,
                   batch_size=8,  max_iters=2000, dropout=0.1),
    # practical on this CPU-only Ryzen 3
    "cpu-tiny": dict(n_layer=6,  n_head=6,  n_embd=384, block_size=192,
                     batch_size=4,  max_iters=600,  dropout=0.1),
    # fastest end-to-end sanity check
    "smoke": dict(n_layer=2,  n_head=4,  n_embd=128, block_size=64,
                  batch_size=2,  max_iters=20,   dropout=0.0),
}


def autocast_ctx(device: torch.device, enabled: bool):
    """fp16+GradScaler on CUDA, bf16 on CPU (no scaler needed)."""
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def lr_at(step, cfg):
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / cfg.warmup_iters
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    progress = min(1.0, max(0.0, progress))
    return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, cfg):
    decay, nodecay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2D params (weights) get decay; 1D (biases, layernorms) do not
        (decay if p.dim() >= 2 else nodecay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=1e-8,
    )


@torch.no_grad()
def estimate_loss(model, dc, tc, device, split, iters, scaler=None, use_amp=True):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(split, dc.batch_size, dc.block_size, dc.binary_dir, str(device))
        with autocast_ctx(device, use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def plot_curves(history, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history if h.get("val_loss") is not None]
    train_l = [h["train_loss"] for h in history if h.get("val_loss") is not None]
    val_l = [h["val_loss"] for h in history if h.get("val_loss") is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(steps, train_l, label="train", color="#2b6cb0")
    ax1.plot(steps, val_l, label="val", color="#c53030")
    ax1.set_xlabel("step")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Training and validation loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(steps, val_l, color="#c53030")
    ax2.set_xlabel("step")
    ax2.set_ylabel("val loss")
    ax2.set_ylim(0, max(val_l) * 1.1 if val_l else 1)
    ax2.set_title("Validation loss (zoomed)")
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot]   loss curves -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Train the TinyLLM GPT model")
    ap.add_argument("--preset", default="cpu-tiny", choices=sorted(PRESETS))
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--checkpoint-interval", type=int, default=200)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--amp", dest="use_amp", action="store_true", default=None,
                    help="force mixed precision on (default: on for CUDA, off for CPU)")
    ap.add_argument("--no-amp", dest="use_amp", action="store_false")
    ap.add_argument("--checkpointing", dest="use_checkpointing",
                    action="store_true", default=None,
                    help="force gradient checkpointing on (default: on for CUDA, off for CPU)")
    ap.add_argument("--no-checkpointing", dest="use_checkpointing", action="store_false")
    ap.add_argument("--resume", default=None, help="path to a checkpoint to resume")
    ap.add_argument("--summary-only", action="store_true",
                    help="print the model summary and exit without training")
    args = ap.parse_args()

    cfg = Config()
    preset = PRESETS[args.preset]

    cfg.model.n_layer = preset["n_layer"]
    cfg.model.n_head = preset["n_head"]
    cfg.model.n_embd = preset["n_embd"]
    cfg.model.block_size = args.block_size or preset["block_size"]
    cfg.model.dropout = preset["dropout"]
    cfg.data.block_size = cfg.model.block_size
    cfg.data.batch_size = args.batch_size or preset["batch_size"]

    tc = cfg.train
    tc.max_iters = args.max_iters or preset["max_iters"]
    tc.epochs = args.epochs
    tc.eval_interval = args.eval_interval
    tc.eval_iters = args.eval_iters
    tc.checkpoint_interval = args.checkpoint_interval
    tc.learning_rate = args.learning_rate
    tc.warmup_iters = max(10, int(0.05 * tc.max_iters))

    # Mixed precision and gradient checkpointing are large wins on an NVIDIA GPU
    # but catastrophic on a CPU-only box: bf16 is emulated on pre-AVX512-BF16
    # parts such as Zen2, and checkpointing blocks the fused SDPA path.
    # Measured here: 195 s/iter with both on vs 2.95 s/iter with both off.
    is_cuda = torch.cuda.is_available()
    tc.use_amp = is_cuda if args.use_amp is None else args.use_amp
    tc.use_checkpointing = is_cuda if args.use_checkpointing is None else args.use_checkpointing
    tc.run_name = args.run_name or f"gpt-{args.preset}"

    set_seed(tc.seed)
    torch.set_num_threads(tc.num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS_DIR / tc.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg.model).to(device)
    print(model_summary(model))
    print(f"\n[device] {device}" + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else "  (no NVIDIA GPU present)"))
    print(f"[preset] {args.preset} | {tc.max_iters} iters | batch {cfg.data.batch_size} "
          f"| block {cfg.model.block_size} | threads {torch.get_num_threads()}")
    print(f"[mem]    amp={tc.use_amp} grad_checkpointing={tc.use_checkpointing}"
          + ("" if is_cuda else "   (both auto-disabled: emulated bf16 and "
                                "checkpointing are ~66x slower on this CPU)"))
    print(f"[run]    {run_dir}\n")

    if args.summary_only:
        return

    if not (cfg.data.binary_dir / "train.bin").exists():
        raise SystemExit("train.bin missing -- run: python scripts/prepare_data.py")

    opt = make_optimizer(model, tc)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and tc.use_amp))

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            opt.load_state_dict(ck["optimizer_state"])
        start_step = ck.get("step", 0)
        print(f"[resume] from {args.resume} at step {start_step}")

    ckpt_path = run_dir / "checkpoint.pt"
    if ckpt_path.exists() and args.resume is None:
        print(f"[info]   existing checkpoint at {ckpt_path} (use --resume to continue)")

    model.train()
    model.set_gradient_checkpointing(tc.use_checkpointing)

    from tqdm import trange
    history = []
    t_start = time.time()
    best_val = float("inf")

    print(f"{'step':>6} {'loss':>8} {'lr':>10} {'val':>8} {'tok/s':>9} {'mem':>7}", flush=True)

    for step in trange(start_step, tc.max_iters, desc="train", ncols=90, leave=False):
        lr = lr_at(step, tc)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = get_batch("train", cfg.data.batch_size, cfg.model.block_size,
                         cfg.data.binary_dir, str(device))

        t0 = time.time()
        with autocast_ctx(device, tc.use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(opt)
        scaler.update()
        dt = time.time() - t0

        do_eval = (step + 1) % tc.eval_interval == 0 or step + 1 == tc.max_iters
        val_loss = None
        if do_eval:
            val_loss = estimate_loss(model, cfg.data, tc, device, "val",
                                     tc.eval_iters, scaler, tc.use_amp)
            history.append({
                "step": step + 1,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "lr": lr,
                "elapsed_s": time.time() - t_start,
            })
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)
            print(f"{step + 1:>6} {loss.item():>8.4f} {lr:>10.2e} {val_loss:>8.4f} "
                  f"{cfg.data.batch_size * cfg.model.block_size / max(dt, 1e-9):>9.0f} "
                  f"{'cuda' if device.type == 'cuda' else 'cpu':>7}"
                  f"{'  *best' if improved else ''}", flush=True)

        if (step + 1) % tc.checkpoint_interval == 0 or step + 1 == tc.max_iters:
            model.save(ckpt_path, optimizer=opt, step=step + 1,
                       extra={"val_loss": val_loss, "config": cfg.to_dict()})
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
            if do_eval:
                plot_curves(history, run_dir / "loss_curves.png", f"TinyLLM - {tc.run_name}")

    total = time.time() - t_start
    model.save(ckpt_path, optimizer=opt, step=tc.max_iters,
               extra={"val_loss": best_val, "config": cfg.to_dict()})
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    if history:
        plot_curves(history, run_dir / "loss_curves.png", f"TinyLLM - {tc.run_name}")

    print(f"\n[done]   {tc.max_iters - start_step} steps in {total / 60:.1f} min")
    print(f"[final]  best val loss {best_val:.4f}")
    print(f"[saved]  {ckpt_path}")
    print(f"[plot]   {run_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()

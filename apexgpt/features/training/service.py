"""Training service: optimizer, schedule, loop, checkpointing, plotting."""
from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from ...core.config import Config, ModelConfig, TrainConfig
from ...core.device import (autocast_dtype, describe_device, get_device, profile,
                            use_amp_by_default, use_checkpointing_by_default)
from ...models.builder import build_model, print_summary
from ...models.gpt import GPT

# Preset model/training shapes, chosen by how much memory the host has.
PRESETS: dict[str, dict] = {
    # 123.8M params; the configuration requested for a GPU
    "full":     dict(n_layer=12, n_head=12, n_embd=768, block_size=256,
                     batch_size=8, max_iters=4000, dropout=0.1),
    # ~50M params; fits a 6-8 GB card
    "medium":   dict(n_layer=8,  n_head=8,  n_embd=512, block_size=256,
                     batch_size=8, max_iters=2000, dropout=0.1),
    # 30.0M params; the practical choice on a CPU-only laptop
    "cpu-tiny": dict(n_layer=6,  n_head=6,  n_embd=384, block_size=192,
                     batch_size=4, max_iters=600,  dropout=0.1),
    # 6.8M params; seconds-long end-to-end sanity check
    "smoke":    dict(n_layer=2,  n_head=4,  n_embd=128, block_size=64,
                     batch_size=2, max_iters=20,   dropout=0.0),
}


# --------------------------------------------------------------------------- #
# precision + schedule
# --------------------------------------------------------------------------- #
def autocast_ctx(device: torch.device, enabled: bool):
    """fp16 on CUDA (with a GradScaler), bf16 on CPU."""
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=autocast_dtype(device))


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_iters)
    span = max(1, cfg.max_iters - cfg.warmup_iters)
    progress = min(1.0, max(0.0, (step - cfg.warmup_iters) / span))
    return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (
        1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with decoupled decay on 2D weights only."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=1e-8,
    )


def steps_per_epoch(train_tokens: int, batch_size: int, block_size: int) -> int:
    tokens_per_step = batch_size * block_size
    return max(1, train_tokens // max(1, tokens_per_step))


@torch.no_grad()
def estimate_loss(model: GPT, batcher, device, iters: int, use_amp: bool,
                  split: str = "val") -> float:
    """Mean loss over ``iters`` random batches, leaving the mode untouched."""
    was_training = model.training
    model.eval()
    try:
        losses = []
        for _ in range(iters):
            x, y = batcher.get_batch(split)
            with autocast_ctx(device, use_amp):
                _, loss = model(x, y)
            losses.append(loss.item())
    finally:
        model.train(was_training)
    return float(np.mean(losses)) if losses else float("nan")


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    steps: int
    best_val_loss: float
    final_val_loss: float
    elapsed_s: float
    checkpoint: Path
    history: list = field(default_factory=list)


def plot_curves(history: list, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [h for h in history if h.get("val_loss") is not None]
    if not rows:
        return
    steps = [h["step"] for h in rows]
    train_l = [h["train_loss"] for h in rows]
    val_l = [h["val_loss"] for h in rows]

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
    ax2.set_ylim(0, max(val_l) * 1.1)
    ax2.set_title("Validation loss (zoomed)")
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot]   loss curves -> {out_path}")


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def train(cfg: Config, preset: str = "cpu-tiny", max_iters: int | None = None,
          epochs: int | None = None, batch_size: int | None = None,
          block_size: int | None = None, learning_rate: float | None = None,
          eval_interval: int | None = None, eval_iters: int | None = None,
          checkpoint_interval: int | None = None, log_interval: int = 10,
          run_name: str | None = None, use_amp: bool | None = None,
          use_checkpointing: bool | None = None, resume: str | None = None,
          device_spec: str | None = None, num_threads: int | None = None,
          seed: int | None = None, summary_only: bool = False,
          progress=print) -> TrainResult | None:
    """Run the training loop.

    ``epochs`` and ``max_iters`` are mutually exclusive targets: supplying
    ``epochs`` derives the step count from the real corpus size. Previously
    ``--epochs`` was accepted and silently ignored.
    """
    p = PRESETS[preset]
    hw = profile()

    # The corpus records which tokenizer its tokens came from; the model's
    # vocabulary has to match it or every logit is indexed against the wrong
    # table. Older corpora predate the record and were always GPT-2.
    from ..data.tokenizers import resolve_corpus_spec
    tok_spec = resolve_corpus_spec(cfg.data.binary_dir, cfg.data.tokenizer)
    cfg.model.vocab_size = tok_spec.vocab_size
    cfg.data.vocab_size = tok_spec.vocab_size
    cfg.data.tokenizer = tok_spec.kind

    cfg.model.n_layer = p["n_layer"]
    cfg.model.n_head = p["n_head"]
    cfg.model.n_embd = p["n_embd"]
    cfg.model.block_size = block_size or p["block_size"]
    cfg.model.dropout = p["dropout"]
    cfg.model.validate()

    cfg.data.block_size = cfg.model.block_size
    cfg.data.batch_size = batch_size or p["batch_size"]

    tc = cfg.train
    tc.max_iters = max_iters or p["max_iters"]
    tc.epochs = epochs if epochs is not None else tc.epochs
    tc.eval_interval = eval_interval or tc.eval_interval
    tc.eval_iters = eval_iters or tc.eval_iters
    tc.checkpoint_interval = checkpoint_interval or tc.checkpoint_interval
    if learning_rate is not None:
        tc.learning_rate = learning_rate
    tc.warmup_iters = max(10, int(0.05 * tc.max_iters))
    tc.run_name = run_name or f"gpt-{preset}"

    device = get_device(device_spec)
    tc.use_amp = use_amp_by_default(device) if use_amp is None else use_amp
    tc.use_checkpointing = (use_checkpointing_by_default(device)
                            if use_checkpointing is None else use_checkpointing)
    if num_threads is not None:
        tc.num_threads = num_threads

    if seed is not None:
        tc.seed = seed

    model = build_model(cfg.model).to(device)
    print_summary(model)

    from ...core.device import configure_threads
    threads = configure_threads(tc.num_threads)
    progress(f"\n[hardware] {hw.describe()}")
    progress(f"[device]   {describe_device(device)} | cpu threads {threads}")
    progress(f"[preset]   {preset} | {tc.max_iters} iters | batch {cfg.data.batch_size} "
             f"| block {cfg.model.block_size}")
    progress(f"[tokenize] {tok_spec.kind} tokenizer, vocab {tok_spec.vocab_size}")
    progress(f"[mem]      amp={tc.use_amp} grad_checkpointing={tc.use_checkpointing}"
             + ("" if device.type != "cpu"
                else "   (CPU: bf16 is emulated without native support and"
                     " checkpointing blocks fused attention)"))

    from ...core.paths import RUNS_DIR
    run_dir = RUNS_DIR / tc.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress(f"[run]    {run_dir}\n")

    if summary_only:
        return None

    if not cfg.data.is_ready():
        raise FileNotFoundError(
            f"token binaries for corpus {cfg.data.dataset!r} are missing "
            f"(expected {cfg.data.binary_dir}) -- run: "
            f"python -m apexgpt data prepare --source {cfg.data.dataset}")

    if epochs is not None and max_iters is None:
        total_tokens = cfg.data.train_path.stat().st_size // 2
        spe = steps_per_epoch(total_tokens, cfg.data.batch_size, cfg.model.block_size)
        derived = spe * max(1, int(epochs))
        progress(f"[epochs] {total_tokens:,} train tokens -> {spe:,} steps/epoch "
                 f"x {epochs} = {derived:,} steps")
        if derived > 10_000:
            progress("[warn]   a full epoch of this corpus is far beyond a CPU "
                     "budget; prefer --max-iters for a bounded run")
        tc.max_iters = derived
        tc.warmup_iters = max(10, int(0.05 * derived))

    optimizer = make_optimizer(model, tc)
    # fp16 needs a loss scaler on every backend that uses it; bf16 does not
    needs_scaler = tc.use_amp and device.type in ("cuda", "xpu")
    scaler = torch.amp.GradScaler(device.type, enabled=needs_scaler)

    start_step = 0
    history: list = []
    best_val = float("inf")

    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            optimizer.load_state_dict(ck["optimizer_state"])
        start_step = int(ck.get("step", 0))
        # Resuming previously discarded these, so curves silently restarted and
        # the "best" marker regressed to whatever this run happened to reach.
        extra = ck.get("extra") or {}
        history = list(extra.get("history") or ck.get("history") or [])
        best_val = float(extra.get("best_val_loss",
                                    extra.get("val_loss", float("inf"))))
        progress(f"[resume] {resume} at step {start_step} "
                 f"({len(history)} history points, best val {best_val:.4f})")
        if start_step >= tc.max_iters:
            progress(f"[info]   already trained to step {tc.max_iters}; "
                     f"raise --max-iters to continue")
            return TrainResult(start_step, best_val, best_val, 0.0,
                               run_dir / "checkpoint.pt", history)

    ckpt_path = run_dir / "checkpoint.pt"
    if ckpt_path.exists() and resume is None:
        progress(f"[info]   existing checkpoint at {ckpt_path} "
                 f"(use --resume to continue)")

    model.train()
    model.set_gradient_checkpointing(tc.use_checkpointing)

    from ..data.service import TokenBatcher
    batcher = TokenBatcher(cfg.data.binary_dir, device=str(device),
                           batch_size=cfg.data.batch_size,
                           block_size=cfg.model.block_size,
                           val_batch_size=cfg.data.val_batch_size)

    progress(f"{'step':>6} {'loss':>8} {'lr':>10} {'val':>8} {'tok/s':>9} {'dev':>5}",
             flush=True)

    t_start = time.time()
    done = 0
    for step in trange(start_step, tc.max_iters, desc="train", ncols=90, leave=False):
        lr = lr_at(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = batcher.get_batch("train")

        t0 = time.time()
        with autocast_ctx(device, tc.use_amp):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        dt = time.time() - t0
        done += 1

        is_last = step + 1 == tc.max_iters
        if (step + 1) % tc.eval_interval == 0 or is_last:
            val_loss = estimate_loss(model, batcher, device, tc.eval_iters,
                                     tc.use_amp)
            history.append({
                "step": step + 1,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "lr": lr,
                "elapsed_s": time.time() - t_start,
            })
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)
            progress(f"{step + 1:>6} {loss.item():>8.4f} {lr:>10.2e} {val_loss:>8.4f} "
                     f"{cfg.data.batch_size * cfg.model.block_size / max(dt, 1e-9):>9.0f} "
                     f"{device.type:>5}{'  *best' if improved else ''}", flush=True)

        if (step + 1) % tc.checkpoint_interval == 0 or is_last:
            model.save(ckpt_path, optimizer=optimizer, step=step + 1,
                       extra={"val_loss": best_val, "best_val_loss": best_val,
                              "history": history, "config": cfg.to_dict(),
                              "preset": preset, "tokenizer": tok_spec.kind,
                              "vocab_size": tok_spec.vocab_size})
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    elapsed = time.time() - t_start
    final_val = history[-1]["val_loss"] if history else float("nan")

    model.save(ckpt_path, optimizer=optimizer, step=tc.max_iters,
               extra={"val_loss": final_val, "best_val_loss": best_val,
                      "history": history, "config": cfg.to_dict(),
                      "preset": preset, "tokenizer": tok_spec.kind,
                      "vocab_size": tok_spec.vocab_size})
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_curves(history, run_dir / "loss_curves.png", f"ApexGPT - {tc.run_name}")

    batcher.close()

    progress(f"\n[done]   {done} steps in {elapsed / 60:.1f} min")
    progress(f"[final]  best val loss {best_val:.4f}")
    progress(f"[saved]  {ckpt_path}")

    return TrainResult(done, best_val, final_val, elapsed, ckpt_path, history)


def show_history(run_dir: Path) -> int:
    """Print a run's loss table."""
    path = run_dir / "history.json"
    if not path.exists():
        print(f"no history at {path}")
        return 1
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        print("history is empty")
        return 1
    print(f"{len(rows)} eval points\nstep  train   val     lr")
    step = max(1, len(rows) // 6)
    picked = rows[::step]
    # avoid printing the final row twice when it already appears in the sample
    if picked[-1] is not rows[-1]:
        picked = picked + [rows[-1]]
    for r in picked:
        print(f"{r['step']:>4} {r['train_loss']:.4f} {r['val_loss']:.4f} {r['lr']:.2e}")
    best = min(rows, key=lambda r: r["val_loss"])
    print(f"\nbest val: {best['val_loss']:.4f} at step {best['step']}")
    print(f"elapsed: {rows[-1]['elapsed_s'] / 60:.1f} min")
    return 0
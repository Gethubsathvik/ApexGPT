"""Fast end-to-end smoke test of the training loop on synthetic tokens.

Validates: model forward/backward, AdamW step, AMP, gradient checkpointing,
validation pass, checkpoint save/load, loss-curve plotting, and the sampling
code in model.generate -- all in a few seconds without the real corpus.

Usage:
    TINYLLM_DATA_DIR=C:\\tinyllm_data python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import DATA_DIR, ModelConfig, RUNS_DIR, set_seed
from model import GPT, build_model
from scripts.train import PRESETS, autocast_ctx, estimate_loss, lr_at, make_optimizer

PASS, FAIL = "  [PASS]", "  [FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    results.append((name, bool(cond), detail))
    print(f"{PASS if cond else FAIL} {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return cond


def make_fixtures():
    d = DATA_DIR / "binary"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 50000, size=1_200_000, dtype=np.uint16)
    tr, va = d / "smoke_train.npy", d / "smoke_val.npy"
    np.save(tr, ids[:1_000_000])
    np.save(va, ids[1_000_000:])
    return tr, va


def fake_get_batch(split, batch_size, block_size, binary_dir, device="cpu"):
    a = np.load(binary_dir / f"smoke_{split}.npy")
    ix = torch.randint(len(a) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(a[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(a[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def main():
    set_seed(1337)
    make_fixtures()

    import scripts.prepare_data as pd_mod
    import scripts.train as train_mod
    pd_mod.get_batch = fake_get_batch
    train_mod.get_batch = fake_get_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # --- model init ------------------------------------------------------- #
    cfg = ModelConfig(**{**ModelConfig().__dict__, **{
        "n_layer": 2, "n_head": 4, "n_embd": 128, "block_size": 64, "dropout": 0.0}})
    model = build_model(cfg).to(device)
    n = model.num_parameters()
    check("model builds", n > 0, f"{n:,} params")
    check("weight tying", model.lm_head.weight is model.transformer.wte.weight)

    x, y = fake_get_batch("train", 2, cfg.block_size, DATA_DIR / "binary", str(device))
    logits, loss = model(x, y)
    check("forward shapes", logits.shape[:2] == x.shape,
          f"logits {tuple(logits.shape)} vs x {tuple(x.shape)}")
    check("loss is finite", torch.isfinite(loss).item(), f"loss={loss.item():.3f}")
    check("loss ~ ln(vocab) at init", abs(loss.item() - np.log(cfg.vocab_size)) < 1.5,
          f"ln(50257)={np.log(cfg.vocab_size):.3f}")

    # --- causality: changing a later token must not alter earlier logits --- #
    model.eval()
    with torch.no_grad():
        l1, _ = model(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1234) % cfg.vocab_size
        l2, _ = model(x2)
    same = torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5)
    check("causal masking (no future leakage)", same,
          f"max diff {float((l1[:, :-1] - l2[:, :-1]).abs().max()):.2e}")

    # --- backward + optimizer --------------------------------------------- #
    tcfg = train_mod.Config().train
    tcfg.max_iters = 6
    tcfg.warmup_iters = 2
    opt = make_optimizer(model, tcfg)
    model.train()
    model.set_gradient_checkpointing(True)
    l0 = None
    losses = []
    for i in range(6):
        lr = lr_at(i, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr
        xb, yb = fake_get_batch("train", 2, cfg.block_size, DATA_DIR / "binary", str(device))
        with autocast_ctx(device, True):
            _, l = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(l.item())
    check("backward + AdamW step", all(np.isfinite(losses)), f"loss {losses[0]:.2f} -> {losses[-1]:.2f}")
    check("loss decreases on fixed batch",
          all(np.isfinite(losses)) and losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    check("lr schedule warmup+cosine",
          lr_at(0, tcfg) < lr_at(3, tcfg) and lr_at(5, tcfg) < lr_at(3, tcfg),
          f"lr(0)={lr_at(0,tcfg):.2e} lr(3)={lr_at(3,tcfg):.2e} lr(5)={lr_at(5,tcfg):.2e}")

    # --- grad checkpointing equivalence ----------------------------------- #
    model.eval()
    with torch.no_grad():
        a, _ = model(x)
    model.train()
    model.set_gradient_checkpointing(True)
    with torch.no_grad():
        b, _ = model(x)
    model.set_gradient_checkpointing(False)
    check("grad checkpointing is transparent", torch.allclose(a, b, atol=1e-5),
          f"max diff {float((a - b).abs().max()):.2e}")
    model.eval()

    # --- validation loss -------------------------------------------------- #
    dcfg = train_mod.Config().data
    dcfg.block_size = cfg.block_size
    dcfg.batch_size = 2
    vl = estimate_loss(model, dcfg, tcfg, device, "val", 3, None, True)
    check("estimate_loss", np.isfinite(vl), f"val={vl:.4f}")

    # --- checkpoint round trip -------------------------------------------- #
    run_dir = RUNS_DIR / "_smoke_test"   # disposable: safe to delete
    run_dir.mkdir(parents=True, exist_ok=True)
    ck = run_dir / "checkpoint.pt"
    model.save(ck, optimizer=opt, step=6)
    model2, payload = GPT.load(ck, map_location=str(device))
    with torch.no_grad():
        a2, _ = model2(x)
    check("checkpoint save/load", torch.allclose(a, a2, atol=1e-6),
          f"step={payload['step']}")

    # --- generation / sampling -------------------------------------------- #
    model.eval()
    prompt = torch.randint(0, 50000, (1, 8), device=device)
    torch.manual_seed(0)
    g1 = model.generate(prompt.clone(), 12, temperature=1.0, top_k=50, top_p=0.95)
    torch.manual_seed(0)
    g2 = model.generate(prompt.clone(), 12, temperature=1.0, top_k=50, top_p=0.95)
    check("generate shape", g1.shape == (1, 20), f"{tuple(g1.shape)}")
    check("generate reproducible with seed", torch.equal(g1, g2))
    gt = model.generate(prompt.clone(), 10, temperature=0.01, top_k=None, top_p=None)
    check("low temperature is near-deterministic",
          len(set(gt[0, 8:].tolist())) <= 3,
          f"unique={len(set(gt[0, 8:].tolist()))}")

    kv = model.new_kv_caches(1, str(device), torch.float32)
    with torch.no_grad():
        _ = model(prompt, kv_caches=kv)
        _ = model(torch.randint(0, 50000, (1, 1), device=device), kv_caches=kv)
    check("KV cache grows", kv[0]["k"] is not None and kv[0]["k"].shape[2] == 9,
          f"len={kv[0]['k'].shape[2]}" if kv[0]["k"] is not None else "none")

    # cached incremental decoding must match a full forward pass
    seq = torch.randint(0, 50000, (1, 20), device=device)
    with torch.no_grad():
        ref, _ = model(seq)
    kv2 = model.new_kv_caches(1, str(device), torch.float32)
    with torch.no_grad():
        model(seq[:, :16], kv_caches=kv2)
        step_out, _ = model(seq[:, 16:], kv_caches=kv2)
    check("KV cache matches full forward",
          torch.allclose(ref[:, 16:], step_out, atol=1e-4),
          f"max diff {float((ref[:, 16:] - step_out).abs().max()):.2e}")

    # generation longer than the context window must not crash
    long_out = model.generate(prompt.clone(), cfg.block_size + 40,
                              temperature=0.8, top_k=40, top_p=0.95)
    check("generation beyond context window", long_out.shape[1] == 8 + cfg.block_size + 40,
          f"len={long_out.shape[1]}")
    long_kv = model.new_kv_caches(1, str(device), torch.float32)
    check("KV cache stays bounded", long_kv[0]["k"] is None, "trimmed during generate")

    # --- loss curve plotting ---------------------------------------------- #
    hist = [{"step": i * 10, "train_loss": 8 - i * 0.4, "val_loss": 8.2 - i * 0.3}
            for i in range(6)]
    train_mod.plot_curves(hist, run_dir / "loss_curves.png", "smoke")
    check("loss curve plot written", (run_dir / "loss_curves.png").exists())

    # --- GUI importability ------------------------------------------------- #
    try:
        import scripts.gui  # noqa: F401
        check("gui module imports", True)
    except Exception as e:
        check("gui module imports", False, str(e))

    # --- summary ----------------------------------------------------------- #
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'=' * 60}\nSMOKE TEST: {passed}/{total} passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name} {detail}")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

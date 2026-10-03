"""Model, sampling and training-service correctness tests.

Fast enough to run on CPU with a tiny config. Run with::

    python -m pytest tests -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyllm.core.config import Config, ModelConfig, TrainConfig
from tinyllm.features.training.service import lr_at, make_optimizer, steps_per_epoch
from tinyllm.models.builder import build_model, count_parameters
from tinyllm.models.gpt import GPT
from tinyllm.models.sampling import (apply_repetition_penalty, apply_top_k,
                                     apply_top_p, sample_next_token)

TINY = dict(vocab_size=512, block_size=32, n_layer=2, n_head=4,
            n_embd=64, dropout=0.0)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_model(ModelConfig(**TINY))
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(1)
    return (torch.randint(0, TINY["vocab_size"], (2, TINY["block_size"])),
            torch.randint(0, TINY["vocab_size"], (2, TINY["block_size"])))


# --------------------------------------------------------------------- model
def test_model_builds_and_counts_parameters(model):
    counts = count_parameters(model)
    assert counts["total"] > 0
    assert counts["embeddings"] + counts["transformer"] == counts["total"]


def test_weight_tying(model):
    assert model.lm_head.weight is model.transformer.wte.weight


def test_config_validation_rejects_bad_head_count():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=768, n_head=7).validate()


def test_forward_shapes_and_loss(model, batch):
    x, y = batch
    logits, loss = model(x, y)
    assert logits.shape == (x.shape[0], x.shape[1], TINY["vocab_size"])
    assert torch.isfinite(loss)
    assert abs(loss.item() - math.log(TINY["vocab_size"])) < 1.5


def test_forward_rejects_too_long_sequence(model):
    too_long = torch.zeros(1, TINY["block_size"] + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds block size"):
        model(too_long)


def test_causal_mask_has_no_future_leakage(model, batch):
    """Changing the last token must not move any earlier logit."""
    x, _ = batch
    with torch.no_grad():
        before, _ = model(x)
        altered = x.clone()
        altered[:, -1] = (altered[:, -1] + 1) % TINY["vocab_size"]
        after, _ = model(altered)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-5)


def test_gradient_checkpointing_is_transparent(model, batch):
    x, _ = batch
    with torch.no_grad():
        plain, _ = model(x)
    model.train()
    model.set_gradient_checkpointing(True)
    with torch.no_grad():
        checked, _ = model(x)
    model.eval()
    model.set_gradient_checkpointing(False)
    assert torch.allclose(plain, checked, atol=1e-5)


def test_gradient_checkpointing_flag_is_order_independent(batch):
    """Regression: the flag used to be folded with .training at set time, so
    enabling it before .train() silently did nothing."""
    torch.manual_seed(2)
    m = build_model(ModelConfig(**TINY))
    m.set_gradient_checkpointing(True)      # before train()
    m.train()
    assert m.gradient_checkpointing is True
    m.eval()
    assert m.gradient_checkpointing is False


def test_checkpoint_round_trip(model, batch, tmp_path):
    x, _ = batch
    with torch.no_grad():
        reference, _ = model(x)
    path = tmp_path / "ckpt.pt"
    model.save(path, step=3)
    reloaded, payload = GPT.load(path)
    reloaded.eval()
    with torch.no_grad():
        after, _ = reloaded(x)
    assert payload["step"] == 3
    assert torch.allclose(reference, after, atol=1e-6)


# ------------------------------------------------------------------- KV cache
def test_kv_cache_matches_full_forward(model):
    """Regression: cached multi-token steps used is_causal=False and every
    query attended to future keys."""
    torch.manual_seed(3)
    seq = torch.randint(0, TINY["vocab_size"], (1, 20))
    with torch.no_grad():
        full, _ = model(seq)
    caches = model.new_kv_caches()
    with torch.no_grad():
        model(seq[:, :16], kv_caches=caches)
        stepped, _ = model(seq[:, 16:], kv_caches=caches)
    assert torch.allclose(full[:, 16:], stepped, atol=1e-4)


def test_kv_cache_is_empty_on_creation(model):
    caches = model.new_kv_caches()
    assert len(caches) == TINY["n_layer"]
    assert all(c["k"] is None and c["v"] is None for c in caches)


def test_generation_beyond_context_window(model):
    prompt = torch.randint(0, TINY["vocab_size"], (1, 8))
    total = 8 + TINY["block_size"] + 20
    out = model.generate(prompt.clone(), TINY["block_size"] + 20,
                         temperature=0.8, top_k=20, top_p=0.95)
    assert out.shape[1] == total


# ------------------------------------------------------------------- sampling
def test_top_k_keeps_exactly_k_tokens():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    kept = apply_top_k(logits, 2)
    assert torch.isinf(kept[0, :3]).all()
    assert torch.isfinite(kept[0, 3:]).all()


def test_top_p_always_keeps_the_most_likely():
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    kept = apply_top_p(logits, 0.1)
    assert torch.isfinite(kept[0, 0])
    assert torch.isinf(kept[0, 1:]).all()


def test_top_p_never_empties_the_distribution():
    logits = torch.full((1, 64), -50.0)
    logits[0, 3] = 50.0
    kept = apply_top_p(logits, 0.001)
    assert torch.isfinite(kept[0, 3])


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = torch.tensor([[2.0, 2.0]])
    out = apply_repetition_penalty(logits, torch.tensor([0]), 2.0)
    assert out[0, 0] == pytest.approx(1.0)


def test_greedy_when_temperature_is_zero():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    token = sample_next_token(logits, temperature=0.0)
    assert token.item() == 1


def test_sampling_is_reproducible_with_a_seed():
    logits = torch.randn(1, 64)
    torch.manual_seed(7)
    first = sample_next_token(logits, temperature=1.0)
    torch.manual_seed(7)
    second = sample_next_token(logits, temperature=1.0)
    assert torch.equal(first, second)


# ------------------------------------------------------------------ training
def test_lr_warms_up_then_decays():
    cfg = TrainConfig(max_iters=100, warmup_iters=10,
                      learning_rate=1e-3, min_lr=1e-5)
    assert lr_at(0, cfg) < lr_at(5, cfg) < lr_at(10, cfg)
    assert lr_at(99, cfg) < lr_at(50, cfg)


def test_optimizer_excludes_norms_from_weight_decay(model):
    cfg = TrainConfig()
    opt = make_optimizer(model, cfg)
    decayed = opt.param_groups[0]
    undecayed = opt.param_groups[1]
    assert decayed["weight_decay"] == cfg.weight_decay
    assert undecayed["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decayed["params"])
    assert all(p.dim() < 2 for p in undecayed["params"])


def test_training_step_reduces_loss_on_a_fixed_batch(model, batch):
    torch.manual_seed(4)
    m = build_model(ModelConfig(**TINY))
    m.train()
    m.set_gradient_checkpointing(True)
    opt = make_optimizer(m, TrainConfig())
    x, y = batch

    losses = []
    for _ in range(5):
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_steps_per_epoch_derives_from_corpus_size():
    # 1M tokens / (batch 4 * block 256) = 976 steps
    assert steps_per_epoch(1_000_000, batch_size=4, block_size=256) == 976


# --------------------------------------------------------------------- config
def test_config_wires_split_and_val_batch_size():
    cfg = Config()
    assert cfg.data.train_split == 0.8
    assert cfg.data.val_batch_size == 8
    assert cfg.data.is_ready() in (True, False)
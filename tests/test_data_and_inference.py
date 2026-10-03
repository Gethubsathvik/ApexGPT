"""Data pipeline and inference service tests.

These use small synthetic corpora so they never touch the real 1.2 GB download.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apexgpt.core.config import DataConfig
from apexgpt.features.data.service import (TokenBatcher, _split_binary,
                                           count_distinct_tokens, tokenize_corpus)
from apexgpt.features.inference.service import (GenerationRequest,
                                                InferenceEngine)
from apexgpt.models.builder import build_model
from apexgpt.models.gpt import GPT


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A small text file plus a tokenizer stub (no network)."""
    d = tmp_path_factory.mktemp("data")
    path = d / "corpus.txt"
    text = ("The quick brown fox jumps over the lazy dog. " * 400)
    path.write_text(text, encoding="utf-8")
    return d, path


class StubTokenizer:
    """Maps a-z to ids so tests stay deterministic and offline."""
    eos_token_id = 0
    vocab_size = 26

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [self(t)["input_ids"] for t in text]}
        ids = [ord(c) - ord("a") + 1 for c in text.lower()
               if "a" <= c.lower() <= "z"]
        return {"input_ids": ids or [1]}


def test_tokenize_and_split_produces_exact_counts(corpus):
    d, path = corpus
    out = d / "binary"
    n_train, n_val = tokenize_corpus(path, StubTokenizer(), out,
                                     train_split=0.8, num_workers=1)
    assert n_train + n_val > 0
    assert n_train == pytest.approx(n_val * 4, rel=0.02)
    assert (out / "train.bin").stat().st_size == n_train * 2
    assert (out / "val.bin").stat().st_size == n_val * 2


def test_split_binary_preserves_the_whole_stream(corpus):
    d, path = corpus
    out = d / "bin2"
    tokenize_corpus(path, StubTokenizer(), out, train_split=0.8)
    train = np.fromfile(out / "train.bin", dtype=np.uint16)
    val = np.fromfile(out / "val.bin", dtype=np.uint16)
    # the two halves must reconstruct the token stream exactly
    combined = np.concatenate([train, val])
    direct = np.asarray(StubTokenizer()(path.read_text(encoding="utf-8"))["input_ids"],
                        dtype=np.uint16)
    assert np.array_equal(combined, direct)


def test_split_binary_at_zero_and_one(tmp_path):
    src = tmp_path / "all.bin"
    values = np.arange(10, dtype=np.uint16)
    values.tofile(src)
    for split, expected_train in ((0.0, 0), (1.0, 10), (0.5, 5)):
        train_p, val_p = tmp_path / "t.bin", tmp_path / "v.bin"
        n = _split_binary(src, train_p, val_p, len(values), split)
        assert n == expected_train
        assert train_p.stat().st_size == expected_train * 2


def test_count_distinct_tokens(tmp_path):
    path = tmp_path / "t.bin"
    np.array([1, 2, 2, 3, 9], dtype=np.uint16).tofile(path)
    assert count_distinct_tokens(path, vocab_size=16) == 4


def test_batcher_returns_shifted_targets(tmp_path):
    binary = tmp_path / "binary"
    binary.mkdir()
    values = np.arange(1000, dtype=np.uint16) % 500
    values.tofile(binary / "train.bin")
    values.tofile(binary / "val.bin")

    cfg = DataConfig(binary_dir=binary, block_size=16)
    assert cfg.is_ready()

    with TokenBatcher(binary, batch_size=4, block_size=16) as batcher:
        x, y = batcher.get_batch("train")
        assert x.shape == (4, 16)
        # y is exactly x shifted one step left
        assert torch.equal(y[:, :-1], x[:, 1:])


def test_batcher_reuses_memmaps(tmp_path):
    """Regression: the 461 MB file used to be reopened on every batch."""
    binary = tmp_path / "binary"
    binary.mkdir()
    np.arange(1000, dtype=np.uint16).tofile(binary / "train.bin")
    np.arange(1000, dtype=np.uint16).tofile(binary / "val.bin")

    batcher = TokenBatcher(binary, batch_size=2, block_size=8)
    batcher.get_batch("train")
    first = batcher._maps["train"]
    batcher.get_batch("train")
    assert batcher._maps["train"] is first
    batcher.close()


def test_batcher_honours_val_batch_size(tmp_path):
    binary = tmp_path / "binary"
    binary.mkdir()
    np.arange(1000, dtype=np.uint16).tofile(binary / "train.bin")
    np.arange(1000, dtype=np.uint16).tofile(binary / "val.bin")
    with TokenBatcher(binary, batch_size=8, block_size=8, val_batch_size=2) as b:
        assert b.get_batch("train")[0].shape[0] == 8
        assert b.get_batch("val")[0].shape[0] == 2


def test_batcher_reports_a_missing_split(tmp_path):
    binary = tmp_path / "binary"
    binary.mkdir()
    np.arange(100, dtype=np.uint16).tofile(binary / "train.bin")
    with TokenBatcher(binary, block_size=8) as b:
        with pytest.raises(FileNotFoundError, match="data prepare"):
            b.get_batch("val")


def test_batcher_rejects_a_tiny_corpus(tmp_path):
    binary = tmp_path / "binary"
    binary.mkdir()
    np.arange(4, dtype=np.uint16).tofile(binary / "train.bin")
    with TokenBatcher(binary, block_size=8) as b:
        with pytest.raises(ValueError, match="need more"):
            b.get_batch("train")


# ------------------------------------------------------------------ inference
class StubEngineTokenizer:
    # outside the vocabulary so sampling never randomly hits eos
    eos_token_id = 999
    vocab_size = 64

    def __call__(self, text, return_tensors=None):
        ids = [1 + (ord(c) % 60) for c in text][:8] or [1]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(chr(ord("a") + (i % 26)) for i in ids)


def _engine(tmp_path):
    model = build_model(
        __import__("apexgpt.core.config", fromlist=["ModelConfig"]).ModelConfig(
            vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32))
    path = tmp_path / "checkpoint.pt"
    model.save(path, step=5)
    engine = InferenceEngine(device="cpu", checkpoint=path)
    engine.load(path)
    engine.tokenizer = StubEngineTokenizer()
    return engine


def test_engine_loads_and_reports_metadata(tmp_path):
    engine = _engine(tmp_path)
    meta = engine.metadata()
    assert meta["step"] == 5
    assert meta["n_layer"] == 1
    assert meta["parameters"] > 0


def test_engine_streams_the_requested_token_count(tmp_path):
    engine = _engine(tmp_path)
    req = GenerationRequest(prompt="hello", max_new_tokens=7, temperature=0.8)
    pieces = list(engine.stream(req))
    assert len(pieces) == 7


def test_engine_generate_text_is_join_of_stream(tmp_path):
    engine = _engine(tmp_path)
    req = GenerationRequest(prompt="hello", max_new_tokens=6, temperature=0.8)
    # sampling is random, so both calls must be seeded to be comparable
    torch.manual_seed(11)
    whole = engine.generate_text(req)
    torch.manual_seed(11)
    streamed = "".join(engine.stream(req))
    assert whole == streamed


def test_request_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        GenerationRequest(prompt="").validate()
    with pytest.raises(ValueError):
        GenerationRequest(prompt="x", max_new_tokens=0).validate()
    with pytest.raises(ValueError):
        GenerationRequest(prompt="x", top_p=1.5).validate()


def test_find_checkpoint_raises_when_absent(tmp_path, monkeypatch):
    from apexgpt.features.inference import service
    monkeypatch.setattr(service, "CHECKPOINTS_DIR", tmp_path)
    monkeypatch.setattr(service, "RUNS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="train one first"):
        service.find_checkpoint()
"""Tokenizer tests: the byte-level one, the recorded spec, and the wiring.

The property that matters is that a corpus's tokens and a checkpoint's vocabulary
can never disagree, so most of these tests are about the spec travelling with
the data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apexgpt.core.config import DataConfig, ModelConfig
from apexgpt.features.data.tokenizers import (BYTE, BYTE_VOCAB_SIZE, GPT2,
                                              ByteTokenizer, TokenizerSpec,
                                              load_tokenizer, read_spec, spec_for,
                                              write_spec)


# ---------------------------------------------------------------- the basics
def test_the_two_tokenizers_are_registered():
    assert spec_for("gpt2") is GPT2
    assert spec_for("char") is BYTE
    assert spec_for(None) is GPT2            # gpt2 is the default
    assert GPT2.vocab_size == 50257
    assert BYTE.vocab_size == 257


@pytest.mark.parametrize("alias", ["byte", "bytes", "char-level", "BPE", "gpt-2"])
def test_tokenizer_aliases_resolve(alias):
    assert spec_for(alias) in (GPT2, BYTE)


def test_an_unknown_tokenizer_lists_the_known_ones():
    with pytest.raises(ValueError) as excinfo:
        spec_for("wordpiece")
    assert "char" in str(excinfo.value) and "gpt2" in str(excinfo.value)


def test_loading_the_byte_tokenizer_needs_no_download():
    tok = load_tokenizer("char")
    assert isinstance(tok, ByteTokenizer)
    assert tok.vocab_size == 257
    assert tok.eos_token_id == 256


# ------------------------------------------------------------ byte encoding
def test_byte_tokenizer_round_trips_ascii():
    tok = ByteTokenizer()
    text = "First Citizen: hear me speak.\n"
    assert tok.decode(tok.encode(text)) == text


def test_byte_tokenizer_round_trips_unicode():
    tok = ByteTokenizer()
    text = "to be, or not to be: café — 北京 \U0001f600"
    ids = tok.encode(text)
    assert all(0 <= i < 256 for i in ids)
    assert tok.decode(ids) == text


def test_byte_tokenizer_ids_are_the_bytes():
    tok = ByteTokenizer()
    assert tok.encode("AB") == [65, 66]


def test_byte_tokenizer_call_matches_the_pipeline_surface():
    tok = ByteTokenizer()
    single = tok("hi", add_special_tokens=False)["input_ids"]
    assert single == [104, 105]
    batch = tok(["hi", "there"], add_special_tokens=False)["input_ids"]
    assert batch == [[104, 105], [116, 104, 101, 114, 101]]
    tensor = tok("hi", return_tensors="pt")["input_ids"]
    assert isinstance(tensor, torch.Tensor) and tensor.shape == (1, 2)


def test_byte_tokenizer_decodes_the_eos_id_to_nothing():
    assert ByteTokenizer().decode([BYTE.eos_token_id]) == ""


def test_byte_tokenizer_survives_a_truncated_multibyte_character():
    """A generation cut mid-character must render, not raise."""
    text = "café"
    partial = ByteTokenizer().encode(text)[:4]      # 'caf' + first byte of é
    assert ByteTokenizer().decode(partial).endswith("\ufffd")


def test_byte_tokenizer_batch_decode():
    tok = ByteTokenizer()
    assert tok.batch_decode([[104, 105], [121]]) == ["hi", "y"]


# ------------------------------------------------------------------- the spec
def test_spec_round_trips_through_disk(tmp_path):
    write_spec(tmp_path, BYTE)
    payload = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "char"
    assert read_spec(tmp_path) == BYTE


def test_a_corpus_without_a_spec_reads_as_gpt2(tmp_path):
    """Corpora built before this existed were all GPT-2."""
    assert read_spec(tmp_path) is None
    from apexgpt.features.data.tokenizers import resolve_corpus_spec
    assert resolve_corpus_spec(tmp_path).kind == "gpt2"


def test_a_corrupt_spec_is_ignored_rather_than_fatal(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{not json", encoding="utf-8")
    assert read_spec(tmp_path) is None


def test_spec_from_dict_rejects_a_payload_without_a_kind():
    with pytest.raises(KeyError):
        TokenizerSpec.from_dict({"vocab_size": 10})


# ------------------------------------------------------------- data service
def test_data_config_tokenizer_follows_the_environment(monkeypatch):
    monkeypatch.setenv("APEXGPT_TOKENIZER", "char")
    assert DataConfig().tokenizer == "char"


def test_prepare_records_the_tokenizer_it_used(tmp_path):
    """The spec is written next to train.bin, so nothing can drift."""
    from apexgpt.features.data.service import prepare

    source = tmp_path / "corpus.txt"
    source.write_text("All's well that ends well.\n" * 200, encoding="utf-8")
    cfg = DataConfig(raw_dir=tmp_path / "raw", binary_dir=tmp_path / "bin")
    cfg.select_dataset("local")
    try:
        result = prepare(cfg, source="local:" + str(source), tokenizer="char",
                         progress=lambda *a: None)
    except Exception as exc:
        if "token" in str(exc).lower() or "connection" in str(exc).lower():
            pytest.skip(f"tokenizer unavailable: {exc}")
        raise

    assert result["tokenizer"] == "char"
    assert read_spec(cfg.binary_dir).kind == "char"
    assert cfg.vocab_size == BYTE_VOCAB_SIZE
    # ids must fit the vocabulary the model will use
    import numpy as np
    train = np.memmap(cfg.train_path, dtype=np.uint16, mode="r")
    assert int(train[:1000].max()) < BYTE_VOCAB_SIZE


# -------------------------------------------------------- train and inference
def _tiny_char_checkpoint(path: Path, tokenizer: str = "char") -> Path:
    from apexgpt.models.gpt import GPT

    spec = spec_for(tokenizer)
    model = GPT(ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=32,
                            vocab_size=spec.vocab_size))
    model.save(path, step=0, extra={"tokenizer": spec.kind,
                                   "vocab_size": spec.vocab_size,
                                   "val_loss": 9.9})
    return path


def test_training_carries_the_corpus_tokenizer_into_the_checkpoint(tmp_path,
                                                                  monkeypatch):
    """A real (tiny) run, end to end: char corpus -> char checkpoint."""
    from apexgpt.core.config import Config
    from apexgpt.core import paths
    from apexgpt.features.data.service import prepare
    from apexgpt.features.training.service import train
    from apexgpt.models.gpt import GPT

    source = tmp_path / "corpus.txt"
    source.write_text("KING RICHARD II:\nWhat must the king do now?\n" * 300,
                      encoding="utf-8")
    cfg = Config()
    cfg.data.select_dataset("local")
    cfg.data.raw_dir = tmp_path / "raw"
    cfg.data.binary_dir = tmp_path / "bin"
    prepare(cfg.data, source="local:" + str(source), tokenizer="char",
            progress=lambda *a: None)

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    result = train(cfg, preset="smoke", max_iters=2, log_interval=1,
                   run_name="char-smoke", progress=lambda *a, **k: None)

    assert cfg.model.vocab_size == BYTE_VOCAB_SIZE
    _, payload = GPT.load(result.checkpoint)
    assert payload["extra"]["tokenizer"] == "char"
    assert payload["extra"]["vocab_size"] == 257
    assert payload["model_config"]["vocab_size"] == 257


def test_engine_decodes_with_the_tokenizer_the_checkpoint_records(tmp_path):
    from apexgpt.features.inference.service import (GenerationRequest,
                                                    InferenceEngine)

    path = _tiny_char_checkpoint(tmp_path / "char.pt")
    engine = InferenceEngine(device="cpu").load(path)
    assert engine.tokenizer_kind == "char"
    assert engine.metadata()["vocab_size"] == 257
    assert engine.metadata()["tokenizer"] == "char"
    ids = engine._encode_prompt("To be,")
    assert ids.shape == (1, len("To be,"))       # one id per byte
    text = engine.generate_text(GenerationRequest(
        prompt="To be,", max_new_tokens=12, temperature=0.8, seed=0))
    assert isinstance(text, str)


def test_engine_rejects_a_checkpoint_whose_vocab_disagrees_with_its_record(tmp_path):
    from apexgpt.features.inference.service import InferenceEngine
    from apexgpt.models.gpt import GPT

    path = tmp_path / "broken.pt"
    model = GPT(ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=32,
                            vocab_size=257))
    model.save(path, extra={"tokenizer": "char", "vocab_size": 50257})
    with pytest.raises(RuntimeError, match="vocab"):
        InferenceEngine(device="cpu").load(path)


def test_an_old_checkpoint_without_metadata_still_loads(tmp_path):
    """Checkpoints saved before the tokenizer was recorded are GPT-2."""
    from apexgpt.features.inference.service import InferenceEngine
    from apexgpt.models.gpt import GPT

    path = tmp_path / "legacy.pt"
    GPT(ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=32,
                    vocab_size=50257)).save(path, extra={"val_loss": 8.0})
    engine = InferenceEngine(device="cpu").load(path)
    assert engine.tokenizer_kind == "gpt2"
    assert engine.metadata()["tokenizer"] == "gpt2"
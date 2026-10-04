"""Corpus source tests: registry, URL/local/Kaggle/HF resolution, conversions.

No network is required: HTTP fetches are exercised against ``file://`` URLs and
the Hugging Face streamer is checked for its failure message, not its download.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apexgpt.core.config import DataConfig
from apexgpt.features.data import sources
from apexgpt.features.data.cli import build_parser, main
from apexgpt.features.data.sources import (REGISTRY, Corpus, convert_to_text,
                                           describe_sources, download_file,
                                           fetch_corpus, legacy_corpus, resolve,
                                           token_estimate)

SHAKESPEARE_LINES = (
    "First Citizen: Before we proceed any further, hear me speak.\n"
    "All: Speak, speak.\n"
    "First Citizen: You are all resolved rather to die than to famish?\n"
)


def file_url(path: Path) -> str:
    return path.as_uri()


# ----------------------------------------------------------------- registry
def test_the_corpus_the_user_asked_for_is_registered():
    assert "shakespeare" in REGISTRY
    shakespeare = REGISTRY["shakespeare"]
    assert shakespeare.kind == "text-url"
    assert shakespeare.locator.endswith("tinyshakespeare/input.txt")
    assert 0.5 < shakespeare.size_mb < 5


def test_every_registered_corpus_is_fully_described():
    for key, corpus in REGISTRY.items():
        assert corpus.key == key
        assert corpus.title and corpus.kind and corpus.locator
        assert corpus.license and corpus.note, f"{key} is undocumented"


def test_default_corpus_is_wikipedia():
    assert sources.DEFAULT_SOURCE == "wikipedia"
    assert resolve().key == "wikipedia"
    assert resolve(None).kind == "wikipedia"


@pytest.mark.parametrize("alias", ["shakespeare", "tiny_shakespeare",
                                  "tinyshakespeare", "shake", "SHAKESPEARE"])
def test_shakespeare_aliases_all_resolve(alias):
    assert resolve(alias).key == "shakespeare"


def test_resolve_accepts_prefixed_forms():
    hf = resolve("hf:roneneldan/TinyStories")
    assert hf.kind == "hf-stream" and hf.locator == "roneneldan/TinyStories"

    kaggle = resolve("kaggle:sahil2901/poem-generator")
    assert kaggle.kind == "kaggle"
    assert kaggle.locator == "sahil2901/poem-generator"

    local = resolve("local:C:/tmp/my corpus.txt")
    assert local.kind == "local" and "my corpus.txt" in local.locator

    url = resolve("url:https://example.invalid/text.txt")
    assert url.kind == "text-url" and url.locator.startswith("https://")


def test_resolve_rejects_an_unknown_source_with_the_known_names():
    with pytest.raises(ValueError) as excinfo:
        resolve("not-a-corpus")
    assert "wikipedia" in str(excinfo.value)


def test_token_estimate_scales_with_size():
    small = Corpus("a", "a", "text-url", "u", size_mb=1.0)
    big = Corpus("b", "b", "text-url", "u", size_mb=10.0)
    assert token_estimate(small) < token_estimate(big)
    assert token_estimate(Corpus("c", "c", "local", "p")) is None


def test_describe_sources_lists_keys_and_build_state(capsys):
    listing = "\n".join(describe_sources())
    for key in REGISTRY:
        assert key in listing
    assert "ready" in listing or "not built" in listing
    assert "public domain" in listing


# -------------------------------------------------------------------- config
def test_each_corpus_gets_its_own_directories():
    cfg = DataConfig()
    assert "wikipedia" in str(cfg.binary_dir)

    cfg.select_dataset("shakespeare")
    assert cfg.dataset == "shakespeare"
    assert cfg.binary_dir.name == "shakespeare"
    assert cfg.raw_dir.name == "shakespeare"
    assert cfg.corpus_path.name == "shakespeare.txt"


def test_switching_corpora_never_leaves_the_old_directory_behind():
    """Assigning .dataset used to leave binary_dir pointing at the old corpus."""
    cfg = DataConfig()
    before = cfg.binary_dir
    cfg.select_dataset("wikitext")
    assert cfg.binary_dir != before
    assert cfg.binary_dir.name == "wikitext"


def test_explicit_directories_survive_a_dataset_switch(tmp_path):
    cfg = DataConfig(binary_dir=tmp_path, raw_dir=tmp_path)
    cfg.select_dataset("shakespeare")
    assert cfg.binary_dir == tmp_path
    assert cfg.raw_dir == tmp_path


def test_legacy_wikipedia_cache_is_only_offered_for_wikipedia(tmp_path):
    (tmp_path / "wiki_corpus.txt").write_text("old", encoding="utf-8")
    assert legacy_corpus(tmp_path, "wikipedia") is not None
    assert legacy_corpus(tmp_path, "shakespeare") is None
    assert legacy_corpus(tmp_path / "empty", "wikipedia") is None


# ------------------------------------------------------------------- fetching
def test_download_file_works_against_a_local_url(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text(SHAKESPEARE_LINES, encoding="utf-8")
    dest = tmp_path / "out" / "corpus.txt"
    download_file(file_url(source), dest, progress=lambda *a: None)
    assert dest.read_text(encoding="utf-8") == SHAKESPEARE_LINES


def test_download_file_rejects_an_empty_payload(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        download_file(file_url(source), tmp_path / "out.txt", progress=lambda *a: None)


def test_download_file_reports_a_bad_url(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(RuntimeError, match="download failed"):
        download_file(file_url(missing), tmp_path / "out.txt", progress=lambda *a: None)


def test_fetch_corpus_from_a_local_file(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text(SHAKESPEARE_LINES, encoding="utf-8")
    dest = tmp_path / "raw" / "local.txt"
    corpus = resolve("local:" + str(source))
    fetch_corpus(corpus, dest, tmp_path, progress=lambda *a: None)
    assert dest.read_text(encoding="utf-8") == SHAKESPEARE_LINES


def test_fetch_corpus_reports_a_missing_local_file(tmp_path):
    corpus = resolve("local:" + str(tmp_path / "absent.txt"))
    with pytest.raises(FileNotFoundError, match="local corpus not found"):
        fetch_corpus(corpus, tmp_path / "out.txt", tmp_path, progress=lambda *a: None)


def test_fetch_corpus_over_a_url_source(tmp_path):
    upstream = tmp_path / "tinyshakespeare.txt"
    upstream.write_text(SHAKESPEARE_LINES, encoding="utf-8")
    corpus = Corpus("shakespeare", "tiny shakespeare", "text-url", file_url(upstream))
    dest = tmp_path / "raw" / "shakespeare.txt"
    fetch_corpus(corpus, dest, tmp_path, progress=lambda *a: None)
    assert "First Citizen" in dest.read_text(encoding="utf-8")


def test_fetch_corpus_reuses_an_existing_cache(tmp_path):
    dest = tmp_path / "shakespeare.txt"
    dest.write_text("cached", encoding="utf-8")
    corpus = resolve("shakespeare")
    result = fetch_corpus(corpus, dest, tmp_path, progress=lambda *a: None)
    assert result == dest
    assert dest.read_text(encoding="utf-8") == "cached"


def test_fetch_corpus_rejects_a_kind_the_service_owns(tmp_path):
    corpus = resolve("wikipedia")
    with pytest.raises(ValueError, match="handled by the data service"):
        fetch_corpus(corpus, tmp_path / "wiki.txt", tmp_path, progress=lambda *a: None)


def test_hf_stream_failure_names_the_dataset(tmp_path):
    """A bad repo id must say so, not fail deep inside the reader."""
    try:
        import datasets                                # noqa: F401
    except Exception as exc:
        # An application-control policy can block a pyarrow DLL, which surfaces
        # as an ImportError from inside datasets. That is the machine, not the code.
        pytest.skip(f"datasets/pyarrow unusable here: {type(exc).__name__}: {exc}")
    corpus = Corpus("x", "x", "hf-stream", "apexgpt/definitely-not-a-dataset")
    with pytest.raises(RuntimeError) as excinfo:
        sources.stream_hf_dataset(corpus, tmp_path / "out.txt", target_mb=1,
                                  progress=lambda *a: None)
    message = str(excinfo.value)
    assert "definitely-not-a-dataset" in message
    assert "--source local" in message


def test_hf_stream_without_datasets_explains_the_install(tmp_path, monkeypatch):
    """A blocked or missing pyarrow must not surface as an ImportError traceback."""
    monkeypatch.setitem(sys.modules, "datasets", None)
    corpus = Corpus("x", "x", "hf-stream", "owner/name")
    with pytest.raises(RuntimeError, match="pip install datasets pyarrow"):
        sources.stream_hf_dataset(corpus, tmp_path / "out.txt", target_mb=1,
                                  progress=lambda *a: None)


def test_kaggle_without_a_client_explains_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "kagglehub", None)
    monkeypatch.setattr(sources.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="kagglehub"):
        sources.download_kaggle("owner/slug", tmp_path, progress=lambda *a: None)


# ----------------------------------------------------------------- conversion
def test_convert_plain_text_is_a_copy(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("hello\n", encoding="utf-8")
    dest = convert_to_text(src, tmp_path / "out.txt", progress=lambda *a: None)
    assert dest.read_text(encoding="utf-8") == "hello\n"


def test_convert_csv_joins_the_columns(tmp_path):
    src = tmp_path / "a.csv"
    src.write_text("title,body\nHamlet,to be or not to be\n", encoding="utf-8")
    dest = convert_to_text(src, tmp_path / "out.txt", progress=lambda *a: None)
    text = dest.read_text(encoding="utf-8")
    assert "Hamlet to be or not to be" in text


def test_convert_jsonl_picks_the_text_field(tmp_path):
    src = tmp_path / "a.jsonl"
    src.write_text(json.dumps({"text": "first", "meta": 1}) + "\n"
                   + json.dumps({"content": "second"}) + "\n", encoding="utf-8")
    dest = convert_to_text(src, tmp_path / "out.txt", progress=lambda *a: None)
    assert dest.read_text(encoding="utf-8").split() == ["first", "second"]


def test_convert_json_list(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps([{"body": "alpha"}, {"body": "beta"}]), encoding="utf-8")
    dest = convert_to_text(src, tmp_path / "out.txt", progress=lambda *a: None)
    assert dest.read_text(encoding="utf-8").split() == ["alpha", "beta"]


def test_convert_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = tmp_path / "a.parquet"
    pq.write_table(pa.table({"text": ["one", "two"]}), src)
    dest = convert_to_text(src, tmp_path / "out.txt", progress=lambda *a: None)
    assert dest.read_text(encoding="utf-8").split() == ["one", "two"]


def test_conversion_failure_on_an_empty_file_cleans_up(tmp_path):
    src = tmp_path / "a.json"
    src.write_text("[]", encoding="utf-8")
    dest = tmp_path / "out.txt"
    with pytest.raises(RuntimeError, match="empty corpus"):
        convert_to_text(src, dest, progress=lambda *a: None)
    assert not dest.exists()


# ---------------------------------------------------------------------- cli
def test_data_cli_parses_the_source_flag():
    args = build_parser().parse_args(["prepare", "--source", "shakespeare"])
    assert args.command == "prepare"
    assert args.source == "shakespeare"
    assert args.dataset is None


def test_data_cli_defaults_to_prepare():
    assert build_parser().parse_args([]).command == "prepare"


def test_data_cli_rejects_an_unknown_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["frobnicate"])


def test_data_cli_sources_lists_the_registry(capsys):
    assert main(["sources"]) == 0
    out = capsys.readouterr().out
    assert "shakespeare" in out and "kaggle" in out


def test_data_cli_reports_an_unknown_source(capsys):
    assert main(["prepare", "--source", "not-a-corpus"]) == 1
    assert "unknown data source" in capsys.readouterr().out


# --------------------------------------------------------------- end to end
def test_prepare_builds_tokens_from_a_local_file(tmp_path):
    """The whole path: local text -> uint16 binaries -> next-token pairs."""
    from apexgpt.features.data.service import TokenBatcher, prepare
    from apexgpt.features.data.tokenizers import load_tokenizer

    # The only step that leaves the machine. Checked on its own so that a hub
    # outage or a rate limit skips the test instead of failing it - and so a
    # real defect in prepare() still fails it.
    try:
        load_tokenizer("gpt2")
    except Exception as exc:
        pytest.skip(f"GPT-2 tokenizer unavailable: {type(exc).__name__}: {exc}")

    source = tmp_path / "corpus.txt"
    source.write_text(SHAKESPEARE_LINES * 40, encoding="utf-8")

    cfg = DataConfig(raw_dir=tmp_path / "raw", binary_dir=tmp_path / "bin")
    cfg.select_dataset("local")
    result = prepare(cfg, source="local:" + str(source),
                     progress=lambda *a: None)

    assert result["dataset"] == "local"
    assert cfg.is_ready()
    assert result["train_tokens"] > 0
    assert cfg.train_path.stat().st_size == result["train_tokens"] * 2

    with TokenBatcher(cfg.binary_dir, batch_size=2, block_size=8) as batcher:
        x, y = batcher.get_batch("train")
    assert x.shape == y.shape == (2, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])          # next-token prediction
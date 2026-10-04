"""``apexgpt data tokens``: the corpus token id table.

Every id the model is fed, with the text it stands for, how often it occurs and
what share of the corpus that is - plus the successor ranking behind
``--predict``. These tests use the byte-level ``char`` tokenizer so they run
offline: no GPT-2 download, no network.
"""
from __future__ import annotations

import re

import pytest

from apexgpt.features.data.cli import (build_parser, corpus_path_for, main,
                                      source_slug, token_table)
from apexgpt.features.data.service import build_inventory
from apexgpt.features.data.sources import resolve
from apexgpt.features.data.tokenizers import load_tokenizer


@pytest.fixture(scope="module")
def char_tokenizer():
    return load_tokenizer("char")


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abab abab\nabab\n", encoding="utf-8")
    return path


def ids_in(out: str) -> list[int]:
    """The left column of the table: every ``id  token  count  share  rank`` row."""
    rows = re.findall(r"^\s{2,}(\d+)\s\s\S*.*#\d+\s*$", out, re.MULTILINE)
    return [int(value) for value in rows]


def test_parser_accepts_the_tokens_command():
    args = build_parser().parse_args(["tokens", "--top", "5", "--predict", "ab"])
    assert args.command == "tokens"
    assert args.top == 5
    assert args.predict == "ab"


def test_table_lists_every_id_with_its_count_and_share(corpus, char_tokenizer,
                                                      capsys):
    token_table(corpus, char_tokenizer)
    out = capsys.readouterr().out
    assert "TOKEN TABLE" in out
    assert "vocabulary     : 257 ids" in out
    assert "distinct ids   : 4" in out
    body = out.split("rank\n", 1)[1]
    # 15 characters over 4 distinct ids, by id order and with escaped whitespace
    assert ids_in(body) == sorted(ids_in(body))
    assert "\\n" in body and "\\u0020" in body
    assert f"{100.0 * 6 / 15:.3f}%" in body  # 'a' is 6 of the 15 characters


def test_top_slices_to_the_most_frequent_ids(corpus, char_tokenizer, capsys):
    token_table(corpus, char_tokenizer, top=2)
    out = capsys.readouterr().out
    body = out.split("rank\n", 1)[1]
    assert len(ids_in(body)) == 2
    assert "-- the 2 most frequent of 4 ids" in out


def test_limit_caps_rows_without_reordering(corpus, char_tokenizer, capsys):
    token_table(corpus, char_tokenizer, limit=2)
    out = capsys.readouterr().out
    body = out.split("rank\n", 1)[1]
    assert len(ids_in(body)) == 2
    assert "use --limit 0 to see them all" in out


def test_slice_at_prints_the_ids_of_a_character_offset(corpus, char_tokenizer,
                                                      capsys):
    token_table(corpus, char_tokenizer, show_text=1)
    out = capsys.readouterr().out
    text = "abab abab\nabab\n"
    end = min(len(text), 1 + 24)
    ids = char_tokenizer(text[1:end], add_special_tokens=False)["input_ids"]
    assert f"ids [1:{end}] = {[int(i) for i in ids]}" in out


def test_predict_ranks_successors_by_bigram_counts(corpus, char_tokenizer,
                                                   capsys):
    # 'a' is always followed by 'b' in this corpus
    token_table(corpus, char_tokenizer, predict="a", top_k=3)
    out = capsys.readouterr().out
    assert "next token, from bigram counts" in out
    assert "p=100.000%" in out
    ranked = re.findall(r"^\s+\d+\. id=(\d+)\s+count=\s*(\d+)", out, re.MULTILINE)
    assert len(ranked) == 1                # only 'b' ever follows 'a' here
    b_id = char_tokenizer("b", add_special_tokens=False)["input_ids"][0]
    assert ranked[0] == (str(b_id), "6")   # 'a' occurs 6 times in the corpus


def test_predict_reports_an_unseen_id(corpus, char_tokenizer, capsys):
    never_seen = load_tokenizer("char")(
        "\u2603", add_special_tokens=False)["input_ids"][0]
    token_table(corpus, char_tokenizer, predict="\u2603")
    out = capsys.readouterr().out
    if never_seen == 0:                    # the byte is present in the corpus
        pytest.skip("this corpus contains the prompt byte")
    assert "no successor was observed for this id" in out


def test_main_runs_tokens_on_a_local_file(tmp_path, monkeypatch, capsys):
    source = tmp_path / "local.txt"
    source.write_text("abab abab\nabab\n", encoding="utf-8")
    monkeypatch.setenv("APEXGPT_DATA_DIR", str(tmp_path / "data"))
    assert main(["tokens", "--source", "local:" + str(source),
                 "--tokenizer", "char", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "TOKEN TABLE" in out
    assert "-- the 2 most frequent" in out
    assert "train --dataset local --tokenizer char" in out


def test_main_refuses_to_download_wikipedia_for_a_token_table(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    monkeypatch.setenv("APEXGPT_DATA_DIR", str(tmp_path / "data"))
    assert main(["tokens", "--source", "wikipedia"]) == 1
    err = capsys.readouterr().err
    assert "is not built" in err
    assert "data prepare --source wikipedia" in err


# ---------------------------------------------------- the shared inventory
def test_inventory_counts_ids_and_keeps_the_pairs(corpus, char_tokenizer):
    inventory = build_inventory(corpus, char_tokenizer)
    assert inventory.total == 15
    assert inventory.distinct == 4
    assert inventory.characters == 15
    a = char_tokenizer("a", add_special_tokens=False)["input_ids"][0]
    assert inventory.counts[a] == 6
    value = inventory.value_of(a, char_tokenizer)
    assert (value.count, value.share) == (6, pytest.approx(6 / 15))
    assert value.label == "a"


def test_inventory_escapes_whitespace_in_a_label(corpus, char_tokenizer):
    inventory = build_inventory(corpus, char_tokenizer)
    space = char_tokenizer(" ", add_special_tokens=False)["input_ids"][0]
    newline = char_tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    assert inventory.value_of(space, char_tokenizer).label == "\\u0020"
    assert inventory.value_of(newline, char_tokenizer).label == "\\n"


def test_successors_are_ranked_pairs_not_frequencies(corpus, char_tokenizer):
    inventory = build_inventory(corpus, char_tokenizer)
    a = int(char_tokenizer("a", add_special_tokens=False)["input_ids"][0])
    b = int(char_tokenizer("b", add_special_tokens=False)["input_ids"][0])
    ranked = inventory.successors_of(a, top_k=3)
    assert [(token_id, count) for token_id, count, _ in ranked] == [(b, 6)]
    assert ranked[0][2] == pytest.approx(1.0)


def test_successors_of_an_unseen_id_are_empty(corpus, char_tokenizer):
    inventory = build_inventory(corpus, char_tokenizer)
    snow = int(char_tokenizer("\u2603", add_special_tokens=False)["input_ids"][0])
    if snow in inventory.counts:
        pytest.skip("this corpus contains that byte")
    assert inventory.successors_of(snow) == []


def test_pairs_are_counted_once_and_only_when_asked_for(corpus, char_tokenizer):
    inventory = build_inventory(corpus, char_tokenizer)
    assert inventory._successors is None          # a plain run must not pay for it
    first = inventory.successors
    assert inventory.successors is first          # and the table is built once


def test_values_for_keeps_the_prompt_order(corpus, char_tokenizer):
    inventory = build_inventory(corpus, char_tokenizer)
    ids = [int(i) for i in
           char_tokenizer("ba", add_special_tokens=False)["input_ids"]]
    assert [v.token_id for v in inventory.values_for(ids, char_tokenizer)] == ids


def test_two_local_files_are_not_read_from_one_cache(tmp_path, monkeypatch,
                                                    capsys):
    """A second ``local:`` file must not be answered with the first one's text.

    Every ``local:`` spec resolves to the corpus key ``local``, so a cache named
    after the key would hand the second file back the first file's tokens.
    """
    raw = tmp_path / "data"
    monkeypatch.setenv("APEXGPT_DATA_DIR", str(raw))
    first = tmp_path / "first.txt"
    first.write_text("aaaa\n", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("zzzz\n", encoding="utf-8")

    assert main(["tokens", "--source", "local:" + str(first),
                 "--tokenizer", "char", "--top", "2"]) == 0
    assert "reading " + str(first) in capsys.readouterr().out

    assert main(["tokens", "--source", "local:" + str(second),
                 "--tokenizer", "char", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "reading " + str(second) in out
    assert str(first) not in out


def test_a_local_text_file_is_read_where_it_lies(tmp_path, monkeypatch):
    source = tmp_path / "notes.md"
    source.write_text("hello\n", encoding="utf-8")
    assert corpus_path_for(resolve("local:" + str(source)),
                           tmp_path / "raw") == source


def test_a_converted_local_file_is_cached_under_its_own_name(tmp_path):
    source = tmp_path / "rows.csv"
    source.write_text("alpha,beta\n", encoding="utf-8")
    cached = corpus_path_for(resolve("local:" + str(source)), tmp_path / "raw")
    assert cached.parent == tmp_path / "raw"
    assert cached.name.endswith("-rows.txt")
    assert cached.read_text(encoding="utf-8") == "alpha beta\n"


def test_two_local_files_of_one_name_get_different_cache_slugs(tmp_path):
    raw = tmp_path / "raw"
    paths = []
    for folder in ("a", "b"):
        directory = tmp_path / folder
        directory.mkdir()
        source = directory / "x.csv"
        source.write_text("one,two\n", encoding="utf-8")
        paths.append(corpus_path_for(resolve("local:" + str(source)), raw))
    assert len(set(paths)) == 2


def test_the_slug_survives_a_long_path(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    source = deep / "corpus.csv"
    source.write_text("x,y\n", encoding="utf-8")
    cached = corpus_path_for(resolve("local:" + str(source)), tmp_path / "raw")
    assert cached.name == "d-corpus.txt"


def test_a_url_corpus_is_named_after_its_url():
    assert source_slug(resolve("url:https://example.com/dir/My Corpus.txt")) \
        == "example.com-dir-My-Corpus"


def test_a_kaggle_corpus_is_named_after_its_slug():
    assert source_slug(resolve("kaggle:owner/dataset-slug")) \
        == "owner-dataset-slug"


def test_the_local_source_hint_names_the_source_not_just_the_key(tmp_path,
                                                                 monkeypatch,
                                                                 capsys):
    source = tmp_path / "local.txt"
    source.write_text("abab\n", encoding="utf-8")
    monkeypatch.setenv("APEXGPT_DATA_DIR", str(tmp_path / "data"))
    main(["tokens", "--source", "local:" + str(source), "--tokenizer", "char"])
    out = capsys.readouterr().out
    assert f"data prepare --source local:{source}" in out
    assert "train --dataset local --tokenizer char" in out
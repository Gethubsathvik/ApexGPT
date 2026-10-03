"""``apexgpt data tokens``: the corpus token id table.

Every id the model is fed, with the text it stands for, how often it occurs and
what share of the corpus that is - plus the successor ranking behind
``--predict``. These tests use the byte-level ``char`` tokenizer so they run
offline: no GPT-2 download, no network.
"""
from __future__ import annotations

import re

import pytest

from apexgpt.features.data.cli import build_parser, main, token_table
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
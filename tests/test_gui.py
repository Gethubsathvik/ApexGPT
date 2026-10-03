"""Headless GUI tests.

Builds the real Tk window (withdrawn) and drives it. Skipped when no display
is available, so the suite still passes over SSH or in CI without a desktop.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("tkinter", reason="tkinter is not available")

tk = pytest.importorskip("tkinter")

from apexgpt.core.config import ModelConfig
from apexgpt.features.inference.gui import ApexGPTApp
from apexgpt.features.inference.service import InferenceEngine
from apexgpt.models.builder import build_model


def _has_display() -> bool:
    try:
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_display(), reason="no display available")


class StubTokenizer:
    # unsampleable id so generation never stops early mid-assertion
    eos_token_id = 999
    vocab_size = 64

    def __call__(self, text, return_tensors=None):
        ids = [1 + (ord(c) % 60) for c in text][:8] or [1]
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(chr(ord("a") + (i % 26)) for i in ids)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    model = build_model(ModelConfig(vocab_size=64, block_size=16, n_layer=1,
                                    n_head=2, n_embd=32))
    path = tmp_path_factory.mktemp("gui") / "checkpoint.pt"
    model.save(path, step=5)
    eng = InferenceEngine(device="cpu", checkpoint=path)
    eng.load(path)
    eng.tokenizer = StubTokenizer()
    return eng


@pytest.fixture(scope="module")
def app(engine):
    root = tk.Tk()
    root.withdraw()
    widget = ApexGPTApp(root, engine)
    root.update()
    yield widget
    root.destroy()


def _run(app, timeout=120):
    """Start generation and pump the Tk loop until the worker finishes."""
    deadline = time.time() + timeout
    app.root.update()
    while time.time() < deadline and app.thread and app.thread.is_alive():
        app.root.update()
        time.sleep(0.02)
    for _ in range(40):
        app.root.update()
        time.sleep(0.02)


def test_window_builds_with_all_controls(app):
    assert app.btn_run is not None
    assert app.btn_stop is not None
    assert set(app.vars) == {"temperature", "top_k", "top_p",
                              "max_new_tokens", "seed", "penalty"}


def test_sliders_write_through(app):
    app.vars["temperature"].set(0.35)
    app.vars["top_k"].set(20)
    app.root.update()
    assert app.vars["temperature"].get() == pytest.approx(0.35)
    assert app.vars["top_k"].get() == pytest.approx(20)


def test_prompt_round_trips(app):
    app.prompt.delete("1.0", "end")
    app.prompt.insert("1.0", "The capital city of")
    assert app.prompt.get("1.0", "end").strip() == "The capital city of"


def test_generation_streams_into_the_output_pane(app):
    app.prompt.delete("1.0", "end")
    app.prompt.insert("1.0", "The capital city of")
    app.vars["max_new_tokens"].set(12)
    app.vars["seed"].set(3)
    app.start()
    _run(app)

    text = app.output.get("1.0", "end").strip()
    assert text.startswith("The capital city of")
    assert len(text) > len("The capital city of")
    status = app.status.cget("text")
    assert "12 tokens" in status
    assert "tok/s" in status
    assert str(app.btn_run.cget("state")) == "normal"


def test_empty_prompt_is_rejected(app):
    app.prompt.delete("1.0", "end")
    app.start()
    assert "prompt" in app.status.cget("text").lower()


def test_clear_empties_the_output(app):
    app.clear()
    assert app.output.get("1.0", "end").strip() == ""


def test_the_next_token_panel_is_on_by_default_and_survives_a_run(app):
    assert app.predict_next.get() is True
    app.prompt.delete("1.0", "end")
    app.prompt.insert("1.0", "The capital city of")
    app.vars["max_new_tokens"].set(6)
    app.start()
    _run(app)
    text = app.output.get("1.0", "end")
    assert "next token:" in text
    assert "entropy" in text and "nats" in text


def test_the_next_token_panel_can_be_turned_off(app):
    app.predict_next.set(False)
    app.prompt.delete("1.0", "end")
    app.prompt.insert("1.0", "The capital city of")
    app.vars["max_new_tokens"].set(6)
    app.start()
    _run(app)
    assert "next token:" not in app.output.get("1.0", "end")
    app.predict_next.set(True)


def test_stop_button_halts_generation(app):
    app.prompt.delete("1.0", "end")
    app.prompt.insert("1.0", "In a distant galaxy")
    app.vars["max_new_tokens"].set(2000)
    app.start()
    app.root.update()
    time.sleep(0.3)
    app.stop()
    _run(app, timeout=60)
    assert not (app.thread and app.thread.is_alive())
    app.vars["max_new_tokens"].set(200)
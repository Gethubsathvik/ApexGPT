"""Next-token prediction: the distribution, and running Hub models.

Nothing here touches the network. ``run_model`` is tested against a stub of
``transformers`` because downloading a real 500 MB checkpoint on every test run
would make the suite useless.
"""
from __future__ import annotations

import math

import pytest
import torch

from apexgpt.features.inference.service import (GenerationRequest,  # noqa: F401
                                               InferenceEngine, Prediction)
from apexgpt.models.builder import build_model
from apexgpt.models.sampling import next_token_distribution


class _StubTokenizer:
    eos_token_id = 0
    vocab_size = 8

    def __call__(self, text, return_tensors=None):
        ids = [ord(c) % 8 for c in text][:8]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


def _engine(tmp_path):
    from apexgpt.core.config import ModelConfig
    model = build_model(ModelConfig(vocab_size=8, block_size=16, n_layer=1,
                                   n_head=2, n_embd=32))
    path = tmp_path / "checkpoint.pt"
    model.save(path, step=3)
    engine = InferenceEngine(device="cpu", checkpoint=path)
    engine.load(path)
    engine.tokenizer = _StubTokenizer()
    return engine


# --------------------------------------------------------------------------- #
# next_token_distribution
# --------------------------------------------------------------------------- #
def test_distribution_is_sorted_and_normalised():
    logits = torch.tensor([[[0.5, 3.0, -1.0, 2.0]]])
    ids, probs, entropy = next_token_distribution(logits, top_k=3)
    assert ids.tolist() == [1, 3, 0]
    assert probs[0] > probs[1] > probs[2]
    assert probs.sum() <= 1.0 + 1e-6


def test_distribution_entropy_of_a_uniform_row_is_ln_vocab():
    logits = torch.zeros(1, 1, 50)
    _, probs, entropy = next_token_distribution(logits, top_k=5)
    assert entropy == pytest.approx(math.log(50), abs=1e-5)
    # top-5 of 50 uniform tokens carry exactly a tenth of the mass
    assert probs.sum() == pytest.approx(5 / 50, abs=1e-6)


def test_a_confident_row_has_near_zero_entropy():
    logits = torch.tensor([[[0.0, 60.0, 0.0, 0.0]]])
    _, probs, entropy = next_token_distribution(logits, top_k=2)
    assert probs[0] == pytest.approx(1.0, abs=1e-6)
    assert entropy < 1e-4


def test_distribution_caps_top_k_at_the_vocabulary():
    logits = torch.zeros(1, 1, 4)
    ids, probs, _ = next_token_distribution(logits, top_k=99)
    assert ids.numel() == 4 and probs.numel() == 4


def test_temperature_flattens_the_distribution():
    logits = torch.tensor([[[0.0, 10.0]]])
    _, hot, _ = next_token_distribution(logits, top_k=2, temperature=1.0)
    _, warm, _ = next_token_distribution(logits, top_k=2, temperature=10.0)
    _, flat, _ = next_token_distribution(logits, top_k=2, temperature=1000.0)
    # a sharper temperature pushes probability onto the favourite; a flatter one
    # drags every candidate back towards the uniform 0.5
    assert hot[0] > warm[0] > flat[0]
    assert flat[0] == pytest.approx(0.5, abs=1e-2)


# --------------------------------------------------------------------------- #
# engine surface
# --------------------------------------------------------------------------- #
def test_predict_next_ranks_the_same_token_the_model_argmaxes(tmp_path):
    engine = _engine(tmp_path)
    rows = engine.predict_next("hello", top_k=4)
    assert [r.rank for r in rows] == [1, 2, 3, 4]
    assert rows[0].token_id == _argmax_last(engine, "hello")
    assert rows[0].probability >= rows[-1].probability
    assert rows[0].probability == pytest.approx(math.exp(rows[0].logprob),
                                                 rel=1e-4)
    assert 0.0 < rows[0].probability <= 1.0


def _argmax_last(engine, prompt):
    with torch.no_grad():
        logits, _ = engine.model(engine._encode_prompt(prompt))
    return int(logits[:, -1, :].argmax(dim=-1)[0])


def test_predict_next_text_is_the_single_best_row(tmp_path):
    engine = _engine(tmp_path)
    best = engine.predict_next("hello", top_k=4)[0]
    assert engine.predict_next_text("hello") == best.text


def test_predict_next_reports_entropy_of_the_whole_row(tmp_path):
    engine = _engine(tmp_path)
    rows, entropy = engine.predict_next_with_entropy("hello", top_k=2)
    assert len(rows) == 2
    assert entropy < math.log(engine.metadata()["vocab_size"])


def test_predict_next_needs_a_loaded_engine():
    with pytest.raises(RuntimeError):
        InferenceEngine(device="cpu").predict_next("hello")


def test_prediction_labels_whitespace_so_a_row_is_readable():
    row = Prediction(rank=1, token_id=32, text=" ", probability=0.5, logprob=-0.7)
    assert row.label == "\\u0020"
    newline = Prediction(rank=2, token_id=10, text="\n", probability=0.2,
                         logprob=-1.6)
    assert newline.label == "\\n"
    assert Prediction(1, 5, "", 0.1, -2.3).label == "''"
    assert Prediction(1, 5, "ab", 0.1, -2.3).as_dict()["token_id"] == 5


# --------------------------------------------------------------------------- #
# per-token logprobs and stop reasons
# --------------------------------------------------------------------------- #
def test_stream_steps_reports_a_logprob_per_token(tmp_path):
    engine = _engine(tmp_path)
    # stop_on_eos off: the stub's eos id is 0, and a random model will hit it
    request = GenerationRequest(prompt="hello", max_new_tokens=5, temperature=0.8,
                                seed=1, stop_on_eos=False)
    steps = list(engine.stream_steps(request))
    assert len(steps) == 5
    assert all(isinstance(s.token_id, int) and s.token_id >= 0 for s in steps)
    assert all(s.logprob <= 0.0 for s in steps)
    assert "".join(s.text for s in steps) == engine.generate_text(request)


def test_only_the_last_step_is_finished_and_says_why(tmp_path):
    engine = _engine(tmp_path)
    steps = list(engine.stream_steps(GenerationRequest(
        prompt="hello", max_new_tokens=4, temperature=0.0)))
    assert not any(s.finished for s in steps[:-1])
    assert steps[-1].finished and steps[-1].stop_reason == "length"
    assert steps[-1].as_dict()["stop_reason"] == "length"


def test_greedy_decoding_reports_a_zero_logprob(tmp_path):
    engine = _engine(tmp_path)
    steps = list(engine.stream_steps(GenerationRequest(
        prompt="hello", max_new_tokens=3, temperature=0.0)))
    assert [s.logprob for s in steps] == [0.0, 0.0, 0.0]


def test_generate_with_details_is_the_openai_logprobs_shape(tmp_path):
    engine = _engine(tmp_path)
    text, details = engine.generate_with_details(GenerationRequest(
        prompt="hello", max_new_tokens=4, temperature=0.8, seed=2))
    logprobs = details["logprobs"]
    assert len(logprobs["tokens"]) == len(logprobs["token_logprobs"]) == 4
    assert logprobs["tokens"] == list(text)
    assert details["generated_tokens"] == 4
    assert details["finish_reason"] == "length"


def test_filtered_probs_is_a_distribution_even_when_everything_is_filtered():
    from apexgpt.models.sampling import filtered_probs

    logits = torch.tensor([[0.0, 5.0, 7.0]])
    probs = filtered_probs(logits, top_k=1)
    assert probs.shape == logits.shape
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
    # top_k=1 is a one-hot on the argmax, not a broken all-zero row
    assert float(probs[0, 2]) == pytest.approx(1.0, abs=1e-6)
    assert float(probs[0, 0]) == 0.0 and float(probs[0, 1]) == 0.0


def test_greedy_filtered_probs_is_one_hot_on_the_argmax():
    from apexgpt.models.sampling import filtered_probs

    logits = torch.tensor([[0.0, 5.0, 7.0, -1.0]])
    probs = filtered_probs(logits, temperature=0.0)
    assert int(probs.argmax()) == 2
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)


def test_sample_next_token_with_logprob_agrees_with_the_drawn_token():
    from apexgpt.models.sampling import (filtered_probs,
                                         sample_next_token_with_logprob)

    logits = torch.tensor([[0.0, 4.0, 1.0, -2.0]])
    torch.manual_seed(0)
    ids, logprob = sample_next_token_with_logprob(logits, temperature=1.0)
    torch.manual_seed(0)
    again = sample_next_token_with_logprob(logits, temperature=1.0)[0]
    assert int(ids) == int(again)
    probs = filtered_probs(logits, temperature=1.0)
    assert float(logprob) == pytest.approx(
        float(probs[0, int(ids)].clamp_min(1e-12).log()), abs=1e-5)


def test_logprobs_are_measured_before_top_p_truncates():
    """A renormalised top-p of 1.0 would report every sure token as free."""
    from apexgpt.models.sampling import sample_next_token_with_logprob

    # one token holds most of the mass, so top_p=0.5 keeps only that one
    logits = torch.tensor([[6.0, 0.0, 0.0, 0.0]])
    probs = torch.softmax(logits, dim=-1)
    assert float(probs[0, 0]) > 0.95

    ids, logprob = sample_next_token_with_logprob(logits, temperature=1.0, top_p=0.5)
    assert int(ids) == 0                     # top_p collapsed onto it
    assert float(logprob) < 0.0              # not the 0.0 of a one-hot filter
    assert float(logprob) == pytest.approx(
        float(probs[0, 0].clamp_min(1e-12).log()), abs=1e-5)


def test_logprob_is_tempered_when_the_sampler_is_tempered():
    from apexgpt.models.sampling import sample_next_token_with_logprob

    logits = torch.tensor([[1.0, 0.0, -1.0]])
    # top_k=1 pins the same token, so only the temperature can move the number
    _ids, cold = sample_next_token_with_logprob(logits, temperature=0.5, top_k=1)
    _ids, hot = sample_next_token_with_logprob(logits, temperature=2.0, top_k=1)
    # a flatter temperature spreads the mass, so even the favourite token is
    # less probable and its log-probability drops
    assert float(hot) < float(cold)
    assert float(cold) == pytest.approx(
        float(torch.softmax(logits / 0.5, dim=-1)[0, 0].log()), abs=1e-5)


def test_sampling_paths_still_agree_with_each_other():
    """filtered_probs is a refactor; the drawn token must not have changed."""
    from apexgpt.models.sampling import sample_next_token

    logits = torch.tensor([[0.5, 3.0, -1.0, 2.0, 0.1]])
    for temperature in (0.0, 0.5, 1.0, 1.7):
        torch.manual_seed(11)
        first = sample_next_token(logits, temperature=temperature, top_k=3,
                                  top_p=0.9)
        torch.manual_seed(11)
        second = sample_next_token(logits, temperature=temperature, top_k=3,
                                   top_p=0.9)
        assert int(first) == int(second)
        assert 0 <= int(first) < logits.size(-1)


# --------------------------------------------------------------------------- #
# hub
# --------------------------------------------------------------------------- #
def test_hub_check_reports_every_capability():
    from apexgpt.features.hub import check

    labels = [label for label, _, _ in check()]
    assert any("Hugging Face" in label for label in labels)
    assert any("Kaggle" in label for label in labels)
    for _, ok, detail in check():
        assert isinstance(ok, bool)
        assert detail


def test_hub_environment_never_prints_the_token():
    from apexgpt.features.hub import environment, hub_token

    env = environment()
    assert isinstance(env["hf_token"], bool)
    assert env["kaggle_client"] in (None, "kagglehub", "kaggle CLI")
    token = hub_token()
    assert token is None or isinstance(token, str)


def test_hub_download_patterns_skip_the_other_framework_exports():
    from apexgpt.features.hub import DEFAULT_MODEL_PATTERNS

    assert "*.safetensors" in DEFAULT_MODEL_PATTERNS
    assert "*.json" in DEFAULT_MODEL_PATTERNS
    assert not any("onnx" in p for p in DEFAULT_MODEL_PATTERNS)


def test_hub_sends_the_allow_patterns_through(tmp_path, monkeypatch):
    """Regression: --allow used to be ignored on the --predict path."""
    from apexgpt.features.hub import service

    seen = {}

    def fake_download(repo_id, repo_type, revision, allow_patterns, local_dir,
                      progress):
        seen.update(repo_id=repo_id, repo_type=repo_type,
                    allow_patterns=allow_patterns, local_dir=local_dir)
        return local_dir

    monkeypatch.setattr(service, "_snapshot", fake_download)
    service.download_model("gpt2", dest=tmp_path / "gpt2", progress=lambda *_: None)
    assert seen["repo_type"] == "model"
    assert "*.safetensors" in seen["allow_patterns"]

    service.download_model("gpt2", allow_patterns=["only-this"], dest=tmp_path / "g",
                           progress=lambda *_: None)
    assert seen["allow_patterns"] == ["only-this"]


def test_hub_model_repo_id_becomes_a_safe_directory(tmp_path, monkeypatch):
    from apexgpt.features.hub import service

    seen = {}

    def fake_snapshot(repo_id, repo_type, revision, allow_patterns, local_dir,
                      progress):
        seen["local_dir"] = local_dir
        return local_dir

    monkeypatch.setattr(service, "_snapshot", fake_snapshot)
    monkeypatch.setattr(service, "HUB_MODEL_DIR", tmp_path)
    service.download_model("meta-llama/Llama-3.2-1B", progress=lambda *_: None)
    assert seen["local_dir"].name == "meta-llama--Llama-3.2-1B"


def test_hub_survives_a_missing_transformers(monkeypatch):
    from apexgpt.features.hub import HubError, run_model, service

    monkeypatch.setattr(service, "_installed", lambda module: False)
    with pytest.raises(HubError) as excinfo:
        run_model("gpt2", "hi")
    assert "requirements-hub.txt" in str(excinfo.value)


def test_hub_run_model_reads_the_distribution(monkeypatch):
    """A stub transformers module: one forward pass, no download."""
    import sys
    import types
    from importlib.machinery import ModuleSpec

    from apexgpt.features.hub import service

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _FakeModel:
        def __init__(self):
            self.eval_called = False

        def to(self, device):
            return self

        def eval(self):
            self.eval_called = True

        def __call__(self, **kwargs):
            return _Out(torch.tensor([[[0.0, 4.0, 1.0, -2.0]]]))

        def generate(self, **kwargs):
            self.generated_with = kwargs
            return torch.tensor([[1, 2, 3, 4, 5]])

    class _FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids, skip_special_tokens=False):
            return "".join("wxyz"[int(i) % 4] for i in ids)

    module = types.ModuleType("transformers")
    # find_spec() rejects a module whose __spec__ is None, and run_model checks
    # the dependency that way
    module.__spec__ = ModuleSpec("transformers", loader=None)
    module.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _FakeModel())
    module.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _FakeTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(service, "download_model",
                        lambda *a, **k: __import__("pathlib").Path("stub"))

    rows, text = service.run_model("gpt2", "hi", top_k=2, max_new_tokens=4,
                                  device="cpu", progress=lambda *_: None)
    assert [r.token_id for r in rows] == [1, 2]
    assert rows[0].probability > rows[1].probability
    # softmax([0, 4, 1, -2]) -> 0.93407 for the leading candidate
    assert rows[0].probability == pytest.approx(0.9340718, rel=1e-5)
    # the stub returns a fixed 5-token row, 3 of which are the prompt
    assert text == "wx"


def test_hub_cli_check_runs_without_network(capsys):
    from apexgpt.features.hub.cli import main

    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "download Kaggle datasets" in out
    assert "SFTTrainer" in out


def test_hub_cli_reports_a_missing_module_rather_than_a_traceback(capsys,
                                                                 monkeypatch):
    from apexgpt.features.hub import cli, service

    monkeypatch.setattr(service, "_installed", lambda module: False)
    with pytest.raises(SystemExit):
        raise SystemExit(cli.main(["files", "gpt2"]))
    assert "requirements-hub.txt" in capsys.readouterr().err


def test_data_cli_kaggle_source_still_resolves():
    """The Kaggle corpus path is reachable by name, as before."""
    from apexgpt.features.data.sources import resolve

    corpus = resolve("kaggle:user/slug")
    assert corpus.kind == "kaggle"
    assert corpus.locator == "user/slug"
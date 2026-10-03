"""API contract tests.

Exercised against a deliberately tiny randomly-initialised model so the suite
stays fast and never needs a trained checkpoint.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apexgpt.api.server import create_app
from apexgpt.features.inference.service import InferenceEngine
from apexgpt.models.gpt import GPT, ModelConfig

TINY_MODEL = dict(n_layer=2, n_head=2, n_embd=32, block_size=32)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """A real, randomly-initialised checkpoint so the engine has something to load."""
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    GPT(ModelConfig(**TINY_MODEL)).save(path, step=0, extra={"val_loss": 10.5})
    return path


@pytest.fixture(scope="module")
def engine(tiny_checkpoint):
    return InferenceEngine(device="cpu").load(tiny_checkpoint)


@pytest.fixture(scope="module")
def client(engine):
    with TestClient(create_app(engine)) as c:
        yield c


# ------------------------------------------------------------------- health
def test_health_reports_ok_and_metadata(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]["device"] == "cpu"
    assert body["model"]["n_layer"] == TINY_MODEL["n_layer"]


# ----------------------------------------------------------------- generate
def test_generate_returns_text_and_token_counts(client):
    r = client.post("/generate", json={"prompt": "Hi", "max_new_tokens": 4})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"text", "prompt_tokens", "elapsed_s", "finish_reason",
                         "logprobs"}
    assert isinstance(body["text"], str)
    assert body["prompt_tokens"] > 0
    assert body["elapsed_s"] >= 0
    assert body["finish_reason"] == "length"
    # one logprob per generated token, and they are negative or zero
    assert len(body["logprobs"]["tokens"]) == 4
    assert all(lp <= 0 for lp in body["logprobs"]["token_logprobs"])


def test_generate_rejects_an_empty_prompt(client):
    assert client.post("/generate", json={"prompt": ""}).status_code == 422


def test_generate_rejects_an_absurd_token_budget(client):
    r = client.post("/generate", json={"prompt": "Hi", "max_new_tokens": 99999})
    assert r.status_code == 422


# ------------------------------------------------------------------- stream
def test_stream_emits_meta_then_tokens_then_done(client):
    with client.stream("GET", "/stream",
                       params={"prompt": "Hi", "max_new_tokens": 3}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines()
                  if line.startswith("data: ")]
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "done"
    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) == 3
    # each event carries the token and how surprised the model was by it
    assert all(isinstance(e["text"], str) for e in tokens)
    assert all(e["logprob"] <= 0 for e in tokens)
    assert all(e["token_id"] >= 0 for e in tokens)


# ------------------------------------------------------------------ predict
def test_predict_ranks_the_distribution_without_generating(client):
    r = client.post("/predict", json={"prompt": "Hi", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["prompt"] == "Hi"
    assert body["tokenizer"] == "gpt2"
    assert len(body["predictions"]) == 5
    ranks = [p["rank"] for p in body["predictions"]]
    assert ranks == [1, 2, 3, 4, 5]
    probs = [p["probability"] for p in body["predictions"]]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(p["text"] is not None for p in body["predictions"])


def test_predict_reports_entropy_against_a_uniform_model(client):
    body = client.post("/predict", json={"prompt": "Hi", "top_k": 3}).json()
    vocab = client.get("/health").json()["model"]["vocab_size"]
    assert body["uniform_entropy_nats"] == pytest.approx(math.log(vocab), abs=1e-4)
    # an untrained model cannot beat uniform, but it must still be a real number
    assert 0.0 <= body["entropy_nats"] <= body["uniform_entropy_nats"] + 1e-6


def test_predict_validates_like_generate(client):
    assert client.post("/predict", json={"prompt": ""}).status_code == 422
    assert client.post("/predict", json={"prompt": "Hi",
                                         "top_k": -1}).status_code == 422


def test_model_discovery_lists_the_loaded_checkpoint(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["id"] == "apexgpt"
    assert entry["tokenizer"] == "gpt2"
    assert entry["context_length"] > 0
    assert entry["parameters"] > 0


def test_stream_can_be_disabled(engine):
    with TestClient(create_app(engine, streaming=False)) as c:
        assert c.get("/stream", params={"prompt": "Hi"}).status_code == 404


# ------------------------------------------------------- openai compatibility
def test_openai_alias_has_the_documented_shape(client):
    r = client.post("/v1/completions",
                    json={"model": "apexgpt", "prompt": "Hi", "max_tokens": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "apexgpt"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["finish_reason"] == "length"
    # was a hardcoded None; now the model's own uncertainty per token
    logprobs = body["choices"][0]["logprobs"]
    assert len(logprobs["tokens"]) == 4
    assert len(logprobs["token_logprobs"]) == 4
    assert all(lp <= 0 for lp in logprobs["token_logprobs"])
    usage = body["usage"]
    assert usage["total_tokens"] == (usage["prompt_tokens"]
                                      + usage["completion_tokens"])
    assert body["apexgpt"]["device"] == "cpu"


def test_openai_alias_refuses_streaming_with_a_useful_message(client):
    r = client.post("/v1/completions",
                    json={"prompt": "Hi", "stream": True})
    assert r.status_code == 400
    assert "/stream" in r.json()["error"]["message"]


def test_openai_alias_rejects_more_than_one_choice(client):
    r = client.post("/v1/completions", json={"prompt": "Hi", "n": 2})
    assert r.status_code == 422


def test_openai_alias_validates_like_the_native_route(client):
    assert client.post("/v1/completions", json={"prompt": ""}).status_code == 422


# ------------------------------------------------------------------ openapi
def test_openapi_documents_every_public_route(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/health", "/generate", "/predict", "/stream",
                 "/v1/completions", "/v1/models"):
        assert path in paths, f"{path} missing from the OpenAPI schema"
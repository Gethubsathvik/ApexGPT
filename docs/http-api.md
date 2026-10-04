# 🌐 HTTP API

The inference service: one command, every endpoint, and how to reach it from another machine.


# 🌐 HTTP API

Optional, and the one part of ApexGPT that runs as its own process.

## The one-line command

```bash
python -m apexgpt serve
```

That is the whole thing: it binds **http://127.0.0.1:8000** (the defaults) and
finds the most recent real checkpoint on its own, loading the tokenizer that
checkpoint recorded. Written out in full — the same thing, with the defaults
spelled out:

```bash
python -m apexgpt serve --host 127.0.0.1 --port 8000
```

`--host`, `--port`, `--checkpoint` and `--device` only override what it would
have picked. Bind `0.0.0.0` instead of `127.0.0.1` to let other machines on your
network reach it. Check it is alive with:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","model":{"checkpoint":"models/runs/gpt-shakespeare-char/checkpoint.pt",
 "parameters_m":10.8,"block_size":192,"vocab_size":257,"tokenizer":"char",
 "step":400,"val_loss":2.461369639635086,"device":"cpu"}}
```

Interactive API docs, including every field of every request, are at
**http://127.0.0.1:8000/docs**.

## Everything else

```bash
pip install -r requirements-api.txt
python -m apexgpt serve --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | landing page: the endpoints below, and which model is loaded |
| `/health` | GET | liveness + model metadata |
| `/generate` | POST | one-shot generation, JSON in / JSON out, with logprobs |
| `/predict` | POST | the ranked next-token distribution, nothing sampled |
| `/stream` | GET | server-sent events, one `data:` line per token |
| `/v1/completions` | POST | OpenAI-compatible alias for third-party clients |
| `/v1/models` | GET | model discovery, which OpenAI clients probe first |
| `/docs` | GET | interactive OpenAPI docs |

> `/predict` takes the same request body as generation, so the number of
> candidates is **`top_k`** — not `k`. An unrecognised field is ignored, so
> `{"prompt": "…", "k": 5}` quietly returns all 50 rows instead of 5.

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The history of the city is","max_new_tokens":40,"temperature":0.8}'

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The history of the city is","top_k":5}'

curl -N "http://localhost:8000/stream?prompt=hello&max_new_tokens=20"
```

```
data: {"type": "meta", "parameters_m": 30.0, ...}
data: {"type": "token", "text": " List", "token_id": 2534, "logprob": -1.83, "stop_reason": null}
data: {"type": "token", "text": " Faction", "token_id": 18965, "logprob": -2.41, "stop_reason": null}
data: {"type": "done"}
```

Every event carries the token's own log-probability — the number the training
loop minimises — so a client can score the model instead of only reading it:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"apexgpt","prompt":"hello","max_tokens":20}'
```

```json
{
  "id": "cmpl-1759000000",
  "object": "text_completion",
  "choices": [{
    "index": 0,
    "text": "...",
    "logprobs": {"tokens": [" List", " Faction"], "token_logprobs": [-1.83, -2.41]},
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 2, "completion_tokens": 20, "total_tokens": 22},
  "apexgpt": {"elapsed_s": 0.31, "device": "cuda:0", "...": "..."}
}
```

`logprobs` used to be hardcoded to `null` and `finish_reason` to `"length"`; both
are now measured — `finish_reason` is `eos` when the model emitted the end token
and `length` when it ran out of budget.

> `stream: true` is refused there with a `400` pointing at `/stream`: OpenAI
> streams token objects, ApexGPT streams bare text, and faking that shape would
> break more clients than it would serve.

This is what makes the project usable from a phone or another program without
linking against Python.

**Security.** No authentication, no rate limiting, no TLS. Bind to `127.0.0.1`
unless you mean otherwise.

---


---

Back to [the README](../README.md).


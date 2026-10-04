"""Optional HTTP inference service.

This is the one part of ApexGPT that runs as a standalone process. It exists so
the model can be shared by other programs - a phone, a web front end, a second
machine - without linking against Python.

Scope note: ApexGPT deliberately stops at a single service. Turning data
preparation and training into separate network services would add ports,
health checks and partial-failure modes to a workflow that is offline and
single-user, without making it better. The feature slices are already isolated
behind ``features/<name>/service.py``, so any of them can be lifted into its own
deployable later without touching the rest.

Run with::

    pip install -r requirements-api.txt
    python -m apexgpt serve --port 8000
"""
import argparse
import json
import math
import os
import sys
import threading
import time

from .. import __version__
from ..features.inference.service import GenerationRequest, InferenceEngine

_LOCK = threading.Lock()   # one generation at a time; the engine is not reentrant


def _pydantic_models():
    """Define the request/response models at module import time.

    They cannot live inside ``create_app``: with postponed annotation evaluation
    they would become unresolved forward references.
    """
    from pydantic import BaseModel, Field

    class GenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1, description="prompt text")
        max_new_tokens: int = Field(200, ge=1, le=4096)
        temperature: float = Field(0.8, ge=0.0, le=5.0)
        top_k: int = Field(50, ge=0, description="0 disables top-k")
        top_p: float = Field(0.95, gt=0.0, le=1.0)
        repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
        seed: int | None = None
        use_cache: bool = True

        def to_request(self) -> GenerationRequest:
            return GenerationRequest(
                prompt=self.prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k or None,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                seed=self.seed,
                use_cache=self.use_cache,
            )

    class GenerateResponse(BaseModel):
        text: str
        prompt_tokens: int
        elapsed_s: float
        finish_reason: str = "length"
        logprobs: dict | None = None

    class PredictResponse(BaseModel):
        prompt: str
        tokenizer: str
        entropy_nats: float
        uniform_entropy_nats: float
        predictions: list[dict]

    class CompletionRequest(BaseModel):
        """OpenAI-compatible subset of ``/v1/completions``."""
        model: str = "apexgpt"
        prompt: str = Field(..., min_length=1)
        max_tokens: int = Field(200, ge=1, le=4096)
        temperature: float = Field(0.8, ge=0.0, le=5.0)
        top_p: float = Field(0.95, gt=0.0, le=1.0)
        n: int = Field(1, ge=1, le=1, description="only n=1 is supported")
        stream: bool = False
        seed: int | None = None
        stop: list[str] | None = None

    return GenerateRequest, GenerateResponse, PredictResponse, CompletionRequest


def create_app(engine: InferenceEngine, streaming: bool = True):
    """Build the FastAPI application around a loaded engine."""
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import (HTMLResponse, JSONResponse, Response,
                                       StreamingResponse)
    except ImportError as exc:
        raise SystemExit(
            "FastAPI is not installed.\n"
            "Install the API extra first:  pip install -r requirements-api.txt"
        ) from exc

    (GenerateRequest, GenerateResponse, PredictResponse,
     CompletionRequest) = _pydantic_models()

    app = FastAPI(
        title="ApexGPT inference API",
        version=__version__,
        description="Next-token generation from an ApexGPT GPT checkpoint.",
    )

    @app.get("/", response_class=HTMLResponse)
    def index():
        """A landing page, so the address in the banner is not a bare 404."""
        meta = engine.metadata()
        rows = "\n".join(
            f'<li><a href="{path}"><code>{path}</code></a> &mdash; {note}</li>'
            for path, note in (
                ("/docs", "interactive documentation for every endpoint"),
                ("/health", "liveness probe and model metadata"),
                ("/generate", "POST a prompt, get a continuation"),
                ("/predict", "POST a prompt, get the ranked next token"),
                ("/stream", "server-sent events, one token at a time"),
                ("/v1/completions", "OpenAI-compatible completions"),
                ("/v1/models", "model discovery for OpenAI clients"),
            ))
        return (f"<h1>ApexGPT {__version__}</h1>"
                f"<p>{meta['parameters_m']}M parameters, {meta['device']}, "
                f"tokenizer {meta['tokenizer']}</p><ul>{rows}</ul>")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        """Browsers ask for this on every page; an empty answer keeps it quiet."""
        return Response(status_code=204)

    @app.get("/health")
    def health():
        """Liveness probe plus model metadata."""
        return {"status": "ok", "model": engine.metadata()}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        """Generate a complete continuation and return it as JSON."""
        request = req.to_request()
        with _LOCK:
            t0 = time.time()
            text, details = engine.generate_with_details(request)
            elapsed = time.time() - t0
        return GenerateResponse(
            text=text,
            prompt_tokens=len(engine.tokenizer(request.prompt)["input_ids"]),
            elapsed_s=round(elapsed, 4),
            finish_reason=details["finish_reason"],
            logprobs=details["logprobs"],
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: GenerateRequest):
        """Rank the next token without sampling anything.

        The same answer ``apexgpt generate --predict`` prints, for programs that
        want the distribution rather than a continuation: a client-side
        autocomplete, a confidence gate, or a scorer.
        """
        request = req.to_request()
        top_k = req.top_k or 10
        with _LOCK:
            rows, entropy = engine.predict_next_with_entropy(
                req.prompt, top_k=top_k, temperature=1.0)
        vocab = engine.metadata().get("vocab_size") or 0
        return PredictResponse(
            prompt=req.prompt,
            tokenizer=engine.tokenizer_kind or "gpt2",
            entropy_nats=round(entropy, 6),
            uniform_entropy_nats=round(math.log(vocab), 6) if vocab > 1 else 0.0,
            predictions=[row.as_dict() for row in rows],
        )

    if streaming:
        @app.get("/stream")
        def stream(
            prompt: str = Query(..., min_length=1),
            max_new_tokens: int = Query(200, ge=1, le=4096),
            temperature: float = Query(0.8, ge=0.0, le=5.0),
            top_k: int = Query(50, ge=0),
            top_p: float = Query(0.95, gt=0.0, le=1.0),
            repetition_penalty: float = Query(1.0, ge=1.0, le=2.0),
            seed: int | None = Query(None),
            use_cache: bool = Query(True),
        ):
            """Server-sent events: one ``data:`` line per generated token."""
            request = GenerationRequest(
                prompt=prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k or None, top_p=top_p,
                repetition_penalty=repetition_penalty, seed=seed,
                use_cache=use_cache,
            )
            try:
                request.validate()
            except ValueError as exc:
                return JSONResponse(status_code=422, content={"detail": str(exc)})

            def events():
                yield f"data: {json.dumps({'type': 'meta', **engine.metadata()})}\n\n"
                for step in engine.stream_steps(request):
                    yield f"data: {json.dumps({'type': 'token', **step.as_dict()})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # this handler must NOT be a generator function, otherwise FastAPI
            # consumes it and the StreamingResponse return value is discarded
            return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/completions")
    def openai_completions(req: CompletionRequest):
        """OpenAI-compatible alias, so off-the-shelf clients can talk to ApexGPT.

        Only the non-streaming path is exposed here: OpenAI streams token objects
        while ApexGPT streams bare text, and faking that shape would break more
        clients than it would serve. Use ``/stream`` for incremental output.
        """
        if req.stream:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "stream=true is not supported here; "
                                             "use GET /stream instead", "type": "invalid_request_error"}},
            )
        request = GenerationRequest(
            prompt=req.prompt, max_new_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p, seed=req.seed,
        )
        try:
            request.validate()
        except ValueError as exc:
            return JSONResponse(status_code=422,
                                content={"error": {"message": str(exc), "type": "invalid_request_error"}})
        with _LOCK:
            t0 = time.time()
            text, details = engine.generate_with_details(request)
            elapsed = time.time() - t0
        prompt_tokens = len(engine.tokenizer(request.prompt)["input_ids"])
        completion_tokens = max(0, len(engine.tokenizer(text)["input_ids"]) - prompt_tokens)
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "text": text,
                # was hardcoded to None: the per-token logprobs are the model's
                # own uncertainty and the engine already has them
                "logprobs": details["logprobs"],
                "finish_reason": details["finish_reason"],
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "apexgpt": {"elapsed_s": round(elapsed, 4), **engine.metadata()},
        }

    @app.get("/v1/models")
    def openai_models():
        """Model discovery: OpenAI clients probe this before anything else."""
        meta = engine.metadata()
        created = int(os.path.getmtime(meta["checkpoint"])) if meta.get(
            "checkpoint") and os.path.exists(meta["checkpoint"]) else 0
        return {
            "object": "list",
            "data": [{
                "id": "apexgpt",
                "object": "model",
                "created": created,
                "owned_by": "apexgpt",
                "root": str(meta.get("checkpoint", "")),
                "parameters": meta.get("parameters"),
                "tokenizer": meta.get("tokenizer"),
                "context_length": meta.get("block_size"),
            }],
        }

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="apexgpt serve",
                                 description="Run the ApexGPT inference API")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (use 0.0.0.0 to expose on the LAN)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-stream", action="store_true",
                    help="disable the /stream endpoint")
    args = ap.parse_args(argv)

    try:
        engine = InferenceEngine(device=args.device).load(args.checkpoint)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    meta = engine.metadata()
    print("=" * 66)
    print(f"ApexGPT API - {meta['parameters_m']}M params on {meta['device']}")
    print(f"checkpoint: {meta['checkpoint']}")
    print(f"listening:  http://{args.host}:{args.port}")
    print("=" * 66)

    app = create_app(engine, streaming=not args.no_stream)
    try:
        import uvicorn
    except ImportError:
        print("[error] uvicorn is missing: pip install -r requirements-api.txt",
              file=sys.stderr)
        return 1

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
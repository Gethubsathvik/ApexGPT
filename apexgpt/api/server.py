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
import sys
import threading
import time

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

    return GenerateRequest, GenerateResponse, CompletionRequest


def create_app(engine: InferenceEngine, streaming: bool = True):
    """Build the FastAPI application around a loaded engine."""
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise SystemExit(
            "FastAPI is not installed.\n"
            "Install the API extra first:  pip install -r requirements-api.txt"
        ) from exc

    GenerateRequest, GenerateResponse, CompletionRequest = _pydantic_models()

    app = FastAPI(
        title="ApexGPT inference API",
        version="1.2.0",
        description="Next-token generation from a ApexGPT GPT checkpoint.",
    )

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
            text = engine.generate_text(request)
            elapsed = time.time() - t0
        return GenerateResponse(
            text=text,
            prompt_tokens=len(engine.tokenizer(request.prompt)["input_ids"]),
            elapsed_s=round(elapsed, 4),
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
                for piece in engine.stream(request):
                    yield f"data: {json.dumps({'type': 'token', 'token': piece})}\n\n"
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
            text = engine.generate_text(request)
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
                "logprobs": None,
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "apexgpt": {"elapsed_s": round(elapsed, 4), **engine.metadata()},
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
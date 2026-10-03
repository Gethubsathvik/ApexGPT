"""Hugging Face and Kaggle access: download models and datasets, run them.

Everything here is an optional extra. The training path in
``features/training/service.py`` needs none of it, so a machine without a Hub
token, without ``transformers`` or without Kaggle credentials still trains.

Two different things are called "the Hub" in this project and it is worth
keeping them apart:

* a *dataset* repo - text to train on. ``features/data/sources.py`` already
  streams those into one text file; :func:`download_dataset` instead fetches the
  repository's raw files, which is what you want when the parquet/json layout
  matters to you.
* a *model* repo - weights you can run. Nothing in ApexGPT trains these; they
  are read with ``transformers`` by :func:`run_model`, purely so a pretrained
  model can answer the same "what comes next" question as a ApexGPT one.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ...core.paths import HUB_DATASET_DIR, HUB_MODEL_DIR

MODEL_REPO = "model"
DATASET_REPO = "dataset"

# What "download a model" means in practice. Most Hub repos ship the same model
# four more times for ONNX, TensorFlow, Flax and Rust - the gpt2 repo alone is
# 3.5 GB of which ~500 MB is loadable PyTorch weights. These patterns keep the
# config, the tokenizer and the weights and skip the other exports; --allow
# overrides them.
DEFAULT_MODEL_PATTERNS = ("*.json", "*.txt", "*.safetensors", "*.bin")


class HubError(RuntimeError):
    """A Hub call failed; the message is written for a human."""


@dataclass(frozen=True)
class RepoFile:
    name: str
    size_bytes: int | None = None

    @property
    def size_mb(self) -> float | None:
        return None if self.size_bytes is None else self.size_bytes / 1e6


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):                # pragma: no cover
        return False


def kaggle_credentials() -> Path | None:
    """Where Kaggle's API token lives, if it is there at all."""
    candidates = [Path(os.environ["KAGGLE_CONFIG_DIR"])] if os.environ.get(
        "KAGGLE_CONFIG_DIR") else []
    home = Path.home()
    candidates += [home / ".kaggle" / "kaggle.json",
                   home / "kaggle.json",
                   home / ".config" / "kaggle" / "kaggle.json"]
    return next((c for c in candidates if c.exists()), None)


def hub_token() -> str | None:
    """A Hub token from the environment or the CLI login, without printing it."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    try:
        return token_path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def environment() -> dict:
    """What this machine can do with the Hub, without touching the network."""
    kaggle_client = ("kagglehub" if _installed("kagglehub")
                     else ("kaggle CLI" if shutil.which("kaggle") else None))
    credentials = kaggle_credentials()
    return {
        "huggingface_hub": _installed("huggingface_hub"),
        "transformers": _installed("transformers"),
        "datasets": _installed("datasets"),
        "kagglehub": _installed("kagglehub"),
        "kaggle_cli": bool(shutil.which("kaggle")),
        "kaggle_credentials": str(credentials) if credentials else None,
        "kaggle_client": kaggle_client,
        "hf_token": bool(hub_token()),
        "model_dir": str(HUB_MODEL_DIR),
        "dataset_dir": str(HUB_DATASET_DIR),
    }


def check() -> list[tuple[str, bool, str]]:
    """``(capability, available, detail)`` rows for the ``hub check`` report."""
    env = environment()
    creds = env["kaggle_credentials"]
    return [
        ("download Hugging Face datasets",
         bool(env["huggingface_hub"]),
         "public repos need no token; private ones need HF_TOKEN"),
        ("stream Hugging Face datasets into a corpus",
         bool(env["datasets"]),
         "python -m apexgpt data prepare --source hf:<repo_id>"),
        ("download and run a Hugging Face model",
         bool(env["transformers"] and env["huggingface_hub"]),
         "pip install -r requirements-hub.txt"),
        ("Hub authentication",
         bool(env["hf_token"]),
         "huggingface-cli login, or set HF_TOKEN"),
        ("download Kaggle datasets",
         bool(env["kaggle_client"] and creds),
         (f"client: {env['kaggle_client']}; credentials: {creds}"
          if env["kaggle_client"] and creds
          else "pip install kagglehub (or the kaggle CLI) and put kaggle.json "
               "in ~/.kaggle/ - Kaggle > Settings > API")),
    ]


def _require(module: str, extra: str = "requirements-hub.txt"):
    if not _installed(module):
        raise HubError(
            f"{module} is not installed. Run:  pip install -r {extra}")


# --------------------------------------------------------------------------- #
# repositories
# --------------------------------------------------------------------------- #
def list_files(repo_id: str, repo_type: str = MODEL_REPO,
               revision: str | None = None) -> list[RepoFile]:
    """Every file in a repository, with sizes when the Hub reports them."""
    _require("huggingface_hub")
    from huggingface_hub import HfApi

    try:
        info = HfApi().repo_info(repo_id, repo_type=repo_type, revision=revision,
                                 files_metadata=True)
    except Exception as exc:
        raise HubError(f"could not read {repo_type} repo {repo_id!r}: {exc}") from exc

    files = []
    for sibling in getattr(info, "siblings", None) or []:
        files.append(RepoFile(name=getattr(sibling, "rfilename", ""),
                              size_bytes=getattr(sibling, "size", None)))
    return sorted((f for f in files if f.name), key=lambda f: f.name)


def _snapshot(repo_id: str, repo_type: str, revision: str | None,
              allow_patterns: list[str] | None, local_dir: Path,
              progress=print) -> Path:
    _require("huggingface_hub")
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    progress(f"[hub]     {repo_type} repo {repo_id}"
             f"{' @ ' + revision if revision else ''}")
    progress(f"[hub]     -> {local_dir}")
    try:
        return Path(snapshot_download(
            repo_id=repo_id, repo_type=repo_type, revision=revision,
            allow_patterns=allow_patterns or None, local_dir=str(local_dir)))
    except Exception as exc:
        hint = ""
        if "xet" in str(exc).lower():
            # the xet storage backend 404s for anonymous downloads on some
            # machines; the plain HTTP path is slower but works
            hint = ("\n           If this mentions xet, retry with "
                    "HF_HUB_DISABLE_XET=1 set, or log in with HF_TOKEN.")
        raise HubError(f"download failed for {repo_id!r}: {exc}{hint}") from exc


def download_model(repo_id: str, revision: str | None = None,
                   allow_patterns: list[str] | None = None,
                   dest: Path | str | None = None,
                   progress=print) -> Path:
    """Fetch a model repository into the data root and return the directory.

    Only the loadable weights and their config/tokenizer are fetched unless
    ``allow_patterns`` says otherwise - see ``DEFAULT_MODEL_PATTERNS``.
    """
    target = Path(dest) if dest else HUB_MODEL_DIR / repo_id.replace("/", "--")
    patterns = allow_patterns or list(DEFAULT_MODEL_PATTERNS)
    progress(f"[hub]     patterns: {' '.join(patterns)}")
    return _snapshot(repo_id, MODEL_REPO, revision, patterns, target, progress)


def download_dataset(repo_id: str, revision: str | None = None,
                     allow_patterns: list[str] | None = None,
                     dest: Path | str | None = None,
                     progress=print) -> Path:
    """Fetch a dataset repository's raw files (parquet, jsonl, csv, ...)."""
    target = Path(dest) if dest else HUB_DATASET_DIR / repo_id.replace("/", "--")
    return _snapshot(repo_id, DATASET_REPO, revision, allow_patterns, target,
                     progress)


def download_kaggle_dataset(slug: str, dest: Path | str | None = None,
                            progress=print) -> Path:
    """Fetch a Kaggle dataset and return the directory it landed in.

    Credential discovery stays with ``kagglehub`` / the ``kaggle`` CLI, exactly
    as ``features/data/sources.py`` does it, so there is one set of rules.
    """
    from ..data.sources import download_kaggle

    target = Path(dest) if dest else HUB_DATASET_DIR / ("kaggle--" +
                                                        slug.replace("/", "--"))
    target.mkdir(parents=True, exist_ok=True)
    path = download_kaggle(slug, target, progress=progress)
    progress(f"[kaggle]  text file: {path}")
    return path.parent


# --------------------------------------------------------------------------- #
# running a pretrained model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HubPrediction:
    """One candidate next token from a pretrained Hub model."""

    rank: int
    token_id: int
    text: str
    probability: float


def run_model(repo_id: str, prompt: str, top_k: int = 10,
              max_new_tokens: int = 0, temperature: float = 0.8,
              device: str = "auto", revision: str | None = None,
              allow_patterns: list[str] | None = None,
              progress=print) -> tuple[list[HubPrediction], str]:
    """Download a Hub model and ask it what comes next after ``prompt``.

    Returns ``(predictions, generated_text)``. ``max_new_tokens=0`` skips
    generation entirely, which keeps the call to one forward pass.

    ``transformers`` is imported lazily: a machine that only trains ApexGPT
    never pays for it.
    """
    _require("transformers")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ...core.device import get_device

    local = download_model(repo_id, revision=revision,
                           allow_patterns=allow_patterns, progress=progress)
    target = get_device(device)
    progress(f"[hub]     loading {repo_id} on {target}")

    tokenizer = AutoTokenizer.from_pretrained(str(local))
    # transformers renamed torch_dtype -> dtype; support both rather than
    # pinning one version of a library this project does not otherwise need
    try:
        model = AutoModelForCausalLM.from_pretrained(str(local), dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(local),
                                                     torch_dtype=torch.float32)
    model.to(target)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    # BatchEncoding has .to(); a plain dict from a stub tokenizer does not
    encoded = (encoded.to(target) if hasattr(encoded, "to")
               else {k: v.to(target) for k, v in encoded.items()})
    with torch.no_grad():
        logits = model(**encoded).logits[:, -1, :]

    probs = torch.softmax(logits.float(), dim=-1)[0]
    k = min(int(top_k), probs.numel())
    top_probs, top_ids = torch.topk(probs, k)
    predictions = [
        HubPrediction(rank=rank, token_id=int(token_id),
                      text=tokenizer.decode([int(token_id)]),
                      probability=float(prob))
        for rank, (token_id, prob) in enumerate(zip(top_ids.tolist(),
                                                    top_probs.tolist()), start=1)
    ]

    text = ""
    if max_new_tokens > 0:
        do_sample = temperature > 0
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                    do_sample=do_sample,
                                    temperature=temperature if do_sample else None,
                                    top_p=0.95 if do_sample else None,
                                    pad_token_id=tokenizer.eos_token_id)
        prompt_len = encoded["input_ids"].shape[1]
        text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return predictions, text
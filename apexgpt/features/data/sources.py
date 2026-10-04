"""Where the text comes from: tiny Shakespeare, Hugging Face, Kaggle, or a file.

Every source ends the same way - a plain UTF-8 text file - so the tokenizer,
the ``uint16`` binaries and the training loop are identical no matter where the
corpus came from. Only the fetch step differs::

    python -m apexgpt data sources                        # what is available
    python -m apexgpt data prepare --source shakespeare  # 1.1 MB, ~300k tokens
    python -m apexgpt data prepare --source wikitext --target-mb 50
    python -m apexgpt data prepare --source hf:roneneldan/TinyStories
    python -m apexgpt data prepare --source kaggle:user/dataset-slug
    python -m apexgpt data prepare --source local:my_text.txt
    python -m apexgpt data tokens --tokenizer char --top 20

Hugging Face datasets are streamed and cut off at ``--target-mb`` rather than
downloaded whole, so a 12 GB corpus is usable on a laptop with 8 GB of RAM.
"""
from __future__ import annotations

import csv
import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

USER_AGENT = "ApexGPT/1.1 (+https://github.com/Gethubsathvik/ApexGPT)"

#: The canonical 1.1 MB corpus used by nanoGPT and char-rnn.
SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")

#: Field names to try when a source has no explicit text column.
TEXT_FIELDS = ("text", "content", "body", "article", "prompt", "completion")


@dataclass(frozen=True)
class Corpus:
    """One selectable corpus."""
    key: str
    title: str
    kind: str                     # wikipedia | text-url | hf-stream | kaggle | local
    locator: str                  # repo id, URL, slug or path
    size_mb: float | None = None
    config: str | None = None
    split: str = "train"
    text_column: str | None = None
    license: str = "see source"
    note: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, Corpus] = {
    "shakespeare": Corpus(
        key="shakespeare",
        title="Tiny Shakespeare (complete works)",
        kind="text-url",
        locator=SHAKESPEARE_URL,
        size_mb=1.1,
        license="public domain (Project Gutenberg)",
        note="~300k GPT-2 tokens: trains to recognisable English verse in "
             "minutes on a CPU. The fastest end-to-end demo in this repo.",
        aliases=("tiny_shakespeare", "tinyshakespeare", "shake"),
    ),
    "wikipedia": Corpus(
        key="wikipedia",
        title="Wikipedia 20231101.en",
        kind="wikipedia",
        locator="wikimedia/wikipedia",
        size_mb=772.0,
        config="20231101.en",
        text_column="text",
        license="CC BY-SA 4.0",
        note="The default corpus. Parquet shards are downloaded and streamed "
             "to disk, so peak RAM stays flat.",
    ),
    "wikitext": Corpus(
        key="wikitext",
        title="WikiText-2 (raw)",
        kind="hf-stream",
        locator="Salesforce/wikitext",
        size_mb=12.0,
        config="wikitext-2-raw-v1",
        license="CC BY-SA 3.0",
        note="Wikipedia's own benchmark set: small enough for a laptop, "
             "real prose instead of wiki markup.",
    ),
    "tinystories": Corpus(
        key="tinystories",
        title="TinyStories",
        kind="hf-stream",
        locator="roneneldan/TinyStories",
        size_mb=2000.0,
        license="MIT",
        note="Synthetic short stories written for small models. A 30M-parameter "
             "model trained on this produces grammatical English far sooner "
             "than it does on Wikipedia.",
    ),
    "openwebtext": Corpus(
        key="openwebtext",
        title="OpenWebText",
        kind="hf-stream",
        locator="Skylion007/openwebtext",
        size_mb=12000.0,
        license="MIT",
        note="8M documents. Streamed and truncated to --target-mb; never "
             "downloaded whole.",
    ),
    "kaggle": Corpus(
        key="kaggle",
        title="A Kaggle dataset",
        kind="kaggle",
        locator="owner/dataset-slug",
        note="Use --source kaggle:<slug>. Needs kaggle.json (Kaggle > Settings > "
             "API) and either kagglehub or the kaggle CLI.",
        aliases=("kaggle:",),
    ),
    "local": Corpus(
        key="local",
        title="A local text file",
        kind="local",
        locator="path/to/file.txt",
        note="Use --source local:<path>. .txt, .csv, .json, .jsonl and .parquet "
             "are converted to a text corpus.",
    ),
}

DEFAULT_SOURCE = "wikipedia"
SHAKESPEARE = REGISTRY["shakespeare"]


# --------------------------------------------------------------------------- #
# resolving a --source value
# --------------------------------------------------------------------------- #
def resolve(source: str | None = None) -> Corpus:
    """Turn a ``--source`` value into a :class:`Corpus`.

    Accepts a registry key, an alias, ``hf:<repo_id>``, ``kaggle:<slug>``,
    ``url:<https url>`` or ``local:<path>``.
    """
    spec = (source or DEFAULT_SOURCE).strip()
    lowered = spec.lower()

    for corpus in REGISTRY.values():
        if lowered == corpus.key or lowered in corpus.aliases:
            return corpus

    if lowered.startswith(("hf:", "huggingface:")):
        repo = spec.split(":", 1)[1]
        return Corpus(key=repo.replace("/", "_"), title=repo, kind="hf-stream",
                      locator=repo, text_column=None,
                      note="streamed from the Hugging Face Hub")
    if lowered.startswith("kaggle:"):
        slug = spec.split(":", 1)[1]
        return Corpus(key="kaggle", title=slug, kind="kaggle", locator=slug,
                      note="downloaded with kagglehub or the kaggle CLI")
    if lowered.startswith("local:"):
        path = spec.split(":", 1)[1]
        return Corpus(key="local", title=Path(path).name, kind="local",
                      locator=path, note="local file")
    if lowered.startswith(("url:", "http://", "https://")):
        url = spec.split(":", 1)[-1] if lowered.startswith("url:") else spec
        return Corpus(key="url", title=Path(url).name, kind="text-url",
                      locator=url, note="plain text over HTTP")

    known = ", ".join(sorted(REGISTRY))
    raise ValueError(f"unknown data source {source!r}. Known: {known} "
                     f"(or hf:<repo>, kaggle:<slug>, local:<path>, url:<link>)")


def token_estimate(corpus: Corpus, chars_per_token: float = 4.0) -> int | None:
    """Rough GPT-2 token count for a corpus of known size."""
    if not corpus.size_mb:
        return None
    return int(corpus.size_mb * 1e6 / chars_per_token)


def describe_sources() -> list[str]:
    """Human-readable listing of every registered corpus."""
    from ...core.config import DataConfig

    lines = ["available corpora (--source):", ""]
    width = max(len(c.key) for c in REGISTRY.values())
    for corpus in REGISTRY.values():
        estimate = token_estimate(corpus)
        size = f"{corpus.size_mb:.1f} MB" if corpus.size_mb else "size unknown"
        tokens = f"~{estimate / 1e6:.1f}M tokens" if estimate else "tokens unknown"
        lines.append(f"  {corpus.key:<{width}}  {corpus.title}")
        lines.append(f"  {'':<{width}}  {size}, {tokens} | {corpus.license}")
        if corpus.note:
            lines.append(f"  {'':<{width}}  {corpus.note}")
        lines.append("")
    lines.append("built token binaries:")
    for key in sorted(REGISTRY):
        cfg = DataConfig()
        cfg.select_dataset(key)
        state = "ready" if cfg.is_ready() else "not built"
        lines.append(f"  {key:<{width}}  {cfg.binary_dir} ({state})")
    return lines


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def download_file(url: str, dest: Path, progress=print, timeout: int = 120) -> Path:
    """Fetch one URL to ``dest`` with a browser-ish user agent."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    progress(f"[fetch]   {url}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, \
                dest.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"download failed ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"download failed: {exc.reason} ({url})") from exc
    size = dest.stat().st_size
    if size == 0:
        raise RuntimeError(f"downloaded an empty file: {url}")
    progress(f"[fetch]   {size / 1e6:.2f} MB -> {dest}")
    return dest


def _pick_text_file(directory: Path) -> Path:
    """Choose the most plausible text file from a downloaded directory."""
    candidates = [p for p in directory.rglob("*") if p.is_file()]
    if not candidates:
        raise RuntimeError(f"no files were downloaded into {directory}")

    def rank(path: Path) -> tuple[int, int]:
        suffix = path.suffix.lower()
        preference = {".txt": 0, ".csv": 1, ".json": 2, ".jsonl": 3,
                      ".parquet": 4, ".md": 5}.get(suffix, 9)
        return (preference, -path.stat().st_size)

    return sorted(candidates, key=rank)[0]


def download_kaggle(slug: str, work_dir: Path, progress=print) -> Path:
    """Download a Kaggle dataset, preferring the Python API over the CLI."""
    target = work_dir / "kaggle"
    target.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        kagglehub = None

    if kagglehub is not None:
        progress(f"[kaggle]  downloading {slug} via kagglehub")
        try:
            kagglehub.dataset_download(slug, output_dir=str(target))
            return _pick_text_file(target)
        except Exception as exc:
            progress(f"[kaggle]  kagglehub failed ({exc}); trying the kaggle CLI")

    try:
        code = shutil.which("kaggle")
    except Exception:                                  # pragma: no cover
        code = None
    if code:
        progress(f"[kaggle]  downloading {slug} via the kaggle CLI")
        done = subprocess_call([code, "datasets", "download", "-d", slug,
                                "-p", str(target), "--unzip"])
        if done == 0:
            return _pick_text_file(target)
        progress(f"[kaggle]  the kaggle CLI exited with {done}")

    raise RuntimeError(
        "Kaggle needs credentials and a client. Either:\n"
        "  pip install kagglehub        (then re-run)\n"
        "  pip install kaggle           (CLI, uses ~/.kaggle/kaggle.json)\n"
        "and put kaggle.json from Kaggle > Settings > API at "
        "~/.kaggle/kaggle.json (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)")


def subprocess_call(cmd: list[str]) -> int:
    import subprocess
    try:
        return subprocess.call(cmd)
    except (OSError, subprocess.SubprocessError) as exc:      # pragma: no cover
        print(f"[kaggle]  {exc}")
        return 1


def stream_hf_dataset(corpus: Corpus, dest: Path, target_mb: int = 1000,
                      progress=print) -> Path:
    """Stream a Hugging Face dataset to text, stopping at ``--target-mb``."""
    from tqdm import tqdm

    try:
        from datasets import load_dataset
    except ImportError as exc:
        # A missing or blocked pyarrow surfaces here as an ImportError, and a
        # traceback from deep inside the reader says nothing actionable.
        raise RuntimeError(
            "streaming a Hugging Face corpus needs the datasets package and a "
            f"working pyarrow ({exc}). Install them with\n"
            "  pip install datasets pyarrow\n"
            "or pass --source local:<path> instead."
        ) from exc

    kwargs = {"path": corpus.locator, "split": corpus.split, "streaming": True}
    if corpus.config:
        kwargs["name"] = corpus.config
    progress(f"[hf]      streaming {corpus.locator}"
             f"{'/' + corpus.config if corpus.config else ''} ({corpus.split})")

    try:
        dataset = load_dataset(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"could not stream {corpus.locator!r}: {exc}. Check the dataset id "
            f"and that it is public, or pass --source local:<path> instead."
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    budget = int(target_mb * 1e6)
    column = corpus.text_column
    written = 0
    with dest.open("w", encoding="utf-8") as out, \
            tqdm(total=budget, desc="[hf] chars", unit="B", unit_scale=True,
                 unit_divisor=1_000_000) as bar:
        for row in dataset:
            if column is None:
                column = next((f for f in TEXT_FIELDS if f in row), None)
                if column is None:
                    raise RuntimeError(
                        f"{corpus.locator!r} has no text column; columns are "
                        f"{sorted(row)}. Pass a file with --source local:<path>.")
            text = row.get(column)
            if not text:
                continue
            text = str(text).replace("\r\n", "\n").strip()
            if not text:
                continue
            out.write(text)
            out.write("\n\n")
            written += len(text)
            bar.update(len(text))
            if written >= budget:
                break
    if written == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{corpus.locator!r} produced no text")
    progress(f"[hf]      {written / 1e6:.1f} M characters -> {dest}")
    return dest


def convert_to_text(path: Path, dest: Path, progress=print) -> Path:
    """Convert a downloaded file into one plain-text corpus."""
    suffix = path.suffix.lower()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if suffix in ("", ".txt", ".md"):
        shutil.copyfile(path, dest)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh, \
                dest.open("w", encoding="utf-8") as out:
            for row in csv.reader(fh):
                out.write(" ".join(str(v) for v in row if v))
                out.write("\n")
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh, \
                dest.open("w", encoding="utf-8") as out:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.write(_row_to_text(json.loads(line)))
                out.write("\n")
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        with dest.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(_row_to_text(row))
                out.write("\n")
    elif suffix == ".parquet":
        import pyarrow.parquet as pq
        reader = pq.ParquetFile(path)
        with dest.open("w", encoding="utf-8") as out:
            for batch in reader.iter_batches(batch_size=2000):
                for row in batch.to_pylist():
                    out.write(_row_to_text(row))
                    out.write("\n")
    else:
        shutil.copyfile(path, dest)

    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name} converted to an empty corpus")
    progress(f"[convert] {path.name} -> {dest} "
             f"({dest.stat().st_size / 1e6:.2f} MB)")
    return dest


def _row_to_text(row) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return str(row)
    for field_name in TEXT_FIELDS:
        if row.get(field_name):
            return str(row[field_name])
    return " ".join(str(v) for v in row.values() if isinstance(v, (str, int, float)))


# --------------------------------------------------------------------------- #
# the entry point the service uses
# --------------------------------------------------------------------------- #
def fetch_corpus(corpus: Corpus, corpus_path: Path, raw_dir: Path,
                 target_mb: int = 1000, progress=print) -> Path:
    """Make sure ``corpus_path`` exists, fetching it from ``corpus`` if needed.

    ``wikipedia`` is handled by the parquet path in the data service; every
    other source lands here.
    """
    if corpus_path.exists() and corpus_path.stat().st_size > 0:
        progress(f"[skip]    reusing {corpus_path} "
                 f"({corpus_path.stat().st_size / 1e6:.0f} MB chars)")
        return corpus_path

    if corpus.kind == "local":
        source = Path(corpus.locator).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"local corpus not found: {source}")
        progress(f"[local]   {source}")
        return convert_to_text(source, corpus_path, progress)

    if corpus.kind == "text-url":
        return download_file(corpus.locator, corpus_path, progress)

    if corpus.kind == "hf-stream":
        return stream_hf_dataset(corpus, corpus_path, target_mb=target_mb,
                                 progress=progress)

    if corpus.kind == "kaggle":
        downloaded = download_kaggle(corpus.locator, raw_dir / "kaggle", progress)
        return convert_to_text(downloaded, corpus_path, progress)

    raise ValueError(f"source kind {corpus.kind!r} is handled by the data service")


def legacy_corpus(raw_dir: Path, dataset: str) -> Path | None:
    """The pre-multi-corpus cache filename, so an old build is not refetched."""
    if dataset != "wikipedia":
        return None
    old = raw_dir / "wiki_corpus.txt"
    return old if old.exists() else None

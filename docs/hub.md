# 🌐 Hugging Face & Kaggle

Pretrained models, hub datasets, and the reference training scripts.


# 🌐 Hugging Face & Kaggle

```bash
python -m apexgpt hub check                              # what this machine can do
python -m apexgpt hub files gpt2                         # what is in a repo
python -m apexgpt hub dataset Salesforce/wikitext        # download dataset files
python -m apexgpt hub model gpt2 --predict "Once upon a time"
python -m apexgpt hub kaggle inria/tiny-imagenet         # download a Kaggle dataset
```

## 🎯 Run a pretrained model and ask it the same question

`hub model <repo_id> --predict "<prompt>"` downloads the repo, loads it with
`transformers`, and prints the next-token distribution — the identical output
format `--predict` gives for a model trained here, so the two are directly
comparable:

```bash
python -m apexgpt hub model gpt2 \
    --predict "Once upon a time" --top-k 8 --max-new-tokens 200 --device auto
```

Two details worth knowing. First, `hub model` fetches **only** the config, the
tokenizer and the weights (`*.json`, `*.txt`, `*.safetensors`, `*.bin`); the
`gpt2` repo also ships the same model as ONNX, TensorFlow, Flax and Rust, which
is 3.5 GB of files PyTorch never reads. `--allow` overrides the list. Second,
`transformers` is imported lazily inside the call, so a machine that only trains
ApexGPT never loads it.

> If a download fails with a **404 on `xet-read-token`**, the xet storage
> backend cannot authenticate anonymously on that network. Retry with
> `HF_HUB_DISABLE_XET=1` set, or log in with `HF_TOKEN`. Both are surfaced by
> `hub check`.

## 📥 Datasets

```bash
python -m apexgpt data prepare --source hf:roneneldan/TinyStories   # streamed, cut at --target-mb
python -m apexgpt data prepare --source kaggle:user/dataset-slug    # Kaggle credentials required
python -m apexgpt hub dataset Salesforce/wikitext --allow '*.parquet'
python -m apexgpt hub kaggle user/dataset-slug
```

`hub dataset` and `hub kaggle` put the raw files on disk under `data/hub/`
without converting them, for when the parquet or JSONL layout matters. `data
prepare` is the other path: it ends with one plain UTF-8 text file, which is what
the tokenizer and the `uint16` binaries are built from. Both land in the same
`data/raw/<key>/` convention afterwards.

**Kaggle credentials are not a pip package.** Create an API token at
*Kaggle → Settings → API* and save it as `~/.kaggle/kaggle.json`
(`%USERPROFILE%\.kaggle\kaggle.json` on Windows), then:

```bash
python -m apexgpt setup --hub --install    # kagglehub, or use the kaggle CLI
python -m apexgpt hub check                # confirms client + credentials
```

`hub check` prints the whole picture in one screen — Hub client, `transformers`,
`datasets`, Kaggle client, Kaggle credentials, token present, and the two
directories downloads land in:

```text
  models   : .../data/hub/models
  datasets : .../data/hub/datasets

  [ok] download Hugging Face datasets             public repos need no token; private ones need HF_TOKEN
  [ok] stream Hugging Face datasets into a corpus python -m apexgpt data prepare --source hf:<repo_id>
  [ok] download and run a Hugging Face model      pip install -r requirements-hub.txt
  [--] Hub authentication                         huggingface-cli login, or set HF_TOKEN
  [--] download Kaggle datasets                   pip install kagglehub (or the kaggle CLI) and put kaggle.json in ~/.kaggle/
```

## 🧑‍🏫 Reference training scripts

ApexGPT trains its own GPT rather than loading one, so the Hub's training scripts
are the equivalent of `features/training/service.py` for pretrained models —
read them, do not depend on them:

| Upstream | Script | Equivalent here |
|----------|--------|-----------------|
| `huggingface/transformers` | `examples/pytorch/language-modeling/run_clm.py` | `features/training/service.py` — corpus → batches → AdamW → checkpoints |
| `huggingface/trl` | `SFTTrainer` | supervised fine-tuning of a pretrained checkpoint; the same loop with a different data source |
| `huggingface/tokenizers` | `Tokenizer` training | `features/data/tokenizers.py` for the byte-level vocabulary |

Fine-tuning a released checkpoint is `--resume`, which restores the model, the
optimizer and the step count rather than starting over:

```bash
python -m apexgpt train --dataset shakespeare --resume <checkpoint.pt> --max-iters 800
```

---


---

Back to [the README](../README.md).


# 🐛 Bugs found and fixed

Defects this project found in itself, and what each one taught.


# 🐛 Bugs found and fixed

Notable defects caught during the audit and regression-tested:

| Bug | Impact |
|-----|--------|
| KV cache used `is_causal=False` on cached multi-token steps | queries attended to **future tokens** |
| `set_gradient_checkpointing()` folded the flag with `.training` | silently no-op'd unless called after `.train()` |
| `--epochs` accepted and ignored | users got a different run than they asked for |
| Resume discarded history and best-val | loss curves silently restarted |
| `np.memmap` reopened per batch | re-read a 461 MB file every step |
| `np.unique` on 242M tokens | sorted the whole corpus for a log line |
| `transformers` 5.18 `save_pretrained` wrote an empty tokenizer | silently produced a **zero-token corpus** |
| `--force` deleted the 1.2 GB corpus | 20-minute rebuild, no warning |
| `_split_binary` returned total tokens instead of train count | validation split came out empty |
| `/stream` handler contained a `yield` | FastAPI consumed the generator and returned an **empty body** |
| Pydantic models defined inside a factory | unresolved forward refs, schema generation crashed |
| Test-artifact checkpoints shadowed real runs | `generate` silently loaded the wrong model |
| GUI read a Tk variable from a worker thread | `RuntimeError: main thread is not in main loop` |
| `python -m apexgpt data prepare` — documented in the README but not implemented | every corpus command in the docs errored out |
| ROCm index built as `cu124rocm`, XPU index missing the `download.` host | both wheel installs 404'd |
| Assigning `DataConfig.dataset` left `binary_dir` alone | trained on the **previous** corpus's tokens |
| `DataConfig` `raw_dir`/`binary_dir` could not be overridden per corpus | every corpus shared one directory |
| Char-corpus reports decoded with GPT-2 BPE | `UnicodeEncodeError: 'charmap' codec can't encode '\ufffd'` on Windows consoles |
| `ByteTokenizer.decode` masked ids into bytes | the eos id (256) decoded to a NUL character instead of nothing |
| Windows `System Idle Process` (pid 0) in the process table | topped the "busiest processes" list forever, at a nonsense 200% CPU |
| `hub model --allow … --predict …` | the file patterns were dropped, so a 3.5 GB repo downloaded in full |
| `tokenizer(...)` assumed a `BatchEncoding` | `AttributeError` on any tokenizer that returns a plain dict |
| The GUI's next-token panel read a Tk variable from the worker thread | `RuntimeError: main thread is not in main loop` — the same trap as the earlier streaming bug |
| `PROJECT_ROOT` was `parent.parent.parent` unconditionally | an installed copy wrote checkpoints and corpora into `site-packages` |
| `/v1/completions` hardcoded `"logprobs": null` and `finish_reason: "length"` | clients could not score the model, and an eos stop was reported as a full-length one |
| `data tokens` sent the default corpus to `fetch_corpus` | `ValueError: source kind 'wikipedia' is handled by the data service`, so the one command that only reads text could not inspect the default corpus at all |
| A lab test called `lab.main(["--register-only"])` for real | it returns 1 when jupyterlab is absent, so all four CI jobs failed on a machine-dependent test while the local suite passed |
| The optimiser section documented $\beta_2 = 0.999$ | the code uses `beta2 = 0.95`; the README described Adam's textbook default, not this project's |
| The GUI text bar skipped a refresh when a count was still running | typing during a corpus count silently left the previous prompt's values on screen; now a stale result is dropped and the newer prompt is counted |
| The token table was printed by the CLI, so nothing else could use it | the counting lived in the view; it is now `build_inventory` in the data service, which the GUI reads |

---


---

Back to [the README](../README.md).


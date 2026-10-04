# AGENTS.md

Working notes for this repository. Nothing here is user-facing; the README is
the user-facing document.

## Commands

```bash
python -m pytest -q                 # the whole suite, ~4 min
python -m pytest tests/test_x.py -q # one file
python -m apexgpt <command> --help
```

`.venv\Scripts\apexgpt.exe` and `.venv\Scripts\pytest.exe` work now; `python -m
...` is still the documented form because it does not depend on the launchers
above.

`apexgpt` is a dispatcher: `env setup doctor data hub train generate gui lab serve`.
`data` itself has `prepare` (default), `sources` and `tokens`.

## Environment constraints on this machine

These cost real time to rediscover. They are properties of the machine, not bugs
in the code — do not "fix" them in the repository.

1. **Use `.venv` (Python 3.14). Do not create a new virtualenv.** An Application
   Control (WDAC) policy on this box blocks the DLLs of a freshly installed
   `torch`, so any venv created here dies with
   `OSError: [WinError 4551] An Application Control policy has blocked this file`
   on `torch\lib\shm.dll`. Probed and confirmed:

   | probe | result |
   |---|---|
   | a system DLL copied to `%TEMP%` and loaded by 3.13 | loads |
   | `shm.dll` from the existing `.venv`, loaded by 3.13 | loads |
   | `shm.dll` from a venv created today | **blocked** |

   So the rule is not the file, the signature or the interpreter — only new
   installs. Python 3.13 is installed (`py -3.13`) and fine for anything that
   does not import torch.

2. **Multi-version testing belongs to CI.** The workflow runs 3.11 and 3.13 on
   Linux and Windows. Do not try to reproduce a CI failure on a local 3.13 venv.

3. **Reading CI failures.** `gh` is not installed, the job-log endpoint answers
   403 without a token, and the unauthenticated API allows 60 requests an hour.
   `tests/conftest.py` therefore turns every failing test into a `::error`
   annotation, which the check-run output carries. Read it with:

   ```powershell
   $h = @{ "User-Agent" = "apexgpt"; "Accept" = "application/vnd.github+json" }
   (Invoke-RestMethod -Uri "https://api.github.com/repos/Gethubsathvik/ApexGPT/commits/<sha>/check-runs" -Headers $h).check_runs
   ```

   then fetch `output.annotations_url`. The junit report is also uploaded as a
   run artifact, but downloading that needs a signed-in session.

4. **The project folder has moved; the venv's console scripts have not.** `.venv`
   was created at `D:\my projects\TinyLLM`. Every distlib launcher pip writes
   (`pip.exe`, `pytest.exe`, `uvicorn.exe`, all of the `jupyter-*.exe`) embeds the
   absolute path of the interpreter it spawns, so all 29 of them still point
   there and fail with `Fatal error in the launcher: Unable to create process`.
   `python -m ...` keeps working, which is why this looks like a broken install
   rather than a moved one. Do **not** delete and recreate `.venv` — rule 1 makes
   that impossible. Repair it instead: reinstall pip once to fix its own
   launchers, then regenerate the rest from the installed entry points with
   `pip._internal.operations.install.wheel.PipScriptMaker`, pointing it at
   `sys.executable` and only rewriting exes that still embed the old path.
   Afterwards `pip install -e . --no-deps --no-build-isolation` puts
   `apexgpt.exe` back. Two leftovers of this: `jupyter.exe` itself is blocked by
   the policy even when rewritten (use `jupyter-lab.exe`, or
   `python -m jupyter`, which works), and a stale `tiny-agents.exe` survives the
   rename to ApexGPT because its dist-info is long gone.

5. **A blocked DLL can surface as an unrelated ImportError.** `import datasets`
   fails inside `pyarrow.dataset` with "An Application Control policy has blocked
   this file", which has nothing to do with datasets; the same policy later began
   blocking `pyarrow._parquet`, which takes six `tests/test_package.py` tests
   with it. The block is not a one-off at install time — it appeared hours later
   on a package that had been loading all day, so treat a green local suite from
   earlier in the day as no guarantee. `stream_hf_dataset` turns that import
   failure into an actionable `RuntimeError`, and the tests that need a working
   pyarrow skip rather than fail. CI installs its own and is unaffected.

6. **A release is a tag, in that order.** Push the commit to `main`, wait for the
   `tests` workflow to be green, bump `version` in `pyproject.toml`, then push
   the tag. `.github/workflows/release.yml` refuses a tag whose commit has no
   green `tests` run and refuses a tag that disagrees with the version, because
   both mistakes produce an artifact nobody can install. `gh` is not installed
   here, so releases are made by the workflow's own `GITHUB_TOKEN`; to undo one,
   delete the tag.

## House rules

- `tiny shakespeare.txt` is untracked on purpose. Leave it that way: it is not
  source, and `tests/test_package.py` asserts that nothing under `apexgpt/`,
  `tests/`, `notebooks/` or `.github/` is ignored or missing.
- `.gitignore` entries for generated artifacts are anchored to the repository
  root (`/models/`, `/data/`). An unanchored `models/` also matches
  `apexgpt/models/` — that mistake hid the model layer from every clone.
- No comments in new code unless asked; docstrings that say *why* are welcome.
- Tests use the byte-level `char` tokenizer when they must not touch the
  network, and skip rather than fail when a tokenizer download is impossible.
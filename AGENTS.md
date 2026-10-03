# AGENTS.md

Working notes for this repository. Nothing here is user-facing; the README is
the user-facing document.

## Commands

```bash
python -m pytest -q                 # the whole suite, ~4 min
python -m pytest tests/test_x.py -q # one file
python -m apexgpt <command> --help
```

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
# 🪟 Platform setup

Per-platform dependencies, and the installs that need more than `pip`.


# 🪟 Windows

<details open>
<summary><b>PowerShell</b></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m apexgpt setup --install
python -m apexgpt env
python -m apexgpt data prepare --source shakespeare
python -m apexgpt train --dataset shakespeare
python -m apexgpt gui
```

</details>

**If `Activate.ps1` is blocked** by the execution policy, use the interpreter directly — no activation needed:

```powershell
.\.venv\Scripts\python.exe -m apexgpt env
.\.venv\Scripts\python.exe -m apexgpt gui
```

Or unblock it once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Keep the corpus off the system drive.** By default data lands in `.\data`. If `C:` is tight, point it elsewhere:

```powershell
$env:APEXGPT_DATA_DIR = "C:\apexgpt_data"
```

<details>
<summary><b>GPU notes for Windows</b></summary>

`setup` detects `nvidia-smi` and installs the matching CUDA wheel automatically.
On a machine with only an AMD or Intel iGPU you will correctly get `cpu`,
because neither has a CUDA path in PyTorch. To try DirectML:

```powershell
python -m apexgpt setup --backend dml --install
python -m apexgpt env                      # dml available: True
python -m apexgpt train --device dml --preset smoke
```

</details>

---


# 🐧 Linux

```bash
sudo apt install python3-venv python3-pip     # Debian/Ubuntu
# Fedora: sudo dnf install python3 python3-pip

python3 -m venv .venv
source .venv/bin/activate

python -m apexgpt setup --install              # picks cu124 / rocm / cpu for you
python -m apexgpt env
python -m apexgpt data prepare --source shakespeare
python -m apexgpt train --dataset shakespeare
```

The GUI needs Tk, which is not always installed:

```bash
sudo apt install python3-tk      # Debian/Ubuntu
sudo dnf install python3-tkinter # Fedora
```

Run in a headless session? `train`, `generate` and `serve` all work without a display. Only `gui` needs one.

<details>
<summary><b>GPU notes for Linux</b></summary>

```bash
# NVIDIA
python -m apexgpt setup --backend cuda --cuda 124 --install

# AMD ROCm
python -m apexgpt setup --backend rocm --rocm 6.2 --install

# Intel
python -m apexgpt setup --backend xpu --install
```

For ROCm, PyTorch needs the matching `rocm-smi` libraries present; verify with
`rocm-smi` before trusting the detection. `python -m apexgpt env` then shows
`rocm build:` instead of a CUDA build.

</details>

---


# 🍎 macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m apexgpt setup --install
# Tk is bundled, but you may need:
xcode-select --install

python -m apexgpt env
python -m apexgpt data prepare --source shakespeare
python -m apexgpt train --dataset shakespeare
```

On **Apple Silicon** `setup` detects `mps` and installs the default wheel, which
includes Metal support. `doctor` will report `mps available: True`, and
`--device auto` selects it automatically — MPS is **not** CUDA, so check the
`mps` line rather than `cuda`. On Intel Macs it falls back to `cpu`.

> The Tk GUI on macOS is the least-tested surface here; the CLI and API are the
> dependable paths.

---


---

Back to [the README](../README.md).


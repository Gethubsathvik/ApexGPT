# 🤖 Android

Driving the inference API from a phone.


# 🤖 Android

Being straight about this: **PyTorch training on Android is not a realistic target**, and this project does not ship an Android app. What is genuinely possible:

**1. Termux — run the CLI on the phone** (slow but real)

```bash
pkg install python clang libjpeg-turbo
python -m apexgpt setup --install          # Termux has no GPU, so this picks cpu
python -m apexgpt env
python -m apexgpt train --dataset shakespeare --preset smoke
```

Expect CPU-only training on a phone to be roughly **20-50x slower** than a laptop — and phone SoCs are usually ARM with far less memory bandwidth. Practical for the CLI, the test suite and inference; not for real training runs. tiny Shakespeare is the corpus to use here: 1.1 MB instead of 1.3 GB.

**2. A client for the HTTP API** — the supported route

Run `serve` on your PC, bind it to the LAN, and call it from any phone browser or app:

```bash
python -m apexgpt serve --host 0.0.0.0 --port 8000
```

```bash
# from the phone's browser or any HTTP client
curl "http://192.168.1.50:8000/stream?prompt=hello&max_new_tokens=20"
```

Because `/v1/completions` speaks the **OpenAI completion shape**, existing
Android/Flutter clients work without modification — point them at your machine
and set the model name to anything:

```bash
curl -X POST http://192.168.1.50:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"apexgpt","prompt":"hello","max_tokens":20}'
```

> **Do not expose this to the internet as-is.** There is no authentication,
> rate limit or TLS. Bind it to `127.0.0.1` or a trusted LAN, and put a reverse
> proxy in front of it if it needs to leave your network.

**3. On-device inference** would need an ONNX/TFLite export plus a Kotlin or
Flutter client. That is a real piece of work with a genuine accuracy cost from
conversion, and it is **not implemented here**.

---


---

Back to [the README](../README.md).


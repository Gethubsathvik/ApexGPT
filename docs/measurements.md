# 📏 Measured

Runs of this repository, recorded: the tokenizer comparison and two completed trainings.


## Measured: GPT-2 BPE vs. byte-level, `cpu-tiny`, 400 steps

| | GPT-2 BPE (`gpt2`) | Byte-level (`char`) |
| --- | --- | --- |
| vocabulary | 50,257 | **257** |
| tokens for the corpus | 338,025 | **1,115,394** |
| characters per token | 3.30 | 1.00 |
| characters actually observed | — | 65 |
| parameters (`cpu-tiny`) | 30.0M | **10.8M** |
| speed on 8 CPU threads | ~230 tok/s | **~500 tok/s** |
| characters seen in 400 steps | 1.01M | 0.31M |
| validation loss | 5.6159 nats/token | 2.4614 nats/char |
| **validation loss per character** | **1.702 nats/char** | 2.4614 nats/char |
| train / validation split | 268,364 / 69,661 | 892,315 / 223,079 |

The two losses are only comparable once they are divided by characters, which is
why the table ends in nats/char. BPE stays the default: at equal steps each of
its sequences covers 3.3× more text, and that outweighs its size. `char` is
worth choosing when you want the smaller model and the faster iteration — 3×
fewer parameters, because the embedding drops from 19.3M to 0.10M, and ~2×
throughput — and its output is visibly rougher at the same step count:

```text
ROMEO:
Wit lllll he, wif me he thome
Fou ase dseis y thers omyongo illll dirold nd are he hey it te mere s
```

---


## Completed run — `cpu-tiny` on tiny Shakespeare, 400 steps, 29 min

```
[mem]      amp=False grad_checkpointing=False   (CPU: bf16 is emulated ...)
 step  train   val     lr
  100 6.1314 6.1929 2.72e-04
  200 5.3189 5.8664 1.77e-04
  300 5.2917 5.7019 7.44e-05
  400 5.4510 5.6159 3.00e-05
best val: 5.6159 at step 400
```

Validation loss fell from **10.82** (ln 50257 — random guessing) to **5.62** on a
270k-token corpus, in 29 minutes on a CPU with no GPU, at ~230 tok/s.


## Completed run — `cpu-tiny` on Wikipedia, 600 steps, 44 min

```
 step  train   val     lr
   25 9.3828 9.2700 2.50e-04
  125 7.6041 7.6740 2.82e-04
  225 7.3825 7.2816 2.30e-04
  325 7.3503 7.2753 1.58e-04
  425 7.3957 7.1531 8.87e-05
  525 6.9428 7.0375 4.17e-05
  600 7.0739 7.2499 3.00e-05
best val: 7.0375 at step 525
```


---

Back to [the README](../README.md).


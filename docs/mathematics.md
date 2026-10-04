# 📐 The mathematics behind ApexGPT

Every formula the code implements, and why each one is shaped the way it is.


# 📐 The mathematics behind ApexGPT

Every formula below is either **used by this code** — with the file that
implements it — or listed as **not used**, with the reason. Nothing is decorative.
Where the distinction matters, the honest answer is more useful than a formula
that looks impressive.

## 📇 Formula index

The complete list, with the notation each symbol carries. Sections further down
derive *why* each one is here; this table is for looking one up.

| # | What it computes | Formula | Notation | Implemented |
|---|---|---|---|---|
| 1 | Training objective | $\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log\,\mathrm{softmax}(z_t)_{x_t}$ | $T$ tokens per block, $z_t$ logits at position $t$, $x_t$ the token that came next, $\theta$ the weights | `models/gpt.py:229` |
| 2 | Output layer / sampling distribution | $p_i = \dfrac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}$ | $z\in\mathbb{R}^{V}$ logits, $V$ vocabulary (50257 or 257) | `models/sampling.py:103` |
| 3 | Multi-head causal attention | $\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | $d_k = d/H$ per-head width (`n_head` × `head_dim` in the code), $Q=XW_Q,\ K=XW_K,\ V=XW_V$ | `models/gpt.py:63` |
| 4 | Causality mask | $M_{ij}=0$ if $j\le i$, $-\infty$ otherwise | $i$ query position, $j$ key position; $-\infty$ kills the softmax term | `models/gpt.py:71` |
| 5 | Every parameter's entry point | $z = Wx+b$ | $W$ weight matrix, $b$ bias vector | `nn.Linear`, `nn.Embedding` |
| 6 | MLP nonlinearity | $\mathrm{GELU}(z)=z\,\Phi(z),\quad \Phi(z)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z}e^{-t^2/2}\,dt$ | $\Phi$ = standard normal CDF | `models/gpt.py:88` |
| 7 | Feature-wise standardisation | $\mathrm{LN}(z)=\gamma\odot\frac{z-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ | $\mu,\sigma^2$ over the $d$ features of **one** token, $\gamma,\beta$ per-feature scale/shift, $\odot$ elementwise | `nn.LayerNorm` |
| 8 | Residual path | $x \leftarrow x + F(\mathrm{LN}(x))$ | pre-norm: keeps the skip path an identity | `models/gpt.py:107` |
| 9 | Weight tying | $W_{lm} = E_{token}$ | one embedding matrix, used as input lookup and output layer | `models/gpt.py:127` |
| 10 | Inverted dropout | $\tilde{z}_i=\dfrac{z_i}{1-p}\cdot m_i,\quad m_i\sim\mathrm{Bernoulli}(1-p)$ | $p=0.1$ on embeddings, attention weights and MLP outputs; scaled by $1/(1-p)$ during training, identity at inference | `models/gpt.py:40` |
| 11 | Initialisation | $w\sim\mathcal{N}(0,0.02^2)$; residual projections $w\sim\mathcal{N}\!\left(0,\left(\tfrac{0.02}{\sqrt{2L}}\right)^2\right)$ | $L$ layers; scales residual branches by $1/\sqrt{2L}$ | `models/gpt.py:131` |
| 12 | Optimiser (AdamW) | $\theta\leftarrow\theta-\alpha\frac{\hat m_t/(1-\beta_1^t)}{\sqrt{\hat v_t/(1-\beta_2^t)}+\epsilon}$ | $\alpha$ learning rate, $\hat m_t,\hat v_t$ bias-corrected moments, $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$ | `features/training/service.py:65` |
| 13 | Decoupled weight decay | $\theta \leftarrow \theta - \lambda\theta$ | $\lambda=0.1$, matrices only ($p.\mathrm{dim}\ge2$) | `features/training/service.py:64` |
| 14 | Gradient clipping | $g\leftarrow g\cdot\min\!\left(1,\frac{\tau}{\lVert g\rVert_2}\right)$ | $\tau=1.0$ | `features/training/service.py:307` |
| 15 | Learning-rate schedule | $\eta_t=\eta_{max}\frac{t+1}{T_w}$, then $\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi p)$ | $T_w$ warmup steps $= \max(10, 0.05T)$, $p$ decay progress, $\eta_{max}=3\times10^{-4}$, $\eta_{min}=3\times10^{-5}$ | `features/training/service.py:48` |
| 16 | Sampling temperature | $p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$ | $T>1$ flattens, $T<1$ sharpens, $T\le 0$ greedy `argmax` | `models/sampling.py:98` |
| 17 | Top-$k$ filter | $z_i\leftarrow-\infty$ when $z_i<z_{(k)}$ | $z_{(k)}$ = $k$-th largest logit | `models/sampling.py:34` |
| 18 | Nucleus (top-$p$) filter | keep the smallest $m$ with $\sum_{i\le m}p_{(i)}\ge p$ | sorted descending; the most likely token is always kept | `models/sampling.py:52` |
| 19 | Repetition penalty | $z_i\leftarrow z_i/\lambda$ if $z_i>0$, else $\lambda z_i$ | for already-generated ids; $\lambda=1.0$ means off | `models/sampling.py:23` |
| 20 | Predictive entropy | $H(p)=-\sum_i p_i\log p_i\in[0,\ln V]$ | natural log, so **nats**; $\ln V$ = uniform guessing | `models/sampling.py:76` |
| 21 | Reported log-probability | $\log p_i$ on the **untruncated** softmax at the same $T$ | top-$p$ would renormalise a certain token to $\log 1 = 0$ | `models/sampling.py:154` |
| 22 | KV cache | $K_{t}=[\,K_{<t}\,;\,k_t\,]$, attend over the concatenation | $O(T)$ per generated token instead of $O(T^2)$ | `models/gpt.py:53` |
| 23 | Steps per epoch | $\left\lfloor \dfrac{N_{train}}{B\cdot T_{blk}}\right\rfloor$ | $B$ batch size, $T_{blk}$ block size | `features/training/service.py:72` |
| 24 | Perplexity | $\mathrm{PPL}=e^{\mathcal{L}}$, comparable as $\mathcal{L}$ per character | only comparable within one tokenizer | reported as `val loss` |
| 25 | Forward-pass cost | $\mathrm{FLOPs}\approx 6N+12LHd^2T$ | $N$ parameters, $L$ layers, $H$ heads, $d$ hidden, $T$ tokens | `models/builder.py:26` |
| 26 | Machine statistics | $\mu=\frac1n\sum x_i$, $\sigma^2=\frac1n\sum(x_i-\mu)^2$, $\rho=\frac{\mathrm{Cov}}{\sigma_X\sigma_Y}$ | scan inputs: CPU, RAM, cores, per-process cost | `core/system.py` |
| 27 | CPU utilisation | $100\left(1-\frac{\Delta\,\mathrm{idle}}{\Delta\,\mathrm{idle}+\Delta\,\mathrm{kernel}+\Delta\,\mathrm{user}}\right)$ | counters are cumulative since boot, so **differences** of two samples | `core/system.py:164` (psutil), `:204` (Win32 `GetSystemTimes`) |
Four conventions that the table alone would hide:

- **Padding is excluded, not predicted.** Targets of `-1` are dropped from the
  mean (`ignore_index=-1`), so a padded position contributes no loss term.
- **No label smoothing.** The target is exactly one-hot; smoothing
  ($\mathcal{L} = (1-\varepsilon)\mathcal{L}_{x_t} + \frac{\varepsilon}{V}\sum_i \mathcal{L}_i$)
  is deliberately not applied, because the point of this project is to report
  the model's real uncertainty.
- **Everything is in nats.** Loss, entropy and log-probabilities all use the
  natural log, so $\ln V$ is the no-information baseline. Divide by
  $\ln 2 = 0.693$ for bits.
- **Log-probability is measured before truncation.** A token drawn after top-$p$
  collapse still reports its probability under the untouched softmax at the same
  temperature, which is the model's own uncertainty rather than an artefact of
  the filter.

The classification formulas that are deliberately *absent* — accuracy, precision,
recall, $F_1$, confusion matrix — are listed with their reasons in
[Deliberately not implemented](#deliberately-not-implemented).

**Notation key.** One row per symbol, so no formula above needs a detour to
read:

| Symbol | Means | Symbol | Means |
|---|---|---|---|
| $T$ | tokens in one training block, or in the sampled window | $\theta$ | the full parameter vector of the network |
| $t$ | index of a token position, $1 \dots T$ | $\epsilon$ | numerical floor ($10^{-8}$ in AdamW, $10^{-5}$ in LayerNorm) |
| $x_t$ | the **target** token at position $t$ (the one that came next) | $\alpha$, $\eta_{max}$ | learning rate, its peak value |
| $z$, $z_t$ | logits, before softmax | $\eta_{min}$ | learning-rate floor after decay |
| $p_i$ | probability of token id $i$ | $\eta_t$, $T_w$ | learning rate at step $t$, warmup length |
| $V$ | vocabulary size: 50257 (GPT-2 BPE) or 257 (byte-level) | $\tau$ | gradient-clipping threshold (1.0) |
| $d$ | hidden width (`n_embd`) | $\lambda$ | weight decay (0.1) or repetition penalty ($\ge 1$) |
| $L$ | number of transformer blocks | $g_t$, $\hat m_t$, $\hat v_t$ | gradient, its first and second moment |
| $H$ | number of attention heads (`n_head` in the code) | $\beta_1$, $\beta_2$ | moment decay rates (0.9, 0.95) |
| $d_k$ | per-head width, $d/H$ | $p$ | nucleus mass (top-$p$) or schedule progress |
| $Q,K,V$ | query, key, value matrices | $k$ | number of tokens kept by top-$k$ |
| $W, b$ | weight matrix, bias vector | $B$, $T_{blk}$ | micro-batch size, block (context) length |
| $\gamma$, $\beta$ | LayerNorm scale and shift (not Adam's $\beta$) | $N$, $N_{train}$ | parameter count, training tokens |
| $\odot$ | elementwise (Hadamard) product | $\Phi$ | standard normal CDF |
| $\mu$, $\sigma^2$, $\sigma$ | mean, variance, standard deviation | $\rho$ | Pearson correlation |
| $m, v$ | Adam's uncorrected moments | $\lVert g\rVert_2$ | Euclidean norm of the gradient |
| $F$ | attention or MLP sublayer | $\ln$, $e$ | natural logarithm, $e^x$ |

## The one loss function this project trains

Language modelling is next-token prediction. For a token sequence
$x_1, \dots, x_T$ the model predicts each token from the ones before it, and the
training objective is the **cross-entropy** between its predicted distribution and
the one-hot distribution of the token that actually came next:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}),
\qquad
p_\theta(x_t \mid x_{<t}) = \mathrm{softmax}(z_t)_i \;\text{ at } i = x_t
$$

Cross-entropy is the KL divergence between the model's distribution and the
target distribution, with the target's own entropy $H(y)$ dropped because it is a
constant that cannot be optimised:

$$
D_{\mathrm{KL}}(P\|Q) = \sum_i P(x_i)\log\frac{P(x_i)}{Q(x_i)},
\qquad
\underbrace{H(P,Q)}_{\text{cross-entropy}} = \underbrace{D_{\mathrm{KL}}(P\|Q)}_{\text{optimised}} + \underbrace{H(P)}_{\text{constant}}
$$

This is the entire objective. Everything else — depth, attention, sampling — is a
way of estimating or using $p_\theta$. It is implemented in
`models/gpt.py` (the `targets` branch of `GPT.forward`) and reported as
`val loss`.

**How to read the numbers in this README.** The loss is in **nats**, i.e. the
average of $-\log p$. For a fresh model that is exactly $\ln V$:

| Vocabulary | Fresh model | Perfect model |
|---|---|---|
| GPT-2 BPE, $V = 50257$ | $\ln 50257 = 10.82$ nats/token | 0 |
| byte-level, $V = 257$ | $\ln 257 = 5.55$ nats/char | 0 |

So a byte-level model can reach a lower *number* than a BPE model and still be
worse at text. The comparable quantity is nats **per character**
(`nats/token ÷ chars-per-token`), which is why the tokenizer comparison in
[🔤 Tokenizers](../README.md#%F0%9F%94%A4-tokenizers) is decided on 1.702 vs 2.461 nats/char rather than
on 5.6159 vs 2.4614.

## Attention

Scaled dot-product attention, one head:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\qquad Q = XW_Q,\; K = XW_K,\; V = XW_V
$$

The $1/\sqrt{d_k}$ factor is what keeps the softmax out of its saturating region:
if the entries of $q\cdot k$ have variance $\propto d_k$, the logits' variance
grows with $d_k$ too, `softmax` becomes a hard argmax, and the gradient vanishes.
Dividing by $\sqrt{d_k}$ restores unit variance. Implemented in
`models/gpt.py::CausalSelfAttention` via
`torch.nn.functional.scaled_dot_product_attention` with a causal mask, so the
mask, the scaling and the softmax are the fused kernel's, not a re-implementation.

## The neuron and its activations

Every parameter in the network enters through an affine map, and the network is a
stack of these with a nonlinearity between them — without the nonlinearity,
stacking $L$ layers is algebraically the same as one layer:

$$
z = Wx + b,
\qquad
\text{ReLU}(z) = \max(0, z),
\qquad
\text{GELU}(z) = z\,\Phi(z),
\qquad
\Phi(z) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z} e^{-t^2/2}\,dt
$$

ApexGPT's MLP is `GELU → Linear(4d) → Linear(d)` (`models/gpt.py::MLP`), and
`GELU` is used because it is smooth, which matters when the network is deep. The
positional and token embeddings are the exception: those are pure affine maps,
with no activation in between, because a lookup has nothing to be nonlinear.

## Normalisation

LayerNorm standardises each token's feature vector across the hidden dimension,
then rescales it so the network can still express whatever magnitude it needs:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} z_i,
\qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(z_i - \mu)^2,
\qquad
\mathrm{LayerNorm}(z) = \gamma \odot \frac{z-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
$$

The statistics are taken over features, never over the batch, which is why a
LayerNorm network is independent of batch size. The blocks here are
**pre-LayerNorm** ($x + \mathrm{Attn}(\mathrm{LN}(x))$), which keeps the residual
path an identity and is what makes a 12-layer stack trainable at this scale.
Implemented in `models/gpt.py` as `LayerNorm`.

## Optimisation

Gradient descent, and the two refinements this trainer uses:

$$
\theta_{t+1} = \theta_t - \alpha\,\nabla_\theta \mathcal{L}(\theta_t)
$$

$$
\underbrace{\hat{m}_t = \beta_1 \hat{m}_{t-1} + (1-\beta_1)\,g_t}_{\text{momentum}},
\qquad
\underbrace{\hat{v}_t = \beta_2 \hat{v}_{t-1} + (1-\beta_2)\,g_t^2}_{\text{second moment}}
$$

$$
\theta_{t+1} = \theta_t - \alpha\left(\frac{\hat{m}_t}{1-\beta_1^t}\right)\Bigg/\left(\sqrt{\frac{\hat{v}_t}{1-\beta_2^t}} + \epsilon\right)
$$

That is AdamW's update, where $g_t = \nabla_\theta\mathcal{L}$ and this project
uses $\beta_1 = 0.9$, $\beta_2 = 0.95$ (`core/config.py`), $\epsilon = 10^{-8}$
— note that $\beta_2$, not the canonical $0.999$: this trainer runs short,
CPU-sized runs where the extra memory of a slower-decaying second moment buys
nothing. Dividing by $\sqrt{\hat v_t}$ makes the step
size roughly $\alpha$ regardless of gradient magnitude, and the bias correction
$m/(1-\beta^t)$ fixes the fact that $\hat m_t$ and $\hat v_t$ start at zero and
would otherwise bias the first steps towards zero. **Decoupled** weight decay
(`AdamW`) applies the penalty to the weights directly instead of folding it into
the gradient, which decouples it from the adaptive rescaling; it is applied to
matrices only ($p.\mathrm{dim} \ge 2$), never to biases or LayerNorm gains, with
$\lambda = 0.1$.

Three more pieces of the training loop:

| | Formula | Where |
|---|---|---|
| gradient clipping | $\min(1,\ \tau/\lVert g\rVert_2)\cdot g$ | bounds a single bad batch; $\tau = 1.0$ |
| linear warmup | $\eta_t = \eta_{max}\dfrac{t+1}{T_w}$ for $t < T_w$ | $T_w = \max(10, 0.05\,T)$ — 5% of the run |
| cosine decay | $\eta_t = \eta_{min} + \tfrac12(\eta_{max}-\eta_{min})(1+\cos(\pi p))$, $p = (t-T_w)/(T-T_w)$ | the `lr` column in every loss table; $\eta_{max}=3\times10^{-4}$, $\eta_{min}=3\times10^{-5}$ |
| backpropagation | $\dfrac{\partial \mathcal{L}}{\partial w} = \dfrac{\partial \mathcal{L}}{\partial a}\cdot\dfrac{\partial a}{\partial w}$, applied by the chain rule backwards through every op | `loss.backward()` |

## Sampling, and why temperature is a division

At generation time the logits are turned into a distribution and sampled from,
after three optional filters — `models/sampling.py`:

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Raising the temperature $T > 1$ **flattens** the distribution (logits are divided,
so differences shrink); $T < 1$ sharpens it; $T = 0$ is greedy decoding and takes
`argmax`. `top_k` keeps the $k$ largest logits, `top_p` keeps the smallest set
whose cumulative probability reaches $p$ (nucleus sampling), and
`repetition_penalty` divides positive logits / multiplies negative ones for tokens
already generated. `--predict` bypasses all of it and shows the untouched
distribution from which sampling would have drawn.

**Entropy** of that distribution, in nats, is the diagnostic used by `--predict`:

$$
H(p) = -\sum_i p_i \log p_i,
\qquad 0 \le H(p) \le \ln V
$$

$H = \ln V$ is uniform guessing; $H = 0$ is a single forced token. The measured
6.632 nats against $\ln 50257 = 10.825$ says the model is far from uniform but
far from confident.

## Evaluation metrics, and which ones apply

Perplexity is the exponentiated loss, and it is the one metric worth quoting for
a language model:

$$
\mathrm{PPL} = e^{\mathcal{L}}
\quad\Rightarrow\quad
\text{BPE val } e^{5.6159} = 275,\qquad
\text{byte-level val } e^{2.4614} = 11.7
$$

Those are **not comparable** across tokenizers — 275 sounds 20× worse than 11.7
while being the better model, because each covers a different amount of text.
Per character, the BPE model's perplexity is $e^{1.702} = 5.48$ against the
byte-level model's $e^{2.4614} = 11.7$.

Classification metrics are **not** what a language model is evaluated with — this
project has no labels and no decision threshold, so there is no confusion matrix
to build one from. For completeness, and because the question always comes up:

$$
\mathrm{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN},
\qquad
\mathrm{Precision} = \frac{TP}{TP+FP},
\qquad
\mathrm{Recall} = \frac{TP}{TP+FN}
$$

$$
F_1 = \frac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}
$$

$F_1$ is their harmonic mean, so it collapses when either one does — which is the
property you want when both matter. The confusion matrix itself is:

$$
C = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}
$$

## Descriptive statistics behind the environment scan

The scan that sizes a run to the machine (`core/system.py`, `core/environment.py`)
is built from these:

$$
\mu = \frac{1}{n}\sum_{i=1}^{n} x_i,
\qquad
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2,
\qquad
\sigma = \sqrt{\sigma^2}
$$

$$
\mathrm{Cov}(X,Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu_X)(y_i-\mu_Y),
\qquad
\rho = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\,\sigma_Y} \in [-1, 1]
$$

CPU utilisation is a *ratio of differences*, because the counters are cumulative
since boot — comparing two samples, not two absolutes:

$$
\text{util} = 100\left(1 - \frac{\Delta\,\text{idle}}{\Delta\,\text{idle} + \Delta\,\text{kernel} + \Delta\,\text{user}}\right)
$$

That is why `_split_cpu_line` exists, and why the scan reports the load average
$\left(\frac{1}{5m}\sum_{i} D_i,\ \frac{1}{15m}\sum_{i} D_i,\ \frac{1}{60m}\sum_{i} D_i\right)$
where $D_i$ is the number of runnable processes — the same quantity the live
scan compares against `DEFAULT_MAX_CPU_PERCENT = 85.0`.

## Deliberately not implemented

Naming these is more useful than quietly implying support:

| Method | Formula | Why not here |
|---|---|---|
| Linear regression | $y = \beta_0 + \beta_1 x$ | nothing here is a continuous target |
| Cost function (MSE) | $J(\theta)=\frac{1}{n}\sum_i (y_i-\hat y_i)^2$ | regression loss; language modelling uses cross-entropy |
| Logistic regression | $p = \sigma(z) = \frac{1}{1+e^{-z}}$, $\ \mathcal{L}=-\frac1n\sum_i\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$ | binary classification — a language model is a softmax over 50,257 classes, which *contains* this as the $K=2$ case |
| Softmax (multiclass) | $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | **used** — `models/sampling.py`, and it *is* the model's output layer |
| Naive Bayes | $P(y\mid x) = \dfrac{P(x\mid y)\,P(y)}{P(x)}$ | conditional independence is false for language |
| K-Means | $\arg\min_c \lVert x_i - \mu_{c_i}\rVert_2^2$ | no clustering step; the tokenizer vocabulary is not learned by clustering |
| SVM | $f(x) = w^\top x + b$, maximise margin subject to $y_i f(x_i) \ge 1$ | no support vectors, no kernel trick |
| L1 / L2 regularisation | $\lambda\sum_i\lvert\beta_i\rvert$ / $\lambda\sum_i \beta_i^2$ | **L2 via AdamW, weight decay only** — no L1, no sparse weights |
| Bias–variance | $\mathbb{E}[(y-\hat f(x))^2] = \mathrm{Bias}^2[\hat f] + \mathrm{Var}[\hat f] + \sigma^2$ | the decomposition is descriptive, not something a run reports |
| Gradient boosting (XGBoost, LightGBM) | $\hat y = \sum_{k} f_k(x)$, $f_k$ fits the residual gradient | trees do not tokenise; a 123.8M-parameter transformer is the model here. `sklearn` is not a dependency |
| Vector database | embeddings + ANN index (HNSW, IVF) | there is no retrieval step: the context is a fixed-size window of the corpus, not a search over a store |
| Mutual information | $I(X;Y) = H(X) - H(Y\mid X) = H(X)+H(Y)-H(X,Y)$ | meaningful, but nothing in this repo measures it; `--predict`'s entropy is the one place it would come from |
| KL divergence | $D_{\mathrm{KL}}(P\|Q)=\sum_i P_i\log\frac{P_i}{Q_i}$ | **used** — it *is* the training loss, cross-entropy minus the target entropy (see above) |

Fine-tuning, reinforcement learning from human feedback, LoRA and quantisation
are equally absent, and saying so is more useful than a stub: a 30M-parameter
model trained on one corpus is not a base model anybody can fine-tune
meaningfully. `--resume` continues *this* trainer's own checkpoints.

## Cost of one forward pass

The `full` preset's estimated FLOPs per token, and the reason the presets differ
so much in wall-clock:

$$
\mathrm{FLOPs} \approx 6N + 12\,LHd^2T
\qquad
(6N:\ \text{matmul},\;\; 12LHd^2T:\ \text{attention scores and values})
$$

with $N$ parameters, $L$ layers, $H$ heads, $d$ hidden, $T$ tokens. The second
term is quadratic in sequence length, which is exactly what the KV cache exists
to avoid: generating token $T+1$ with a cache costs one forward pass over the new
token only, not over the whole window. `models/builder.py::estimate_flops`
implements it.

---


---

Back to [the README](../README.md).


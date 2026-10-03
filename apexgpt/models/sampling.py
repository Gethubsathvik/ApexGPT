"""Token sampling: temperature, top-k, top-p (nucleus) and repetition penalty.

This is the single implementation used by the CLI, the GUI and the HTTP API.
Those three used to carry near-identical copies of this logic, which had
already drifted apart.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor,
                            penalty: float) -> torch.Tensor:
    """Divide positive logits and multiply negative ones by ``penalty``."""
    if not penalty or penalty == 1.0 or generated is None or generated.numel() == 0:
        return logits
    out = logits.clone()
    for token_id in torch.unique(generated):
        value = out[0, token_id]
        out[0, token_id] = value / penalty if value > 0 else value * penalty
    return out


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Keep only the ``top_k`` most likely tokens."""
    if not top_k or top_k <= 0:
        return logits
    k = min(int(top_k), logits.size(-1))
    if k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
    return logits.masked_fill(logits < kth, NEG_INF)


def apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative probability >= ``top_p``.

    The most likely token is always retained, so the distribution never becomes
    empty and ``multinomial`` cannot fail.
    """
    if top_p is None or top_p >= 1.0 or top_p <= 0.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)

    # remove a token only once the set *before* it already met the threshold
    remove_sorted = (cumulative - probs) > top_p
    remove_sorted[..., 0] = False

    remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, NEG_INF)


def next_token_distribution(logits: torch.Tensor, top_k: int | None = 10,
                            temperature: float = 1.0
                            ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The model's actual beliefs about the next token, before any sampling.

    Returns ``(ids, probs, entropy)``: the ``min(top_k, vocab)`` most likely
    token ids with their probabilities, sorted most likely first, and the
    entropy of the *whole* distribution in nats. Sampling throws most of this
    away; keeping it means "what does the model think comes next" is a question
    the tool can answer rather than a number it hides.
    """
    row = logits[:, -1, :].float() if logits.dim() == 3 else logits.float()
    if temperature > 0:
        row = row / temperature
    probs = F.softmax(row, dim=-1)
    # clamp before the log so a filtered token contributes 0 * -inf = nan
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = float(-(probs * log_probs).sum())
    k = min(int(top_k), probs.size(-1)) if top_k else probs.size(-1)
    top_probs, top_idx = torch.topk(probs, k, dim=-1)
    return top_idx[0], top_probs[0], entropy


def filtered_probs(logits: torch.Tensor, temperature: float = 1.0,
                   top_k: int | None = None, top_p: float | None = None,
                   repetition_penalty: float = 1.0,
                   generated: torch.Tensor | None = None) -> torch.Tensor:
    """The sampling distribution: temperature, penalties, top-k, top-p.

    Shared by :func:`sample_next_token` and the logprob path so the probability
    reported for a token is the one it was actually drawn from.
    """
    if temperature <= 0:
        # greedy decoding still needs a distribution; a one-hot on the argmax is
        # exactly what greedy picks, and it keeps logprob at 0.0
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs

    logits = logits.float() / temperature
    logits = apply_repetition_penalty(logits, generated, repetition_penalty)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    # guard against an all -inf row, which happens if every token is filtered
    probs = torch.nan_to_num(probs, nan=0.0)
    if float(probs.sum()) <= 0.0:
        probs = torch.zeros_like(probs)
        probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
    return probs


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0,
                      top_k: int | None = None, top_p: float | None = None,
                      repetition_penalty: float = 1.0,
                      generated: torch.Tensor | None = None,
                      generator=None) -> torch.Tensor:
    """Return one sampled token id of shape ``(batch, 1)``.

    ``logits`` is expected to be the final position only, shaped
    ``(batch, vocab)``.
    """
    probs = filtered_probs(logits, temperature=temperature, top_k=top_k,
                           top_p=top_p, repetition_penalty=repetition_penalty,
                           generated=generated)
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def sample_next_token_with_logprob(logits: torch.Tensor, temperature: float = 1.0,
                                   top_k: int | None = None,
                                   top_p: float | None = None,
                                   repetition_penalty: float = 1.0,
                                   generated: torch.Tensor | None = None,
                                   generator=None
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token *and* report how surprised the model was by it.

    The token is drawn from the filtered distribution (temperature, top-k,
    top-p, repetition penalty) exactly as :func:`sample_next_token` does. The
    reported log-probability is measured on the **untruncated** softmax at the
    same temperature, because that is the model's own uncertainty: after
    top-p collapses the distribution onto one token, a renormalised log-prob of
    0.0 would make every confident token look free.
    """
    probs = filtered_probs(logits, temperature=temperature, top_k=top_k,
                           top_p=top_p, repetition_penalty=repetition_penalty,
                           generated=generated)
    if temperature <= 0:
        ids = logits.argmax(dim=-1, keepdim=True)
        return ids, torch.zeros_like(ids, dtype=torch.float32)
    ids = torch.multinomial(probs, num_samples=1, generator=generator)

    honest = filtered_probs(logits, temperature=temperature)
    chosen = honest.gather(1, ids).clamp_min(1e-12).log()
    return ids, chosen

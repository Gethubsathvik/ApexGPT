"""The model layer: GPT architecture, construction, and sampling."""
from .builder import (build_model, count_parameters, estimate_flops,
                      print_summary, summary_lines)
from .gpt import Block, CausalSelfAttention, GPT, MLP
from .sampling import (apply_repetition_penalty, apply_top_k, apply_top_p,
                       filtered_probs, next_token_distribution,
                       sample_next_token, sample_next_token_with_logprob)

__all__ = [
    "GPT", "Block", "MLP", "CausalSelfAttention",
    "build_model", "count_parameters", "estimate_flops",
    "print_summary", "summary_lines",
    "sample_next_token", "sample_next_token_with_logprob",
    "next_token_distribution", "filtered_probs", "apply_top_k", "apply_top_p",
    "apply_repetition_penalty",
]

"""Printing text that the terminal can actually print.

Decoded model output is arbitrary text. On a byte-level tokenizer a truncated
multi-byte character decodes to U+FFFD, and on a legacy Windows console
(``cp1252``) printing that raises ``UnicodeEncodeError`` mid-stream - so a
generation run dies at the first bad character instead of showing it.

:func:`console_safe` is the one place that knows about the terminal's codec.
"""
from __future__ import annotations

import sys


def console_encoding(stream=None) -> str:
    """The codec the given stream (stdout by default) expects."""
    target = stream if stream is not None else sys.stdout
    return getattr(target, "encoding", None) or "utf-8"


def console_safe(text: str, encoding: str | None = None) -> str:
    """Replace anything ``encoding`` cannot represent, e.g. U+FFFD on cp1252."""
    if not text:
        return text
    codec = encoding or console_encoding()
    try:
        return text.encode(codec, errors="replace").decode(codec, errors="replace")
    except (LookupError, UnicodeError):        # unknown or broken codec
        return text
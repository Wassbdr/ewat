"""Lexical entropy of log-line token distributions.

L_LEXICAL_ENTROPY = H(token distribution over a log window)
H = -Σ p_i log2(p_i)

High entropy → diverse vocabulary (possibly many unique error messages).
Low entropy  → repetitive logs (normal heartbeats, periodic health-checks).

Tokenisation: whitespace split after stripping ANSI codes and timestamps.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Regex to strip common log prefixes: timestamps (ISO-8601, epoch ms, level tag)
_STRIP_PATTERN = re.compile(
    r"""
    (?:
        \d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?  # ISO timestamp
        |\d{10,13}  # epoch sec/ms
        |\[?(?:DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\]?  # log level
        |\x1B\[[0-9;]*m  # ANSI colours
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def tokenise(line: str) -> list[str]:
    """Strip timestamps/levels/ANSI then split on whitespace.

    Parameters
    ----------
    line:
        Raw log line string.

    Returns
    -------
    list[str]
        Lower-cased tokens with len >= 2 (drops single-char noise).
    """
    cleaned = _STRIP_PATTERN.sub(" ", line)
    return [tok.lower() for tok in cleaned.split() if len(tok) >= 2]


def lexical_entropy(log_lines: list[str]) -> float:
    """Compute lexical entropy H of the token distribution over a log window.

    Parameters
    ----------
    log_lines:
        List of raw log line strings from the current window.

    Returns
    -------
    float
        Shannon entropy in bits (log base 2). Returns 0.0 if the window is
        empty or contains only one unique token.
    """
    if not log_lines:
        return 0.0

    counts: Counter[str] = Counter()
    for line in log_lines:
        counts.update(tokenise(line))

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy

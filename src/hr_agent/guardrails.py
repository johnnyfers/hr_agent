"""Pre/post processing for safety, privacy, and prompt-injection resistance.

These checks run *outside* the LLM. The LLM also has guardrail instructions
in its system prompt, but defence-in-depth: we trust neither the model nor
the user input in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Length cap per user message — protects from token-flooding and copy-paste of
# entire CVs that would derail the screening.
MAX_USER_MESSAGE_CHARS = 2000

# Patterns that indicate prompt-injection attempts against our system prompt.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instruct", re.I),
    re.compile(r"disregard\s+(your|the)\s+(system\s+)?prompt", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"act\s+as\s+(a|an)\s+(?!delivery)", re.I),  # allow "act as a delivery driver"
    re.compile(r"reveal\s+(your|the)\s+(system\s+)?prompt", re.I),
    re.compile(r"olvida\s+(las\s+)?instrucciones", re.I),
    re.compile(r"ignora\s+(las\s+)?instrucciones", re.I),
]

# Off-topic patterns that warrant a soft redirect.
_OFFTOPIC_PATTERNS = [
    re.compile(r"\b(loan|prestamo|préstamo|credit\s*card)\b", re.I),
    re.compile(r"\b(visa|inmigracion|inmigración|immigration\s+advice)\b", re.I),
]

# Inappropriate / abusive language — high-confidence subset only. We err on
# the side of letting the LLM handle nuance; this is for unambiguous cases.
_ABUSE_PATTERNS = [
    re.compile(r"\b(fuck\s+you|hijo\s+de\s+puta|pendejo\s+de\s+mierda)\b", re.I),
]

# PII patterns we redact from logs (not from messages — candidates need to
# share their info to apply). Used for analytics exports.
_PII_PATTERNS = [
    (re.compile(r"\b\d{8}[A-Z]\b"), "[DNI]"),  # Spanish DNI
    (re.compile(r"\b[A-Z]{4}\d{6}[A-Z0-9]{3}\b"), "[CURP]"),  # Mexican CURP
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"(?:\+\d{1,3}[\s-]?)?\(?\d{2,3}\)?[\s.-]?\d{3}[\s.-]?\d{3,4}"), "[PHONE]"),
]


@dataclass
class GuardrailResult:
    allowed: bool
    cleaned_input: str
    flag: Optional[str] = None  # "injection" | "abuse" | "offtopic" | "too_long"
    note: Optional[str] = None  # human-readable reason


def check_user_input(text: str) -> GuardrailResult:
    """First line of defence on every incoming user message."""
    if text is None:
        return GuardrailResult(False, "", flag="empty", note="empty message")

    cleaned = text.strip()
    if not cleaned:
        return GuardrailResult(False, "", flag="empty", note="empty message")

    if len(cleaned) > MAX_USER_MESSAGE_CHARS:
        # Truncate rather than reject — preserve enough to extract the intent.
        cleaned = cleaned[:MAX_USER_MESSAGE_CHARS]
        return GuardrailResult(
            True,
            cleaned,
            flag="too_long",
            note=f"truncated from {len(text)} to {MAX_USER_MESSAGE_CHARS} chars",
        )

    for pat in _INJECTION_PATTERNS:
        if pat.search(cleaned):
            return GuardrailResult(
                True,  # we still process, but with a flag the agent sees
                cleaned,
                flag="injection",
                note="possible prompt-injection attempt",
            )

    for pat in _ABUSE_PATTERNS:
        if pat.search(cleaned):
            return GuardrailResult(
                True,
                cleaned,
                flag="abuse",
                note="abusive language detected",
            )

    for pat in _OFFTOPIC_PATTERNS:
        if pat.search(cleaned):
            return GuardrailResult(
                True,
                cleaned,
                flag="offtopic",
                note="off-topic question",
            )

    return GuardrailResult(True, cleaned)


def redact_pii(text: str) -> str:
    """Remove obvious PII before logs/analytics export."""
    if not text:
        return text
    out = text
    for pat, replacement in _PII_PATTERNS:
        out = pat.sub(replacement, out)
    return out

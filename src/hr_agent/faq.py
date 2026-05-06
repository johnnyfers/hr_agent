"""Lightweight FAQ retrieval over a JobSpec's FAQ list.

Keyword/tag matching with diacritic-insensitive scoring. Returns the best
match plus a confidence score so the agent can decide whether to quote it
or punt to a recruiter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .jobspec import FAQEntry
from .validators import _normalize


@dataclass
class FAQResult:
    id: str
    answer: str
    score: float
    matched_terms: list[str]


def search(
    query: str,
    entries: list[FAQEntry],
    language: str = "es",
    min_score: float = 0.2,
) -> Optional[FAQResult]:
    """Return the best-matching FAQ entry for ``query`` in ``language``."""
    if not query or not entries:
        return None
    q_norm = _normalize(query)
    q_tokens = set(q_norm.split())
    if not q_tokens:
        return None

    best: Optional[FAQResult] = None
    for entry in entries:
        matched: list[str] = []
        score = 0.0
        for tag in entry.tags:
            tag_norm = _normalize(tag)
            if tag_norm and tag_norm in q_norm:
                score = max(score, 1.0)
                matched.append(tag)
                continue
            tag_tokens = set(tag_norm.split())
            overlap = q_tokens & tag_tokens
            if overlap:
                partial = 0.5 * len(overlap) / max(len(tag_tokens), 1)
                if partial > score:
                    score = partial
                matched.extend(overlap)

        if score >= min_score and (best is None or score > best.score):
            answer = entry.text.get(language) or entry.text.get("es") or next(iter(entry.text.values()), "")
            best = FAQResult(id=entry.id, answer=answer, score=score, matched_terms=matched)
    return best

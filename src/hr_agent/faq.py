"""Lightweight FAQ retrieval.

The FAQ has ~10 entries — vector embeddings would be overkill. We do
keyword/tag matching with diacritic-insensitive scoring. Returns the
best match plus a confidence score so the agent can decide whether to
quote it or punt to a recruiter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Literal, Optional

from .validators import _normalize


@dataclass
class FAQResult:
    id: str
    answer: str
    score: float
    matched_terms: list[str]


def _load_faq() -> dict:
    with resources.files("hr_agent.data").joinpath("faq.json").open() as f:
        return json.load(f)


_FAQ = _load_faq()


def search(query: str, language: Literal["es", "en"] = "es", min_score: float = 0.2) -> Optional[FAQResult]:
    """Return the best-matching FAQ entry, or None if below threshold."""
    if not query:
        return None
    q_norm = _normalize(query)
    q_tokens = set(q_norm.split())
    if not q_tokens:
        return None

    best: Optional[FAQResult] = None
    for entry in _FAQ["faqs"]:
        matched: list[str] = []
        score = 0.0
        for tag in entry["tags"]:
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
            best = FAQResult(
                id=entry["id"],
                answer=entry[language],
                score=score,
                matched_terms=matched,
            )
    return best


def company_blurb(language: Literal["es", "en"] = "es") -> str:
    return _FAQ["company"][language]

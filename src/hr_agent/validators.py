"""Deterministic validation for screening fields.

The LLM proposes values via tool calls; this module is the source of truth
for whether those values are accepted. Keeping validation out of the LLM
makes it testable and gives us reliable disqualification logic.
"""

from __future__ import annotations

import json
import re
import unicodedata
from importlib import resources
from typing import Optional

from .schema import Availability, Schedule


def _normalize(s: str) -> str:
    """Lowercase + strip diacritics for fuzzy comparison."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def _load_service_areas() -> dict:
    with resources.files("hr_agent.data").joinpath("service_areas.json").open() as f:
        return json.load(f)


_SERVICE_AREAS = _load_service_areas()


def validate_name(value: str) -> Optional[str]:
    """Return cleaned name or None if invalid."""
    if not value:
        return None
    cleaned = " ".join(value.split())
    if len(cleaned) > 80 or len(cleaned) < 2:
        return None
    if not re.match(r"^[a-zA-Z\s\-'\.áéíóúüñÁÉÍÓÚÜÑ]+$", cleaned):
        return None
    if len(cleaned.split()) < 2:
        return None  # at least first + last
    return cleaned


def validate_city(value: str) -> tuple[Optional[str], Optional[str], bool]:
    """Resolve a candidate-typed city to a canonical service area.

    Returns ``(canonical_city, country, in_service_area)``. If the input
    matches an alias or a known city (case/diacritics-insensitive), the
    canonical name and country are returned. If unmatched, returns
    ``(input_stripped, None, False)`` so the recruiter still sees what
    the candidate said.
    """
    if not value:
        return None, None, False
    raw = value.strip()
    if raw in _SERVICE_AREAS["aliases"]:
        canonical = _SERVICE_AREAS["aliases"][raw]
    else:
        canonical = None
        norm = _normalize(raw)
        for alias, target in _SERVICE_AREAS["aliases"].items():
            if _normalize(alias) == norm:
                canonical = target
                break

    candidates = canonical and [canonical] or [raw]
    for cand in candidates + [raw]:
        norm = _normalize(cand)
        for country in ("ES", "MX"):
            for city in _SERVICE_AREAS[country]:
                if _normalize(city) == norm:
                    return city, country, True

    return raw, None, False


_LICENSE_YES = {"yes", "y", "si", "sí", "claro", "tengo", "tengo licencia", "yep", "yeah"}
_LICENSE_NO = {"no", "nope", "no tengo", "todavia no", "todavía no", "aun no", "aún no", "not yet"}


def validate_license(value: str) -> Optional[bool]:
    """Map free-text license response to a bool, or None if ambiguous."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    norm = _normalize(str(value))
    if norm in _LICENSE_YES:
        return True
    if norm in _LICENSE_NO:
        return False
    if any(p in norm for p in ("no tengo", "no licen", "don't have", "do not have")):
        return False
    if any(p in norm for p in ("tengo", "i have", "have one", "have a", "permiso b")):
        return True
    return None


def validate_availability(value: str) -> Optional[Availability]:
    if not value:
        return None
    norm = _normalize(value)
    mapping = {
        Availability.full_time: ("full", "tiempo completo", "completo", "jornada completa"),
        Availability.part_time: ("part", "medio", "parcial", "media jornada"),
        Availability.weekends_only: ("weekend", "fin de semana", "fines de semana", "sabado", "domingo"),
        Availability.flexible: ("flex", "cuando", "lo que sea"),
    }
    for value_enum, keys in mapping.items():
        if any(k in norm for k in keys):
            return value_enum
    try:
        return Availability(norm)
    except ValueError:
        return None


def validate_schedule(value: str) -> Optional[Schedule]:
    if not value:
        return None
    norm = _normalize(value)
    mapping = {
        Schedule.morning: ("morning", "mañana", "manana", "am"),
        Schedule.afternoon: ("afternoon", "tarde", "midday"),
        Schedule.evening: ("evening", "noche temprana", "early night"),
        Schedule.night: ("night", "noche", "madrugada"),
        Schedule.flexible: ("flex", "any", "cualquier"),
    }
    for value_enum, keys in mapping.items():
        if any(k in norm for k in keys):
            return value_enum
    try:
        return Schedule(norm)
    except ValueError:
        return None


def validate_experience_years(value) -> Optional[int]:
    """Accept int, numeric string, or 'none'/'ninguna' → 0."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is subclass of int; reject explicitly
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 40 else None
    norm = _normalize(str(value))
    if norm in {"none", "ninguna", "ninguno", "ningún", "ningun", "zero", "cero", "0"}:
        return 0
    m = re.search(r"\d+", norm)
    if m:
        n = int(m.group())
        return n if 0 <= n <= 40 else None
    return None


_KNOWN_PLATFORMS = {
    "glovo": "Glovo",
    "uber": "Uber Eats",
    "uber eats": "Uber Eats",
    "ubereats": "Uber Eats",
    "rappi": "Rappi",
    "just eat": "Just Eat",
    "justeat": "Just Eat",
    "deliveroo": "Deliveroo",
    "doordash": "DoorDash",
    "grubhub": "Grubhub",
    "didi": "DiDi Food",
    "didi food": "DiDi Food",
    "sin delay": "Sin Delay",
}


def normalize_platforms(values: list[str]) -> list[str]:
    """Canonicalize platform names; preserve unknown ones as-is."""
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if not v:
            continue
        key = _normalize(v)
        canonical = _KNOWN_PLATFORMS.get(key, v.strip())
        if canonical.lower() not in seen:
            out.append(canonical)
            seen.add(canonical.lower())
    return out

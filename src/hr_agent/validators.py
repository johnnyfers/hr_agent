"""Field validation.

The LLM proposes values via tool calls; this module is the source of truth
for whether those values are accepted. ``validate_field`` dispatches on
``FieldSpec.type`` and returns either a normalised value or an error
message the model can use to ask again.

Keeping validation deterministic (no LLM in the loop) means we can cover
every field type with a fast pytest run, independent of model output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .jobspec import FieldSpec, JobSpec, ServiceAreas


# --- Generic helpers --------------------------------------------------------


def _normalize(s: str) -> str:
    """Lowercase + strip diacritics for fuzzy comparison."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


@dataclass
class ValidationResult:
    """Outcome of validating one field."""

    ok: bool
    value: Any = None
    error: Optional[str] = None
    disqualifying: bool = False  # value is valid but disqualifies the candidate
    reason: Optional[str] = None  # human/recruiter-readable, e.g. "no_license"


# --- Per-type validators ----------------------------------------------------


_LICENSE_YES = {"yes", "y", "si", "sí", "claro", "tengo", "tengo licencia", "yep", "yeah"}
_LICENSE_NO = {"no", "nope", "no tengo", "todavia no", "todavía no", "aun no", "aún no", "not yet"}


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
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


def _validate_bool(field: FieldSpec, value: Any) -> ValidationResult:
    v = _coerce_bool(value)
    if v is None:
        return ValidationResult(ok=False, error=f"ambiguous value for {field.name}, re-ask")
    disqualifying = (
        (field.disqualify_when == "false" and v is False)
        or (field.disqualify_when == "true" and v is True)
    )
    return ValidationResult(
        ok=True,
        value=v,
        disqualifying=disqualifying,
        reason=field.name if disqualifying else None,
    )


def _validate_string(field: FieldSpec, value: Any) -> ValidationResult:
    if value is None:
        return ValidationResult(ok=False, error="missing value")
    s = " ".join(str(value).split())
    if not s or len(s) > 200:
        return ValidationResult(ok=False, error="invalid length")
    # For names specifically, require at least two tokens of letters.
    if field.name == "full_name":
        if len(s) < 2 or len(s) > 80:
            return ValidationResult(ok=False, error="invalid name length")
        if not re.match(r"^[a-zA-Z\s\-'\.áéíóúüñÁÉÍÓÚÜÑ]+$", s):
            return ValidationResult(ok=False, error="invalid characters in name")
        if len(s.split()) < 2:
            return ValidationResult(ok=False, error="need first + last name")
    return ValidationResult(ok=True, value=s)


def _validate_int(field: FieldSpec, value: Any) -> ValidationResult:
    if isinstance(value, bool):  # bool is int subclass — reject explicitly
        return ValidationResult(ok=False, error="expected number")
    if isinstance(value, int):
        n = value
    else:
        norm = _normalize(str(value or ""))
        if norm in {"none", "ninguna", "ninguno", "ningún", "ningun", "zero", "cero"}:
            n = 0
        else:
            m = re.search(r"-?\d+", norm)
            if not m:
                return ValidationResult(ok=False, error="not a number")
            n = int(m.group())
    lo = field.min if field.min is not None else 0
    hi = field.max if field.max is not None else 1_000_000
    if not (lo <= n <= hi):
        return ValidationResult(ok=False, error=f"out of range [{lo}, {hi}]")
    return ValidationResult(ok=True, value=n)


def _validate_enum(field: FieldSpec, value: Any) -> ValidationResult:
    if not field.enum_values:
        return ValidationResult(ok=False, error="enum field has no allowed values")
    if value is None:
        return ValidationResult(ok=False, error="missing value")
    norm = _normalize(str(value))
    # Direct match (case/diacritic-insensitive)
    for choice in field.enum_values:
        if _normalize(choice) == norm:
            return ValidationResult(ok=True, value=choice)
    # Loose-substring fallback for friendly free-text answers
    for choice in field.enum_values:
        if _normalize(choice) in norm or norm in _normalize(choice):
            return ValidationResult(ok=True, value=choice)
    # Heuristic synonyms — keep small and obvious
    synonyms = {
        "full_time": ("full", "tiempo completo", "completo", "jornada completa"),
        "part_time": ("part", "medio", "parcial", "media jornada"),
        "weekends_only": ("weekend", "fin de semana", "fines de semana", "sabado", "domingo"),
        "flexible": ("flex", "cuando", "lo que sea", "any", "cualquier"),
        "morning": ("morning", "mañana", "manana", "am"),
        "afternoon": ("afternoon", "tarde", "midday"),
        "evening": ("evening", "noche temprana", "early night"),
        "night": ("night", "noche", "madrugada"),
    }
    for choice in field.enum_values:
        for syn in synonyms.get(choice, ()):
            if syn in norm:
                return ValidationResult(ok=True, value=choice)
    options = " / ".join(field.enum_values)
    disqualifying = (
        field.disqualify_when == "out_of_set"
        and any(_normalize(v) == norm for v in field.disqualify_values)
    )
    return ValidationResult(
        ok=False,
        error=f"must be one of {options}",
        disqualifying=disqualifying,
        reason=field.name if disqualifying else None,
    )


def _validate_city(field: FieldSpec, value: Any, areas: Optional[ServiceAreas]) -> ValidationResult:
    raw = (str(value).strip() if value is not None else "")
    if not raw:
        return ValidationResult(ok=False, error="missing city")

    canonical: Optional[str] = None
    country: Optional[str] = None
    in_area = False

    if areas:
        # Alias lookup (exact and diacritic-insensitive)
        norm_raw = _normalize(raw)
        for alias, target in areas.aliases.items():
            if alias == raw or _normalize(alias) == norm_raw:
                canonical = target
                break
        # Canonical city scan
        for cand in [canonical, raw]:
            if not cand:
                continue
            cnorm = _normalize(cand)
            for ctry, cities in areas.countries.items():
                for city in cities:
                    if _normalize(city) == cnorm:
                        canonical = city
                        country = ctry
                        in_area = True
                        break
                if in_area:
                    break
            if in_area:
                break
    else:
        # No service-area filter configured — accept any non-empty city.
        canonical = raw
        in_area = True

    record = {
        "raw": raw,
        "canonical": canonical,
        "country": country,
        "in_service_area": in_area,
    }
    disqualifying = (
        field.disqualify_when == "out_of_service_area" and not in_area
    )
    return ValidationResult(
        ok=True,
        value=record,
        disqualifying=disqualifying,
        reason=("out_of_zone:" + raw) if disqualifying else None,
    )


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


def _normalize_platforms(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if not v:
            continue
        key = _normalize(v)
        canonical = _KNOWN_PLATFORMS.get(key, str(v).strip())
        if canonical.lower() not in seen:
            out.append(canonical)
            seen.add(canonical.lower())
    return out


def _validate_experience(field: FieldSpec, value: Any) -> ValidationResult:
    if not isinstance(value, dict):
        return ValidationResult(ok=False, error="experience must be {years, platforms}")
    yrs = value.get("years")
    if isinstance(yrs, bool):
        return ValidationResult(ok=False, error="expected number for years")
    if isinstance(yrs, int):
        n = yrs
    else:
        norm = _normalize(str(yrs or ""))
        if norm in {"none", "ninguna", "ninguno", "zero", "cero"}:
            n = 0
        else:
            m = re.search(r"\d+", norm)
            if not m:
                return ValidationResult(ok=False, error="invalid years")
            n = int(m.group())
    if not (0 <= n <= 40):
        return ValidationResult(ok=False, error="years out of range (0-40)")
    platforms = _normalize_platforms(value.get("platforms") or [])
    return ValidationResult(ok=True, value={"years": n, "platforms": platforms})


def _validate_date(field: FieldSpec, value: Any) -> ValidationResult:
    """Normalize common free-text dates to ISO ``YYYY-MM-DD``.

    Supported examples:
    - absolute: ``2026-05-19``
    - relative days/weeks: ``tomorrow``, ``in 2 weeks``, ``una semana``
    - next weekday: ``next monday``, ``próximo lunes``
    """
    if value is None:
        return ValidationResult(ok=False, error="missing date")
    today = datetime.now(timezone.utc).date()
    s = str(value).strip()
    if not s or len(s) > 100:
        return ValidationResult(ok=False, error="invalid date string")

    # ISO date first.
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        if d < today:
            return ValidationResult(ok=False, error="date is in the past")
        return ValidationResult(ok=True, value=d.isoformat())
    except ValueError:
        pass

    norm = _normalize(s)
    norm = re.sub(r"\s+", " ", norm).strip()

    # Simple direct tokens.
    if norm in {"today", "hoy"}:
        return ValidationResult(ok=True, value=today.isoformat())
    if norm in {"tomorrow", "manana"}:
        return ValidationResult(ok=True, value=(today + timedelta(days=1)).isoformat())

    number_words = {
        "a": 1,
        "an": 1,
        "one": 1,
        "una": 1,
        "un": 1,
        "dos": 2,
        "two": 2,
        "tres": 3,
        "three": 3,
        "cuatro": 4,
        "four": 4,
    }

    def parse_qty(token: str) -> Optional[int]:
        if token.isdigit():
            return int(token)
        return number_words.get(token)

    month_map = {
        "january": 1,
        "jan": 1,
        "enero": 1,
        "february": 2,
        "feb": 2,
        "febrero": 2,
        "march": 3,
        "mar": 3,
        "marzo": 3,
        "april": 4,
        "apr": 4,
        "abril": 4,
        "may": 5,
        "mayo": 5,
        "june": 6,
        "jun": 6,
        "junio": 6,
        "july": 7,
        "jul": 7,
        "julio": 7,
        "august": 8,
        "aug": 8,
        "agosto": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "septiembre": 9,
        "october": 10,
        "oct": 10,
        "octubre": 10,
        "november": 11,
        "nov": 11,
        "noviembre": 11,
        "december": 12,
        "dec": 12,
        "diciembre": 12,
    }

    def infer_date(day: int, month: int, year: Optional[int]) -> Optional[datetime.date]:
        try:
            if year is not None:
                return datetime(year, month, day).date()
            d = datetime(today.year, month, day).date()
            # If user omitted year and this date has already passed, assume next year.
            if d < today:
                d = datetime(today.year + 1, month, day).date()
            return d
        except ValueError:
            return None

    # Patterns like "in 2 weeks", "2 weeks", "one week", "en una semana".
    m = re.search(
        r"^(?:in |en )?(?P<n>\d+|a|an|one|two|three|four|un|una|dos|tres|cuatro)\s+"
        r"(?P<u>day|days|dia|dias|week|weeks|semana|semanas)$",
        norm,
    )
    if m:
        n = parse_qty(m.group("n"))
        if n is None or n < 0:
            return ValidationResult(ok=False, error="invalid relative date")
        unit = m.group("u")
        if unit in {"day", "days", "dia", "dias"}:
            d = today + timedelta(days=n)
        else:
            d = today + timedelta(weeks=n)
        return ValidationResult(ok=True, value=d.isoformat())

    # Month-name dates:
    # - "may 7", "may 7 2026"
    # - "7 may", "7 de mayo", "7 mayo 2026"
    m = re.search(
        r"^(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:[,\s]+(?P<year>\d{4}))?$",
        norm,
    )
    if not m:
        m = re.search(
            r"^(?P<day>\d{1,2})(?:\s+de)?\s+(?P<month>[a-z]+)(?:[,\s]+(?P<year>\d{4}))?$",
            norm,
        )
    if m:
        month = month_map.get(m.group("month"))
        day = int(m.group("day"))
        year = int(m.group("year")) if m.group("year") else None
        if month is None:
            return ValidationResult(ok=False, error="unrecognized month")
        d = infer_date(day, month, year)
        if d is None:
            return ValidationResult(ok=False, error="invalid calendar date")
        if d < today:
            return ValidationResult(ok=False, error="date is in the past")
        return ValidationResult(ok=True, value=d.isoformat())

    # Weekday forms:
    # - "next monday", "proximo lunes"
    # - "el lunes", "this monday", "este lunes"
    weekday_map = {
        "monday": 0,
        "lunes": 0,
        "tuesday": 1,
        "martes": 1,
        "wednesday": 2,
        "miercoles": 2,
        "thursday": 3,
        "jueves": 3,
        "friday": 4,
        "viernes": 4,
        "saturday": 5,
        "sabado": 5,
        "sunday": 6,
        "domingo": 6,
    }
    m = re.search(
        r"^(?:next|proximo|proxima)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|lunes|martes|miercoles|jueves|viernes|sabado|domingo)$",
        norm,
    )
    if m:
        target = weekday_map[m.group(1)]
        delta = (target - today.weekday()) % 7
        if delta == 0:
            delta = 7
        return ValidationResult(ok=True, value=(today + timedelta(days=delta)).isoformat())

    m = re.search(
        r"^(?:el|this|este|esta)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|lunes|martes|miercoles|jueves|viernes|sabado|domingo)$",
        norm,
    )
    if m:
        target = weekday_map[m.group(1)]
        delta = (target - today.weekday()) % 7
        if delta == 0:
            delta = 7
        return ValidationResult(ok=True, value=(today + timedelta(days=delta)).isoformat())

    return ValidationResult(ok=False, error="unrecognized date format")


# --- Public entry point -----------------------------------------------------


def validate_field(field: FieldSpec, value: Any, job: JobSpec) -> ValidationResult:
    """Validate ``value`` for ``field`` in the context of ``job``."""
    if field.type == "bool":
        return _validate_bool(field, value)
    if field.type == "string":
        return _validate_string(field, value)
    if field.type == "int":
        return _validate_int(field, value)
    if field.type == "enum":
        return _validate_enum(field, value)
    if field.type == "city":
        return _validate_city(field, value, job.service_areas)
    if field.type == "experience":
        return _validate_experience(field, value)
    if field.type == "date":
        return _validate_date(field, value)
    return ValidationResult(ok=False, error=f"unknown field type {field.type}")

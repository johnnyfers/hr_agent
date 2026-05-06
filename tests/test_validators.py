"""Validator tests — these are the deterministic core, must be airtight.

We exercise the type-dispatched ``validate_field`` against the seeded
Grupo Sazón JobSpec so the tests reflect what the running agent sees.
"""

import pytest
from datetime import datetime, timedelta, timezone

from hr_agent.validators import validate_field


# --- helpers ---------------------------------------------------------------

def _field(spec, name):
    return spec.field(name)


# --- bool (license) --------------------------------------------------------


class TestLicense:
    @pytest.mark.parametrize("v,expected_bool,expected_dq", [
        ("yes", True, False),
        ("Yes", True, False),
        ("sí", True, False),
        ("Tengo licencia", True, False),
        ("I have one", True, False),
        ("no", False, True),
        ("No", False, True),
        ("no tengo", False, True),
        ("not yet", False, True),
        ("todavía no", False, True),
        (True, True, False),
        (False, False, True),
    ])
    def test_resolves_yes_no(self, grupo_sazon_spec, v, expected_bool, expected_dq):
        f = _field(grupo_sazon_spec, "has_license")
        r = validate_field(f, v, grupo_sazon_spec)
        assert r.ok is True
        assert r.value is expected_bool
        assert r.disqualifying is expected_dq

    @pytest.mark.parametrize("v", ["maybe", "idk", "kinda"])
    def test_ambiguous_rejected(self, grupo_sazon_spec, v):
        f = _field(grupo_sazon_spec, "has_license")
        r = validate_field(f, v, grupo_sazon_spec)
        assert r.ok is False


# --- string (full_name) ----------------------------------------------------


class TestFullName:
    def test_valid(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "full_name")
        r = validate_field(f, "María García", grupo_sazon_spec)
        assert r.ok and r.value == "María García"

    def test_collapses_whitespace(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "full_name")
        r = validate_field(f, "  Juan   Pérez  ", grupo_sazon_spec)
        assert r.ok and r.value == "Juan Pérez"

    def test_rejects_single_token(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "full_name")
        assert not validate_field(f, "Juan", grupo_sazon_spec).ok

    def test_rejects_digits(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "full_name")
        assert not validate_field(f, "Juan 2 Pérez", grupo_sazon_spec).ok

    def test_rejects_too_short(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "full_name")
        assert not validate_field(f, "J", grupo_sazon_spec).ok


# --- city ------------------------------------------------------------------


class TestCity:
    def test_canonical_match(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "city")
        r = validate_field(f, "Madrid", grupo_sazon_spec)
        assert r.ok and r.value["canonical"] == "Madrid" and r.value["country"] == "ES"
        assert r.value["in_service_area"] is True
        assert r.disqualifying is False

    def test_alias_resolves(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "city")
        r = validate_field(f, "CDMX", grupo_sazon_spec)
        assert r.ok and r.value["canonical"] == "Ciudad de México"
        assert r.value["country"] == "MX"

    def test_diacritic_insensitive(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "city")
        r = validate_field(f, "ciudad de mexico", grupo_sazon_spec)
        assert r.ok and r.value["in_service_area"] is True

    def test_unknown_city_disqualifies(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "city")
        r = validate_field(f, "Springfield", grupo_sazon_spec)
        assert r.ok  # the call succeeded — the city is recorded
        assert r.value["in_service_area"] is False
        assert r.disqualifying is True
        assert r.reason and "Springfield" in r.reason


# --- enum (availability, schedule) -----------------------------------------


class TestAvailability:
    @pytest.mark.parametrize("v,expected", [
        ("full time", "full_time"),
        ("Tiempo completo", "full_time"),
        ("part-time please", "part_time"),
        ("Media jornada", "part_time"),
        ("solo fines de semana", "weekends_only"),
        ("flex", "flexible"),
        ("flexible", "flexible"),
    ])
    def test_resolves(self, grupo_sazon_spec, v, expected):
        f = _field(grupo_sazon_spec, "availability")
        r = validate_field(f, v, grupo_sazon_spec)
        assert r.ok and r.value == expected

    def test_garbage_rejected(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "availability")
        assert not validate_field(f, "whenever I feel like it", grupo_sazon_spec).ok


class TestSchedule:
    @pytest.mark.parametrize("v,expected", [
        ("morning", "morning"),
        ("Mañana", "morning"),
        ("tarde", "afternoon"),
        ("evening", "evening"),
        ("nights please", "night"),
        ("flexible", "flexible"),
    ])
    def test_resolves(self, grupo_sazon_spec, v, expected):
        f = _field(grupo_sazon_spec, "preferred_schedule")
        r = validate_field(f, v, grupo_sazon_spec)
        assert r.ok and r.value == expected


# --- experience ------------------------------------------------------------


class TestExperience:
    def test_valid(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "experience")
        r = validate_field(f, {"years": 3, "platforms": ["glovo", "Uber Eats"]}, grupo_sazon_spec)
        assert r.ok
        assert r.value == {"years": 3, "platforms": ["Glovo", "Uber Eats"]}

    def test_zero_years(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "experience")
        r = validate_field(f, {"years": 0, "platforms": []}, grupo_sazon_spec)
        assert r.ok and r.value["years"] == 0

    def test_string_years_resolved(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "experience")
        r = validate_field(f, {"years": "about 2", "platforms": []}, grupo_sazon_spec)
        assert r.ok and r.value["years"] == 2

    def test_dedupes_platforms(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "experience")
        r = validate_field(f, {"years": 1, "platforms": ["GLOVO", "glovo"]}, grupo_sazon_spec)
        assert r.value["platforms"] == ["Glovo"]

    def test_out_of_range(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "experience")
        assert not validate_field(f, {"years": 50, "platforms": []}, grupo_sazon_spec).ok

    def test_bad_shape(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "experience")
        assert not validate_field(f, "lots", grupo_sazon_spec).ok


# --- date ------------------------------------------------------------------


class TestStartDate:
    def test_accepts_relative_one_week(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "start_date")
        today = datetime.now(timezone.utc).date()
        r = validate_field(f, "one week", grupo_sazon_spec)
        assert r.ok
        parsed = datetime.strptime(r.value, "%Y-%m-%d").date()
        assert parsed == today + timedelta(days=7)

    def test_accepts_iso(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "start_date")
        assert validate_field(f, "2026-05-19", grupo_sazon_spec).ok

    def test_accepts_next_weekday(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "start_date")
        r = validate_field(f, "next monday", grupo_sazon_spec)
        assert r.ok
        parsed = datetime.strptime(r.value, "%Y-%m-%d").date()
        assert parsed > datetime.now(timezone.utc).date()

    def test_rejects_empty(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "start_date")
        assert not validate_field(f, "", grupo_sazon_spec).ok

    def test_rejects_past_date(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "start_date")
        assert not validate_field(f, "2000-01-01", grupo_sazon_spec).ok

    def test_rejects_overlong(self, grupo_sazon_spec):
        f = _field(grupo_sazon_spec, "start_date")
        assert not validate_field(f, "x" * 200, grupo_sazon_spec).ok

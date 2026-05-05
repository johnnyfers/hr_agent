"""Validator tests — these are the deterministic core, must be airtight."""

import pytest

from hr_agent.schema import Availability, Schedule
from hr_agent.validators import (
    normalize_platforms,
    validate_availability,
    validate_city,
    validate_experience_years,
    validate_license,
    validate_name,
    validate_schedule,
)


class TestValidateName:
    def test_valid(self):
        assert validate_name("María García") == "María García"
        assert validate_name("Jean-Pierre O'Brien") == "Jean-Pierre O'Brien"

    def test_collapses_whitespace(self):
        assert validate_name("  Juan   Pérez  ") == "Juan Pérez"

    def test_rejects_single_token(self):
        assert validate_name("Juan") is None

    def test_rejects_too_short(self):
        assert validate_name("J") is None

    def test_rejects_too_long(self):
        assert validate_name("x" * 100) is None

    def test_rejects_digits(self):
        assert validate_name("Juan 2 Pérez") is None


class TestValidateCity:
    def test_canonical_match(self):
        city, country, ok = validate_city("Madrid")
        assert city == "Madrid" and country == "ES" and ok is True

    def test_alias_resolves(self):
        city, country, ok = validate_city("CDMX")
        assert city == "Ciudad de México" and country == "MX" and ok is True

    def test_diacritic_insensitive(self):
        city, _, ok = validate_city("ciudad de mexico")
        assert ok is True and city == "Ciudad de México"

    def test_alias_lowercase(self):
        city, _, ok = validate_city("mexico city")
        assert ok is True and city == "Ciudad de México"

    def test_unknown_city(self):
        city, country, ok = validate_city("Springfield")
        assert ok is False and country is None and city == "Springfield"

    def test_empty(self):
        city, country, ok = validate_city("")
        assert ok is False and city is None


class TestValidateLicense:
    @pytest.mark.parametrize("v,expected", [
        ("yes", True), ("Yes", True), ("sí", True), ("si", True),
        ("Tengo licencia", True), ("I have one", True),
        ("no", False), ("No", False), ("no tengo", False),
        ("not yet", False), ("todavía no", False),
        ("maybe", None), ("idk", None),
    ])
    def test_cases(self, v, expected):
        assert validate_license(v) == expected

    def test_passthrough_bool(self):
        assert validate_license(True) is True
        assert validate_license(False) is False


class TestValidateAvailability:
    @pytest.mark.parametrize("v,expected", [
        ("full time", Availability.full_time),
        ("Tiempo completo", Availability.full_time),
        ("part-time please", Availability.part_time),
        ("Media jornada", Availability.part_time),
        ("solo fines de semana", Availability.weekends_only),
        ("flex", Availability.flexible),
        ("whenever", None),
    ])
    def test_cases(self, v, expected):
        assert validate_availability(v) == expected


class TestValidateSchedule:
    @pytest.mark.parametrize("v,expected", [
        ("morning", Schedule.morning),
        ("Mañana", Schedule.morning),
        ("tarde", Schedule.afternoon),
        ("evening", Schedule.evening),
        ("nights please", Schedule.night),
        ("flexible", Schedule.flexible),
        ("garbage", None),
    ])
    def test_cases(self, v, expected):
        assert validate_schedule(v) == expected


class TestValidateExperience:
    @pytest.mark.parametrize("v,expected", [
        (0, 0), (5, 5), ("3", 3), ("3 years", 3), ("about 2", 2),
        ("none", 0), ("Ninguna", 0), ("zero", 0),
        (-1, None), (50, None), ("nope", None),
        (True, None),  # bool subclass of int but rejected
    ])
    def test_cases(self, v, expected):
        assert validate_experience_years(v) == expected


class TestNormalizePlatforms:
    def test_canonicalizes(self):
        assert normalize_platforms(["glovo", "Uber Eats"]) == ["Glovo", "Uber Eats"]

    def test_dedupes(self):
        assert normalize_platforms(["Glovo", "glovo", "GLOVO"]) == ["Glovo"]

    def test_preserves_unknown(self):
        out = normalize_platforms(["Glovo", "MisteryFood"])
        assert "Glovo" in out and "MisteryFood" in out

    def test_filters_empty(self):
        assert normalize_platforms(["", "Glovo", None]) == ["Glovo"]

"""Tool-dispatch tests — exercise the agent's deterministic core without LLM.

Every state mutation goes through ``dispatch_tool``; that's the contract the
LLM hits, so testing it directly proves the system's correctness independent
of model output.
"""

from hr_agent.agent import dispatch_tool
from hr_agent.schema import Decision, ScreeningState


def _state() -> ScreeningState:
    return ScreeningState(job_id="grupo-sazon/delivery-guy", client_id="grupo-sazon")


# --- record_field ----------------------------------------------------------


def test_record_license_yes(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "has_license", "value": "sí"})
    assert out.output["ok"] is True
    assert state.fields["has_license"] is True
    assert state.disqualifying_field is None


def test_record_license_no_marks_disqualifying(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "has_license", "value": "no"})
    assert out.output["ok"] is True and out.output["disqualifying"] is True
    assert state.fields["has_license"] is False
    assert state.disqualifying_field == "has_license"


def test_record_license_ambiguous_rejected(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "has_license", "value": "maybe"})
    assert out.output["ok"] is False
    assert "has_license" not in state.fields


def test_record_full_name(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "full_name", "value": "  María  García  "})
    assert out.output["ok"] is True
    assert state.fields["full_name"] == "María García"


def test_record_invalid_name_rejected(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "full_name", "value": "X"})
    assert out.output["ok"] is False
    assert "full_name" not in state.fields


def test_record_experience_with_platforms(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(
        state,
        grupo_sazon_spec,
        "record_field",
        {"field": "experience", "value": {"years": 2, "platforms": ["glovo", "Uber Eats"]}},
    )
    assert out.output["ok"] is True
    assert state.fields["experience"] == {"years": 2, "platforms": ["Glovo", "Uber Eats"]}


def test_record_experience_zero_years(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(
        state,
        grupo_sazon_spec,
        "record_field",
        {"field": "experience", "value": {"years": 0, "platforms": []}},
    )
    assert out.output["ok"] is True
    assert state.fields["experience"]["years"] == 0


def test_record_availability(grupo_sazon_spec):
    state = _state()
    dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "availability", "value": "tiempo completo"})
    assert state.fields["availability"] == "full_time"


def test_record_schedule_invalid(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "preferred_schedule", "value": "garbage"})
    assert out.output["ok"] is False


def test_record_unknown_field_rejected(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "shoe_size", "value": "42"})
    assert out.output["ok"] is False


def test_record_language_change(grupo_sazon_spec):
    state = _state()
    state.language = "es"
    dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "language", "value": "en"})
    assert state.language == "en"


def test_stage_index_advances(grupo_sazon_spec):
    state = _state()
    assert state.stage_index == 0
    dispatch_tool(state, grupo_sazon_spec, "record_field", {"field": "has_license", "value": "yes"})
    assert state.stage_index == 1


# --- validate_city ---------------------------------------------------------


def test_validate_city_in_area(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "validate_city", {"city": "Barcelona"})
    assert out.output["in_service_area"] is True
    assert state.fields["city"]["country"] == "ES"
    assert state.disqualifying_field is None


def test_validate_city_out_of_area(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "validate_city", {"city": "Toledo"})
    assert out.output["in_service_area"] is False
    assert out.output["disqualifying"] is True
    assert state.disqualifying_field == "city"


# --- lookup_faq ------------------------------------------------------------


def test_lookup_faq_hit(grupo_sazon_spec):
    state = _state()
    state.language = "es"
    out = dispatch_tool(state, grupo_sazon_spec, "lookup_faq", {"question": "¿cuánto pagan?"})
    assert out.output["match"] == "pay"


def test_lookup_faq_miss(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "lookup_faq", {"question": "what's the meaning of life"})
    assert out.output["match"] is None


# --- complete_screening ----------------------------------------------------


def test_complete_qualified(grupo_sazon_spec):
    state = _state()
    dispatch_tool(state, grupo_sazon_spec, "complete_screening", {"decision": "qualified", "reason": "all good"})
    assert state.decision == Decision.qualified


def test_complete_disqualified_records_field(grupo_sazon_spec):
    state = _state()
    dispatch_tool(
        state,
        grupo_sazon_spec,
        "complete_screening",
        {"decision": "disqualified", "disqualifying_field": "has_license", "reason": "no license"},
    )
    assert state.decision == Decision.disqualified
    assert state.disqualifying_field == "has_license"


def test_complete_invalid_decision(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(
        state, grupo_sazon_spec, "complete_screening", {"decision": "fired", "reason": "lol"}
    )
    assert out.output["ok"] is False


def test_unknown_tool(grupo_sazon_spec):
    state = _state()
    out = dispatch_tool(state, grupo_sazon_spec, "make_coffee", {})
    assert out.output["ok"] is False


# --- full happy-path flow --------------------------------------------------


def test_full_qualified_flow(grupo_sazon_spec):
    """Walk through every tool call a successful screening would make."""
    state = _state()
    spec = grupo_sazon_spec
    dispatch_tool(state, spec, "record_field", {"field": "has_license", "value": "yes"})
    dispatch_tool(state, spec, "validate_city", {"city": "Madrid"})
    dispatch_tool(state, spec, "record_field", {"field": "full_name", "value": "Ana García"})
    dispatch_tool(state, spec, "record_field", {"field": "availability", "value": "full_time"})
    dispatch_tool(state, spec, "record_field", {"field": "preferred_schedule", "value": "morning"})
    dispatch_tool(
        state, spec, "record_field", {"field": "experience", "value": {"years": 4, "platforms": ["Glovo"]}}
    )
    dispatch_tool(state, spec, "record_field", {"field": "start_date", "value": "next monday"})
    dispatch_tool(state, spec, "complete_screening", {"decision": "qualified", "reason": "fits"})

    assert state.decision == Decision.qualified
    assert state.is_complete()
    assert state.stage_index == len([f for f in spec.fields if f.required])

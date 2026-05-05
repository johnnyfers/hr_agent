"""Tool-dispatch tests — exercise the agent's deterministic core without LLM.

The agent's correctness is determined by what tool calls it makes; the wrapper
that maps Anthropic tool_use blocks to state mutations is the part that needs
to be bulletproof. We test that here directly.
"""

from hr_agent.agent import dispatch_tool
from hr_agent.schema import Decision, ScreeningState, Stage


def test_record_license_yes():
    state = ScreeningState()
    out = dispatch_tool(state, "record_field", {"field": "has_license", "value": "sí"})
    assert out.output["ok"] is True
    assert state.has_license is True
    assert state.stage == Stage.license


def test_record_license_no_marks_disqualifying():
    state = ScreeningState()
    out = dispatch_tool(state, "record_field", {"field": "has_license", "value": "no"})
    assert out.output["disqualifying"] is True
    assert state.has_license is False


def test_record_license_ambiguous_rejected():
    state = ScreeningState()
    out = dispatch_tool(state, "record_field", {"field": "has_license", "value": "maybe"})
    assert out.output["ok"] is False
    assert state.has_license is None


def test_validate_city_in_area():
    state = ScreeningState()
    out = dispatch_tool(state, "validate_city", {"city": "Barcelona"})
    assert out.output["in_service_area"] is True
    assert state.city == "Barcelona"
    assert state.country == "ES"


def test_validate_city_out_of_area():
    state = ScreeningState()
    out = dispatch_tool(state, "validate_city", {"city": "Toledo"})
    assert out.output["in_service_area"] is False
    assert out.output["disqualifying"] is True


def test_record_full_name():
    state = ScreeningState()
    out = dispatch_tool(state, "record_field", {"field": "full_name", "value": "  María  García  "})
    assert out.output["value"] == "María García"


def test_record_invalid_name_rejected():
    state = ScreeningState()
    out = dispatch_tool(state, "record_field", {"field": "full_name", "value": "X"})
    assert out.output["ok"] is False
    assert state.full_name is None


def test_record_experience_with_platforms():
    state = ScreeningState()
    out = dispatch_tool(
        state,
        "record_field",
        {"field": "experience", "value": {"years": 2, "platforms": ["glovo", "Uber Eats"]}},
    )
    assert out.output["ok"] is True
    assert state.experience.years == 2
    assert state.experience.platforms == ["Glovo", "Uber Eats"]


def test_record_experience_zero_years():
    state = ScreeningState()
    out = dispatch_tool(
        state, "record_field", {"field": "experience", "value": {"years": 0, "platforms": []}}
    )
    assert out.output["ok"] is True
    assert state.experience.years == 0


def test_record_availability():
    state = ScreeningState()
    out = dispatch_tool(state, "record_field", {"field": "availability", "value": "tiempo completo"})
    assert state.availability.value == "full_time"


def test_record_schedule_invalid():
    state = ScreeningState()
    out = dispatch_tool(state, "record_field", {"field": "preferred_schedule", "value": "garbage"})
    assert out.output["ok"] is False


def test_record_language_change():
    state = ScreeningState(language="es")
    dispatch_tool(state, "record_field", {"field": "language", "value": "en"})
    assert state.language == "en"


def test_lookup_faq_hit():
    state = ScreeningState(language="es")
    out = dispatch_tool(state, "lookup_faq", {"question": "¿cuánto pagan?"})
    assert out.output["match"] == "pay"


def test_lookup_faq_miss():
    state = ScreeningState()
    out = dispatch_tool(state, "lookup_faq", {"question": "what's the meaning of life"})
    assert out.output["match"] is None


def test_complete_screening_qualified():
    state = ScreeningState()
    out = dispatch_tool(
        state, "complete_screening", {"decision": "qualified", "reason": "all good"}
    )
    assert state.decision == Decision.qualified
    assert state.stage == Stage.confirmed


def test_complete_screening_disqualified():
    state = ScreeningState()
    dispatch_tool(
        state,
        "complete_screening",
        {"decision": "disqualified_no_license", "reason": "no license"},
    )
    assert state.decision == Decision.disqualified_no_license


def test_unknown_tool():
    state = ScreeningState()
    out = dispatch_tool(state, "make_coffee", {})
    assert out.output["ok"] is False


def test_full_qualified_flow_via_tools():
    """Simulate a perfect screening run, asserting end state."""
    state = ScreeningState()
    dispatch_tool(state, "record_field", {"field": "has_license", "value": "yes"})
    dispatch_tool(state, "validate_city", {"city": "Madrid"})
    dispatch_tool(state, "record_field", {"field": "full_name", "value": "Ana García"})
    dispatch_tool(state, "record_field", {"field": "availability", "value": "full_time"})
    dispatch_tool(state, "record_field", {"field": "preferred_schedule", "value": "morning"})
    dispatch_tool(
        state, "record_field", {"field": "experience", "value": {"years": 4, "platforms": ["Glovo"]}}
    )
    dispatch_tool(state, "record_field", {"field": "start_date", "value": "next monday"})
    dispatch_tool(state, "complete_screening", {"decision": "qualified", "reason": "fits"})

    assert state.decision == Decision.qualified
    assert state.required_remaining() == []
    assert state.is_complete()

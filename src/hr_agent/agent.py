"""LLM-driven screening agent.

We use Anthropic's tool use to give the model deterministic operations
(record_field, validate_city, lookup_faq, complete_screening). Everything
that affects state goes through a tool — the model never silently mutates
state in prose. This makes the system testable: we replay tool calls and
assert state transitions, independent of natural-language phrasing.

Prompt caching: the system prompt is the bulk of input tokens and is
identical across turns within a conversation. We mark it as a cache
breakpoint to keep per-turn costs low at scale.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from anthropic import Anthropic
from anthropic.types import Message as AnthropicMessage

from . import faq as faq_module
from . import validators
from .guardrails import GuardrailResult, check_user_input
from .prompts import render_system_prompt
from .schema import (
    Availability,
    Conversation,
    Decision,
    Experience,
    Message,
    Schedule,
    ScreeningState,
    Stage,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("HR_AGENT_MODEL", "claude-sonnet-4-6")
MAX_TURN_ITERATIONS = 6  # safety cap on tool-use loop within a single user turn


# --- Tool schemas -----------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "record_field",
        "description": (
            "Save one screening field after the candidate has given a clear answer. "
            "Call this every time you confirm a piece of information — do not batch. "
            "The validator will reject malformed values; you'll see the result and can re-ask."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": [
                        "has_license",
                        "full_name",
                        "availability",
                        "preferred_schedule",
                        "experience",
                        "start_date",
                        "language",
                    ],
                },
                "value": {
                    "description": (
                        "For has_license: bool. For experience: an object {years: int, platforms: [str]}. "
                        "For availability: one of full_time/part_time/weekends_only/flexible. "
                        "For preferred_schedule: one of morning/afternoon/evening/night/flexible. "
                        "For language: 'es' or 'en'. Otherwise free text."
                    ),
                },
            },
            "required": ["field", "value"],
        },
    },
    {
        "name": "validate_city",
        "description": (
            "Check whether a city or zone the candidate mentioned is in our service area. "
            "Call this BEFORE confirming the city. Returns the canonical city name, country, "
            "and whether it's served. If not served, the candidate is disqualified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "lookup_faq",
        "description": (
            "Answer a candidate question about Grupo Sazón (pay, hours, vehicles, documents, process, etc.). "
            "Returns the canonical FAQ answer or null if no good match. If null, tell the candidate "
            "the recruiter will confirm that and continue with the screening — DO NOT make up an answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "complete_screening",
        "description": (
            "End the screening with a final decision. Call this exactly once when: "
            "(a) the candidate has no licence, (b) their city is out of service area, "
            "(c) all fields are collected and confirmed, or (d) the candidate is non-cooperative."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": [
                        "qualified",
                        "disqualified_no_license",
                        "disqualified_out_of_zone",
                        "needs_review",
                    ],
                },
                "reason": {
                    "type": "string",
                    "description": "Brief recruiter-facing reason. 1 sentence.",
                },
            },
            "required": ["decision", "reason"],
        },
    },
]


# --- Tool dispatch ----------------------------------------------------------


@dataclass
class ToolOutcome:
    """Result of executing one tool call. Surfaced back to the model."""

    output: dict[str, Any]
    state_changed: bool = False


def _record_field(state: ScreeningState, field: str, value: Any) -> ToolOutcome:
    if field == "language":
        if value in ("es", "en"):
            state.language = value
            return ToolOutcome({"ok": True, "field": "language", "value": value}, True)
        return ToolOutcome({"ok": False, "error": "language must be 'es' or 'en'"})

    if field == "has_license":
        v = validators.validate_license(value if isinstance(value, str) else str(value))
        if v is None and isinstance(value, bool):
            v = value
        if v is None:
            return ToolOutcome({"ok": False, "error": "ambiguous license value, re-ask"})
        state.has_license = v
        if state.stage == Stage.greet:
            state.stage = Stage.license
        if v is False:
            return ToolOutcome(
                {"ok": True, "field": "has_license", "value": False, "disqualifying": True}, True
            )
        return ToolOutcome({"ok": True, "field": "has_license", "value": True}, True)

    if field == "full_name":
        cleaned = validators.validate_name(str(value))
        if not cleaned:
            return ToolOutcome({"ok": False, "error": "invalid name (need first + last)"})
        state.full_name = cleaned
        state.stage = Stage.name
        return ToolOutcome({"ok": True, "field": "full_name", "value": cleaned}, True)

    if field == "availability":
        v = validators.validate_availability(str(value))
        if v is None:
            return ToolOutcome(
                {
                    "ok": False,
                    "error": "must be one of full_time / part_time / weekends_only / flexible",
                }
            )
        state.availability = v
        state.stage = Stage.availability
        return ToolOutcome({"ok": True, "field": "availability", "value": v.value}, True)

    if field == "preferred_schedule":
        v = validators.validate_schedule(str(value))
        if v is None:
            return ToolOutcome(
                {
                    "ok": False,
                    "error": "must be one of morning / afternoon / evening / night / flexible",
                }
            )
        state.preferred_schedule = v
        state.stage = Stage.schedule
        return ToolOutcome({"ok": True, "field": "preferred_schedule", "value": v.value}, True)

    if field == "experience":
        if not isinstance(value, dict):
            return ToolOutcome({"ok": False, "error": "experience must be an object"})
        years = validators.validate_experience_years(value.get("years"))
        if years is None:
            return ToolOutcome({"ok": False, "error": "invalid years (0-40)"})
        platforms = validators.normalize_platforms(value.get("platforms") or [])
        state.experience = Experience(years=years, platforms=platforms)
        state.stage = Stage.experience
        return ToolOutcome(
            {"ok": True, "field": "experience", "value": {"years": years, "platforms": platforms}},
            True,
        )

    if field == "start_date":
        v = str(value).strip()
        if not v or len(v) > 100:
            return ToolOutcome({"ok": False, "error": "invalid start_date"})
        state.start_date = v
        state.stage = Stage.start_date
        return ToolOutcome({"ok": True, "field": "start_date", "value": v}, True)

    return ToolOutcome({"ok": False, "error": f"unknown field {field}"})


def _validate_city(state: ScreeningState, city: str) -> ToolOutcome:
    canonical, country, in_area = validators.validate_city(city)
    state.city = canonical
    state.country = country
    state.city_in_service_area = in_area
    if state.stage == Stage.license:
        state.stage = Stage.location
    return ToolOutcome(
        {
            "city": canonical,
            "country": country,
            "in_service_area": in_area,
            "disqualifying": not in_area,
        },
        True,
    )


def _lookup_faq(state: ScreeningState, question: str) -> ToolOutcome:
    result = faq_module.search(question, language=state.language)
    if result is None:
        return ToolOutcome({"match": None, "hint": "no FAQ entry — punt to recruiter"})
    return ToolOutcome(
        {"match": result.id, "answer": result.answer, "score": round(result.score, 2)}
    )


def _complete_screening(state: ScreeningState, decision: str, reason: str) -> ToolOutcome:
    try:
        d = Decision(decision)
    except ValueError:
        return ToolOutcome({"ok": False, "error": f"unknown decision {decision}"})
    state.decision = d
    state.decision_reason = reason
    if d == Decision.qualified:
        state.stage = Stage.confirmed
    return ToolOutcome({"ok": True, "decision": d.value}, True)


def dispatch_tool(state: ScreeningState, name: str, arguments: dict) -> ToolOutcome:
    if name == "record_field":
        return _record_field(state, arguments["field"], arguments["value"])
    if name == "validate_city":
        return _validate_city(state, arguments["city"])
    if name == "lookup_faq":
        return _lookup_faq(state, arguments["question"])
    if name == "complete_screening":
        return _complete_screening(state, arguments["decision"], arguments["reason"])
    return ToolOutcome({"ok": False, "error": f"unknown tool {name}"})


# --- Agent runner -----------------------------------------------------------


@dataclass
class AgentTurn:
    """Output of one user→assistant turn."""

    assistant_text: str
    state: ScreeningState
    tool_calls: list[dict]  # for analytics/debugging
    guardrail_flag: Optional[str]


class ScreeningAgent:
    """Runs one Anthropic message loop per user turn, dispatching tools."""

    def __init__(self, client: Optional[Anthropic] = None, model: str = DEFAULT_MODEL):
        self.client = client or Anthropic()
        self.model = model

    def respond(self, conversation: Conversation, user_message: str) -> AgentTurn:
        """Process one user message and return the assistant reply."""
        guard = check_user_input(user_message)
        if not guard.allowed:
            # Empty / rejected message — bounce a generic re-prompt without LLM.
            return AgentTurn(
                assistant_text=_fallback_reprompt(conversation.state.language),
                state=conversation.state,
                tool_calls=[],
                guardrail_flag=guard.flag,
            )

        cleaned = guard.cleaned_input
        if guard.flag:
            log.info("guardrail flagged user message: %s — %s", guard.flag, guard.note)

        # Append the user message to the on-record history before the LLM call.
        conversation.messages.append(Message(role="user", content=cleaned))

        # Build the message list for Anthropic from our history.
        anthropic_messages = _to_anthropic_messages(conversation.messages)

        tool_calls_log: list[dict] = []
        assistant_text = ""

        for iteration in range(MAX_TURN_ITERATIONS):
            system = render_system_prompt(conversation.state, guardrail_flag=guard.flag)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=anthropic_messages,
                tools=TOOLS,
            )

            # Collect any tool calls in this response and execute them.
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_parts = [b.text for b in response.content if b.type == "text"]
            assistant_text = "\n".join(t for t in text_parts if t).strip()

            if not tool_uses:
                # Plain text reply — turn is done.
                break

            # Append the assistant turn (including tool_use blocks) to messages.
            anthropic_messages.append({"role": "assistant", "content": response.content})

            # Dispatch each tool, build tool_result blocks for the next call.
            tool_results = []
            for tu in tool_uses:
                outcome = dispatch_tool(conversation.state, tu.name, tu.input)
                tool_calls_log.append(
                    {"name": tu.name, "input": tu.input, "output": outcome.output}
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(outcome.output, ensure_ascii=False),
                    }
                )

            anthropic_messages.append({"role": "user", "content": tool_results})

            if response.stop_reason == "end_turn":
                break

        if not assistant_text:
            assistant_text = _fallback_reprompt(conversation.state.language)

        conversation.messages.append(Message(role="assistant", content=assistant_text))

        return AgentTurn(
            assistant_text=assistant_text,
            state=conversation.state,
            tool_calls=tool_calls_log,
            guardrail_flag=guard.flag,
        )


def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
    """Convert our stored messages into the Anthropic chat format.

    We persist plain user/assistant text only — tool_use/tool_result blocks
    are ephemeral within a turn loop and don't need to survive turns.
    """
    out = []
    for m in messages:
        if m.role in ("user", "assistant"):
            out.append({"role": m.role, "content": m.content})
    return out


def _fallback_reprompt(language: str) -> str:
    if language == "en":
        return "Sorry — could you say that again?"
    return "Perdona, ¿puedes repetirlo?"


# --- Initial greeting -------------------------------------------------------


def initial_greeting(language: str = "es") -> str:
    """The very first agent message — no LLM call needed."""
    if language == "en":
        return (
            "Hi! I'm Grupo Sazón's screening assistant — quick chat to see if the "
            "delivery driver role is a fit. To start: do you have a valid driver's licence?"
        )
    return (
        "¡Hola! Soy el asistente de Grupo Sazón. Te haré unas preguntas rápidas para "
        "ver si el puesto de repartidor encaja. Para empezar, ¿tienes permiso de "
        "conducir vigente?"
    )


# --- Summary generation -----------------------------------------------------


def generate_summary(conversation: Conversation, client: Optional[Anthropic] = None, model: str = DEFAULT_MODEL) -> str:
    """Produce a recruiter-facing summary using the LLM.

    Falls back to a deterministic summary if the LLM call fails — recruiters
    always get something, even if the API is down.
    """
    state = conversation.state
    deterministic = _deterministic_summary(state)

    if client is None:
        try:
            client = Anthropic()
        except Exception:
            return deterministic

    transcript = "\n".join(
        f"{m.role}: {m.content}" for m in conversation.messages if m.role in ("user", "assistant")
    )[-4000:]

    prompt = f"""You are summarising a delivery-driver screening for a recruiter at Grupo Sazón.

Output strict JSON:
{{
  "headline": "1-line tl;dr (e.g. '5y experienced driver, full-time, Madrid, can start immediately')",
  "highlights": ["3-5 short bullets of positive signals"],
  "concerns": ["any red/yellow flags — empty list if none"]
}}

Decision: {state.decision.value}
Decision reason: {state.decision_reason or '-'}
Collected: {state.model_dump_json(exclude={'decision', 'decision_reason', 'needs_review_fields'})}

Transcript (last 4k chars):
{transcript}
"""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        # Strip markdown code fences if present.
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        lines = [f"**{data.get('headline', '').strip()}**", "", "Highlights:"]
        lines += [f"- {h}" for h in data.get("highlights", [])]
        if data.get("concerns"):
            lines += ["", "Concerns:"]
            lines += [f"- {c}" for c in data["concerns"]]
        return "\n".join(lines)
    except Exception as e:
        log.warning("summary LLM call failed (%s) — using deterministic fallback", e)
        return deterministic


def _deterministic_summary(state: ScreeningState) -> str:
    parts = [f"Decision: **{state.decision.value}**"]
    if state.decision_reason:
        parts.append(f"Reason: {state.decision_reason}")
    if state.full_name:
        parts.append(f"Name: {state.full_name}")
    if state.city:
        parts.append(
            f"Location: {state.city} ({state.country or '?'}) — "
            f"{'in service area' if state.city_in_service_area else 'OUT OF SERVICE AREA'}"
        )
    if state.has_license is not None:
        parts.append(f"Licence: {'yes' if state.has_license else 'no'}")
    if state.availability:
        parts.append(f"Availability: {state.availability.value}")
    if state.preferred_schedule:
        parts.append(f"Schedule: {state.preferred_schedule.value}")
    if state.experience:
        plats = ", ".join(state.experience.platforms) or "no platforms specified"
        parts.append(f"Experience: {state.experience.years}y ({plats})")
    if state.start_date:
        parts.append(f"Start: {state.start_date}")
    return "\n".join(f"- {p}" for p in parts)

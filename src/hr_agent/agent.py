"""LLM-driven screening agent — JobSpec-aware.

Tool definitions and dispatch are derived from a JobSpec each turn:
- ``record_field`` accepts only fields named in the spec
- ``validate_city`` is exposed only when the spec has a city field
- ``lookup_faq`` searches the spec's FAQ
- ``complete_screening`` records a final decision

Every state mutation flows through ``dispatch_tool`` and through
``validate_field`` from validators.py — the LLM proposes, validators dispose.
This is what makes the system testable: we can replay tool calls and assert
state transitions independent of natural-language phrasing.

Prompt caching: the system prompt is the bulk of input tokens and identical
across turns within a conversation; we mark it as a cache breakpoint.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from anthropic import Anthropic
try:
    from anthropic import (
        APITimeoutError,
        APIConnectionError,
        NotFoundError,
        OverloadedError,
        RateLimitError,
    )
except ImportError:  # Older SDK path fallback
    from anthropic._exceptions import (  # type: ignore
        APITimeoutError,
        APIConnectionError,
        NotFoundError,
        OverloadedError,
        RateLimitError,
    )

from . import faq as faq_module
from .guardrails import check_user_input
from .jobspec import JobSpec
from .prompts import render_system_prompt
from .schema import (
    Conversation,
    Decision,
    Message,
    ScreeningState,
)
from .validators import validate_field

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("HR_AGENT_MODEL", "claude-haiku-4-5")
FALLBACK_MODEL = os.environ.get("HR_AGENT_FALLBACK_MODEL", "claude-sonnet-4-6")
MAX_TURN_ITERATIONS = 10  # safety cap on the tool-use loop within a single user turn
MAX_PROVIDER_RETRIES = int(os.environ.get("HR_AGENT_PROVIDER_MAX_RETRIES", "4"))
BACKOFF_BASE_SECONDS = float(os.environ.get("HR_AGENT_PROVIDER_BACKOFF_BASE", "1.0"))
BACKOFF_JITTER_SECONDS = float(os.environ.get("HR_AGENT_PROVIDER_BACKOFF_JITTER", "0.25"))


# --- Tool schemas (dynamic per JobSpec) ------------------------------------


def build_tools(job: JobSpec) -> list[dict[str, Any]]:
    """Build the tool list scoped to ``job``.

    The set of fields the model can record is exactly the spec's fields,
    plus ``language`` for explicit ES/EN switches.
    """
    field_names = [f.name for f in job.fields] + ["language"]
    field_descriptions = "\n".join(
        f"  - {f.name} ({f.type}): {f.label.get('en') or f.label.get('es') or ''}"
        for f in job.fields
    )

    tools: list[dict[str, Any]] = [
        {
            "name": "record_field",
            "description": (
                "Save one screening field after the candidate has given a clear answer. "
                "Call this every time you confirm a piece of information — do not batch. "
                "The validator will reject malformed values; you'll see the result and can re-ask.\n\n"
                f"Fields for this job:\n{field_descriptions}\n  - language (string): 'es' or 'en'"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": field_names},
                    "value": {
                        "description": (
                            "Scalar for bool/int/string/enum/date. "
                            "For type=experience: {years: int, platforms: [str]}. "
                            "For type=city: a single city/zone string."
                        ),
                    },
                },
                "required": ["field", "value"],
            },
        },
        {
            "name": "lookup_faq",
            "description": (
                "Answer a candidate question about the role/company (pay, hours, vehicles, documents, process, etc.). "
                "Returns a canonical answer or null. If null, tell the candidate the recruiter will confirm — "
                "do NOT make up an answer."
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
                "(a) a disqualifying answer is given, (b) all fields are collected and confirmed, "
                "or (c) the candidate is non-cooperative."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["qualified", "disqualified", "needs_review"],
                    },
                    "disqualifying_field": {
                        "type": "string",
                        "description": "Field name that triggered DQ. Required when decision=disqualified.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "1-sentence recruiter-facing reason.",
                    },
                },
                "required": ["decision", "reason"],
            },
        },
    ]

    if job.has_city_field():
        tools.append(
            {
                "name": "validate_city",
                "description": (
                    "Resolve a candidate-typed city to the canonical name and check it against the "
                    "job's service area. Call this BEFORE confirming the city. If the city is not in "
                    "the service area, end the screening with disqualified."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        )

    return tools


# --- Tool dispatch ----------------------------------------------------------


@dataclass
class ToolOutcome:
    output: dict[str, Any]
    state_changed: bool = False


def _record_field(state: ScreeningState, job: JobSpec, name: str, value: Any) -> ToolOutcome:
    if name == "language":
        if value in ("es", "en"):
            state.language = value
            return ToolOutcome({"ok": True, "field": "language", "value": value}, True)
        return ToolOutcome({"ok": False, "error": "language must be 'es' or 'en'"})

    field = job.field(name)
    if field is None:
        return ToolOutcome({"ok": False, "error": f"unknown field {name} for this job"})

    result = validate_field(field, value, job)
    if not result.ok:
        output: dict[str, Any] = {"ok": False, "error": result.error}
        if field.type == "date":
            # Give the model an explicit anchor so it doesn't reason against a
            # different implicit "today" in its own context.
            output["server_today_utc"] = datetime.now(timezone.utc).date().isoformat()
            output["hint"] = "Please provide a future start date."
        return ToolOutcome(output)

    state.fields[name] = result.value

    # Advance stage_index if this completes the next required field in order.
    required = [f for f in job.fields if f.required]
    state.stage_index = sum(1 for f in required if f.name in state.fields)

    response: dict[str, Any] = {"ok": True, "field": name, "value": result.value}
    if result.disqualifying:
        state.disqualifying_field = name
        state.decision_reason = result.reason
        response["disqualifying"] = True
        response["reason"] = result.reason
    return ToolOutcome(response, True)


def _validate_city_tool(state: ScreeningState, job: JobSpec, city: str) -> ToolOutcome:
    """Out-of-band city validation that *also* records the field.

    The agent calls this before record_field to surface the in_service_area
    check; we record at the same time so the model doesn't need two round-trips.
    """
    field = next((f for f in job.fields if f.type == "city"), None)
    if field is None:
        return ToolOutcome({"ok": False, "error": "this job has no city field"})

    result = validate_field(field, city, job)
    if not result.ok:
        return ToolOutcome({"ok": False, "error": result.error})

    state.fields[field.name] = result.value
    rec = result.value or {}
    response: dict[str, Any] = {
        "city": rec.get("canonical") or rec.get("raw"),
        "country": rec.get("country"),
        "in_service_area": rec.get("in_service_area", False),
        "disqualifying": result.disqualifying,
    }
    if result.disqualifying:
        state.disqualifying_field = field.name
        state.decision_reason = result.reason
        response["reason"] = result.reason
    return ToolOutcome(response, True)


def _lookup_faq(state: ScreeningState, job: JobSpec, question: str) -> ToolOutcome:
    result = faq_module.search(question, job.faq, language=state.language)
    if result is None:
        return ToolOutcome({"match": None, "hint": "no FAQ entry — punt to recruiter"})
    return ToolOutcome(
        {"match": result.id, "answer": result.answer, "score": round(result.score, 2)}
    )


def _complete_screening(
    state: ScreeningState,
    job: JobSpec,
    decision: str,
    reason: str,
    disqualifying_field: Optional[str],
) -> ToolOutcome:
    try:
        d = Decision(decision)
    except ValueError:
        return ToolOutcome({"ok": False, "error": f"unknown decision {decision}"})
    if d not in (Decision.qualified, Decision.disqualified, Decision.needs_review):
        return ToolOutcome({"ok": False, "error": f"invalid terminal decision {decision}"})

    if d in (Decision.qualified, Decision.needs_review):
        missing_required = [f.name for f in job.fields if f.required and f.name not in state.fields]
        if missing_required:
            return ToolOutcome(
                {
                    "ok": False,
                    "error": f"cannot complete: missing required fields: {', '.join(missing_required)}",
                    "missing_required_fields": missing_required,
                }
            )

    state.decision = d
    state.decision_reason = reason or state.decision_reason
    if d == Decision.disqualified and disqualifying_field:
        state.disqualifying_field = disqualifying_field
    return ToolOutcome({"ok": True, "decision": d.value}, True)


def dispatch_tool(state: ScreeningState, job: JobSpec, name: str, arguments: dict) -> ToolOutcome:
    if name == "record_field":
        return _record_field(state, job, arguments["field"], arguments["value"])
    if name == "validate_city":
        return _validate_city_tool(state, job, arguments["city"])
    if name == "lookup_faq":
        return _lookup_faq(state, job, arguments["question"])
    if name == "complete_screening":
        return _complete_screening(
            state,
            job,
            arguments.get("decision", ""),
            arguments.get("reason", ""),
            arguments.get("disqualifying_field"),
        )
    return ToolOutcome({"ok": False, "error": f"unknown tool {name}"})


# --- Agent runner -----------------------------------------------------------


@dataclass
class AgentTurn:
    assistant_text: str
    state: ScreeningState
    tool_calls: list[dict]
    guardrail_flag: Optional[str]


class ScreeningAgent:
    """Runs one Anthropic message loop per user turn. Stateless w.r.t. job."""

    def __init__(self, client: Optional[Anthropic] = None, model: str = DEFAULT_MODEL):
        self.client = client or Anthropic()
        self.model = model

    @staticmethod
    def _retryable_provider_error(exc: Exception) -> bool:
        return isinstance(exc, (OverloadedError, RateLimitError, APIConnectionError, APITimeoutError))

    def _retry_model_for_attempt(self, attempt: int) -> str:
        """Alternate between primary model and fallback model on retries."""
        if not FALLBACK_MODEL or FALLBACK_MODEL == self.model:
            return self.model
        return self.model if attempt % 2 == 0 else FALLBACK_MODEL

    def _create_message_with_backoff(self, *, system: list[dict], messages: list[dict], tools: list[dict]):
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_PROVIDER_RETRIES + 1):
            model = self._retry_model_for_attempt(attempt)
            try:
                if attempt > 0:
                    log.info("retrying provider request attempt=%d model=%s", attempt, model)
                return self.client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                    tools=tools,
                )
            except Exception as e:
                # Fallback model may not exist for this account/region.
                # If that happens, stay on primary model for remaining attempts.
                if isinstance(e, NotFoundError) and model == FALLBACK_MODEL:
                    log.warning(
                        "fallback model unavailable (%s); using primary model only",
                        FALLBACK_MODEL,
                    )
                    continue
                last_exc = e
                if not self._retryable_provider_error(e) or attempt >= MAX_PROVIDER_RETRIES:
                    raise
                delay = (BACKOFF_BASE_SECONDS * (2 ** attempt)) + random.uniform(
                    0, BACKOFF_JITTER_SECONDS
                )
                log.warning(
                    "provider request failed (%s), backing off %.2fs before retry",
                    type(e).__name__,
                    delay,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("provider request failed unexpectedly without exception")

    def respond(
        self,
        conversation: Conversation,
        user_message: str,
        job: JobSpec,
    ) -> AgentTurn:
        """Process one user message in the context of ``job``."""
        guard = check_user_input(user_message)
        if not guard.allowed:
            return AgentTurn(
                assistant_text=_fallback_reprompt(conversation.state.language),
                state=conversation.state,
                tool_calls=[],
                guardrail_flag=guard.flag,
            )

        cleaned = guard.cleaned_input
        if guard.flag:
            log.info("guardrail flagged user message: %s — %s", guard.flag, guard.note)

        lang_hint = _detect_language_hint(cleaned)
        if lang_hint and lang_hint != conversation.state.language:
            conversation.state.language = lang_hint

        conversation.messages.append(Message(role="user", content=cleaned))
        anthropic_messages = _to_anthropic_messages(conversation.messages)
        tools = build_tools(job)

        tool_calls_log: list[dict] = []
        assistant_text = ""

        for _ in range(MAX_TURN_ITERATIONS):
            system = render_system_prompt(job, conversation.state, guardrail_flag=guard.flag)
            response = self._create_message_with_backoff(
                system=[
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ],
                messages=anthropic_messages,
                tools=tools,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_parts = [b.text for b in response.content if b.type == "text"]
            assistant_text = "\n".join(t for t in text_parts if t).strip()

            if not tool_uses:
                break

            anthropic_messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                outcome = dispatch_tool(conversation.state, job, tu.name, tu.input)
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
            assistant_text = _state_based_fallback(job, conversation.state)

        conversation.messages.append(Message(role="assistant", content=assistant_text))

        return AgentTurn(
            assistant_text=assistant_text,
            state=conversation.state,
            tool_calls=tool_calls_log,
            guardrail_flag=guard.flag,
        )


def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages if m.role in ("user", "assistant")]


def _detect_language_hint(text: str) -> Optional[str]:
    """Best-effort language hint from latest user message (es/en)."""
    t = text.lower()
    es_markers = {
        "hola",
        "gracias",
        "mañana",
        "manana",
        "lunes",
        "martes",
        "miercoles",
        "jueves",
        "viernes",
        "sabado",
        "domingo",
        "puedo",
        "empezar",
        "si",
        "sí",
        "tengo",
    }
    en_markers = {
        "hello",
        "thanks",
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "i can",
        "i could",
        "start",
        "yes",
        "have",
    }
    es_score = sum(1 for w in es_markers if w in t)
    en_score = sum(1 for w in en_markers if w in t)
    if es_score > en_score:
        return "es"
    if en_score > es_score:
        return "en"
    return None


def _fallback_reprompt(language: str) -> str:
    if language == "en":
        return "Sorry — could you say that again?"
    return "Perdona, ¿puedes repetirlo?"


def _state_based_fallback(job: JobSpec, state: ScreeningState) -> str:
    """Deterministic assistant text when model returns no natural-language text."""
    if state.is_complete():
        if state.language == "en":
            return "Thanks — I've captured everything. A recruiter will follow up soon."
        return "Gracias — ya tengo toda la información. Un reclutador te contactará pronto."

    next_required = next((f for f in job.fields if f.required and f.name not in state.fields), None)
    if next_required:
        hint = next_required.prompt_hint.get(state.language) or next_required.prompt_hint.get("es")
        if hint:
            return hint
    return _fallback_reprompt(state.language)


# --- Initial greeting -------------------------------------------------------


def initial_greeting(job: JobSpec, language: str = "es") -> str:
    """First agent message — no LLM call. Pulled from the JobSpec."""
    title = job.title.get(language) or job.title.get("es") or ""
    first_field = job.fields[0] if job.fields else None
    first_q = ""
    if first_field:
        first_q = first_field.prompt_hint.get(language) or first_field.prompt_hint.get("es") or ""

    if language == "en":
        return (
            f"Hi! I'm {job.client.name}'s screening assistant — quick chat to see if the "
            f"{title} role is a fit. {first_q}"
        ).strip()
    return (
        f"¡Hola! Soy el asistente de {job.client.name}. Te haré unas preguntas rápidas para "
        f"ver si el puesto de {title} encaja. {first_q}"
    ).strip()


# --- Summary generation -----------------------------------------------------


def generate_summary(
    conversation: Conversation,
    job: JobSpec,
    client: Optional[Anthropic] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Recruiter-facing summary. Falls back to deterministic if LLM fails."""
    deterministic = _deterministic_summary(conversation.state, job)

    if client is None:
        try:
            client = Anthropic()
        except Exception:
            return deterministic

    transcript = "\n".join(
        f"{m.role}: {m.content}" for m in conversation.messages if m.role in ("user", "assistant")
    )[-4000:]

    prompt = f"""You are summarising a screening for a recruiter at {job.client.name} ({job.title.get('en') or job.title.get('es')}).

Output strict JSON:
{{
  "headline": "1-line tl;dr",
  "highlights": ["3-5 short bullets of positive signals"],
  "concerns": ["any red/yellow flags — empty list if none"]
}}

Decision: {conversation.state.decision.value}
Decision reason: {conversation.state.decision_reason or '-'}
Collected fields: {json.dumps(conversation.state.fields, ensure_ascii=False)}

Transcript (last 4k chars):
{transcript}
"""
    try:
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_PROVIDER_RETRIES + 1):
            retry_model = model if attempt % 2 == 0 else (FALLBACK_MODEL or model)
            try:
                if attempt > 0:
                    log.info(
                        "retrying summary request attempt=%d model=%s",
                        attempt,
                        retry_model,
                    )
                response = client.messages.create(
                    model=retry_model,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                if isinstance(e, NotFoundError) and retry_model == FALLBACK_MODEL:
                    log.warning(
                        "fallback model unavailable for summary (%s); using primary model only",
                        FALLBACK_MODEL,
                    )
                    continue
                last_exc = e
                is_retryable = isinstance(
                    e, (OverloadedError, RateLimitError, APIConnectionError, APITimeoutError)
                )
                if not is_retryable or attempt >= MAX_PROVIDER_RETRIES:
                    raise
                delay = (BACKOFF_BASE_SECONDS * (2 ** attempt)) + random.uniform(
                    0, BACKOFF_JITTER_SECONDS
                )
                log.warning(
                    "summary request failed (%s), backing off %.2fs before retry",
                    type(e).__name__,
                    delay,
                )
                time.sleep(delay)
        else:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("summary request failed unexpectedly without exception")

        text = "".join(b.text for b in response.content if b.type == "text").strip()
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


def _deterministic_summary(state: ScreeningState, job: JobSpec) -> str:
    parts = [f"Decision: **{state.decision.value}**"]
    if state.decision_reason:
        parts.append(f"Reason: {state.decision_reason}")
    if state.disqualifying_field:
        parts.append(f"DQ field: {state.disqualifying_field}")
    for f in job.fields:
        v = state.fields.get(f.name)
        if v is None:
            continue
        if isinstance(v, dict) and "in_service_area" in v:
            parts.append(
                f"{f.name}: {v.get('canonical') or v.get('raw')} ({v.get('country') or '?'}) — "
                f"{'in service area' if v.get('in_service_area') else 'OUT OF SERVICE AREA'}"
            )
        elif isinstance(v, dict) and "years" in v:
            plats = ", ".join(v.get("platforms") or []) or "no platforms specified"
            parts.append(f"{f.name}: {v['years']}y ({plats})")
        else:
            parts.append(f"{f.name}: {v}")
    return "\n".join(f"- {p}" for p in parts)

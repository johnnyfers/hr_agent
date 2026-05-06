"""Conversation + screening state.

The state shape is now job-agnostic: ``fields`` is a dict keyed by
``FieldSpec.name`` (defined in the JobSpec). The agent and validators read
the JobSpec to know what's required and how to validate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Decision(str, Enum):
    in_progress = "in_progress"
    qualified = "qualified"
    disqualified = "disqualified"
    needs_review = "needs_review"
    dropped_off = "dropped_off"


class ScreeningState(BaseModel):
    """Per-conversation state.

    ``fields`` holds whatever values the JobSpec asked for, validated. The
    agent never reads ``fields`` directly to decide what to do next — it
    re-reads the JobSpec and consults ``fields`` for "is this set yet?".
    """

    job_id: str
    client_id: str
    language: str = "es"

    # Field name → validated value. Shape per type:
    #   bool/string/int/date/enum  → the scalar value
    #   city                        → {"raw": str, "canonical": str|None,
    #                                  "country": str|None,
    #                                  "in_service_area": bool}
    #   experience                  → {"years": int, "platforms": list[str]}
    fields: dict[str, Any] = Field(default_factory=dict)

    decision: Decision = Decision.in_progress
    decision_reason: Optional[str] = None
    disqualifying_field: Optional[str] = None  # name of the field that triggered DQ
    needs_review_fields: list[str] = Field(default_factory=list)

    # Index of the next required field in JobSpec.fields. Drives the
    # drop-off-by-stage analytic and lets the prompt show progress.
    stage_index: int = 0

    def is_complete(self) -> bool:
        return self.decision != Decision.in_progress


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Conversation(BaseModel):
    id: str
    candidate_id: Optional[str] = None
    state: ScreeningState
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: Optional[str] = None

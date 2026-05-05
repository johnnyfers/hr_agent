"""Pydantic models for screening state and persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Availability(str, Enum):
    full_time = "full_time"
    part_time = "part_time"
    weekends_only = "weekends_only"
    flexible = "flexible"


class Schedule(str, Enum):
    morning = "morning"
    afternoon = "afternoon"
    evening = "evening"
    night = "night"
    flexible = "flexible"


class Decision(str, Enum):
    in_progress = "in_progress"
    qualified = "qualified"
    disqualified_no_license = "disqualified_no_license"
    disqualified_out_of_zone = "disqualified_out_of_zone"
    needs_review = "needs_review"
    dropped_off = "dropped_off"


class Stage(str, Enum):
    """Tracks the highest-numbered stage reached, for analytics drop-off."""

    greet = "0_greet"
    license = "1_license"
    location = "2_location"
    name = "3_name"
    availability = "4_availability"
    schedule = "5_schedule"
    experience = "6_experience"
    start_date = "7_start_date"
    confirmed = "8_confirmed"


class Experience(BaseModel):
    years: int = Field(ge=0, le=40)
    platforms: list[str] = Field(default_factory=list)


class ScreeningState(BaseModel):
    """The structured data we extract from the conversation.

    All fields are optional until set. Disqualifying fields short-circuit
    further collection; the agent stops asking and moves to closure.
    """

    full_name: Optional[str] = None
    has_license: Optional[bool] = None
    city: Optional[str] = None
    country: Optional[Literal["ES", "MX"]] = None
    city_in_service_area: Optional[bool] = None
    availability: Optional[Availability] = None
    preferred_schedule: Optional[Schedule] = None
    experience: Optional[Experience] = None
    start_date: Optional[str] = None  # free-text, normalized later

    decision: Decision = Decision.in_progress
    decision_reason: Optional[str] = None
    needs_review_fields: list[str] = Field(default_factory=list)
    stage: Stage = Stage.greet
    language: Literal["es", "en"] = "es"

    def required_remaining(self) -> list[str]:
        """Names of required fields still unset, in stage order."""
        order = [
            ("has_license", self.has_license is None),
            ("city", self.city is None),
            ("full_name", self.full_name is None),
            ("availability", self.availability is None),
            ("preferred_schedule", self.preferred_schedule is None),
            ("experience", self.experience is None),
            ("start_date", self.start_date is None),
        ]
        return [name for name, missing in order if missing]

    def is_complete(self) -> bool:
        return self.decision != Decision.in_progress or not self.required_remaining()


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Conversation(BaseModel):
    id: str
    candidate_id: Optional[str] = None
    state: ScreeningState = Field(default_factory=ScreeningState)
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: Optional[str] = None


class ConversationSummary(BaseModel):
    """What we hand to the recruiter."""

    conversation_id: str
    decision: Decision
    decision_reason: Optional[str]
    candidate: ScreeningState
    highlights: list[str]
    concerns: list[str]
    created_at: datetime

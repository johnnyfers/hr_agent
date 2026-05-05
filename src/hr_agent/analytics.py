"""Conversation analytics.

Reads from storage; never mutates. The metrics here are the ones a hiring
manager actually asks about: completion rate, where candidates drop off,
average duration, qualified-vs-disqualified mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .schema import Decision, ScreeningState, Stage
from .storage import Storage


@dataclass
class FunnelMetrics:
    total: int = 0
    qualified: int = 0
    disqualified_no_license: int = 0
    disqualified_out_of_zone: int = 0
    needs_review: int = 0
    dropped_off: int = 0
    in_progress: int = 0

    completion_rate: float = 0.0
    qualification_rate: float = 0.0  # qualified / completed (excludes in_progress)
    drop_off_by_stage: dict[str, int] = field(default_factory=dict)
    avg_messages_per_completed: float = 0.0
    avg_duration_seconds_completed: float = 0.0


def compute(storage: Storage, since: Optional[datetime] = None, limit: int = 1000) -> FunnelMetrics:
    rows = storage.list_conversations(limit=limit)
    metrics = FunnelMetrics()

    durations: list[float] = []
    message_counts: list[int] = []

    for row in rows:
        if since:
            updated = datetime.fromisoformat(row["updated_at"])
            if updated < since:
                continue

        metrics.total += 1
        decision = Decision(row["decision"])

        # Bucket — match attribute name to Decision value.
        attr = decision.value
        if hasattr(metrics, attr):
            setattr(metrics, attr, getattr(metrics, attr) + 1)

        # Drop-off-by-stage from the saved state.
        state = ScreeningState.model_validate_json(row["state_json"])
        if decision in (Decision.in_progress, Decision.dropped_off):
            metrics.drop_off_by_stage[state.stage.value] = (
                metrics.drop_off_by_stage.get(state.stage.value, 0) + 1
            )

        # For completed (qualified or any disqualified terminal), gather perf.
        if decision != Decision.in_progress and decision != Decision.dropped_off:
            conv = storage.get_conversation(row["id"])
            if conv:
                message_counts.append(len(conv.messages))
                if conv.messages:
                    duration = (
                        conv.messages[-1].timestamp - conv.messages[0].timestamp
                    ).total_seconds()
                    durations.append(duration)

    completed = (
        metrics.qualified
        + metrics.disqualified_no_license
        + metrics.disqualified_out_of_zone
        + metrics.needs_review
    )
    if metrics.total:
        metrics.completion_rate = round(completed / metrics.total, 3)
    if completed:
        metrics.qualification_rate = round(metrics.qualified / completed, 3)
    if message_counts:
        metrics.avg_messages_per_completed = round(sum(message_counts) / len(message_counts), 1)
    if durations:
        metrics.avg_duration_seconds_completed = round(sum(durations) / len(durations), 0)

    return metrics


def stage_drop_off_table(metrics: FunnelMetrics) -> list[tuple[str, int]]:
    """Return drop-off counts ordered by stage progression."""
    order = [s.value for s in Stage]
    counts = metrics.drop_off_by_stage
    return [(s, counts.get(s, 0)) for s in order if counts.get(s, 0) > 0]

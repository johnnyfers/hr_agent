"""Conversation analytics.

Reads from storage; never mutates. Metrics a hiring manager actually asks
about: completion rate, where candidates drop off, average duration,
qualified-vs-disqualified mix. Per-job scoping via ``job_id``.

Drop-off "stage" is the index of the next required field that wasn't
collected — derived from the saved ScreeningState.stage_index. The stage
labels are the field names from whichever job the conversation belongs to,
so the numbers are meaningful even across heterogeneous jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .schema import Decision, ScreeningState
from .storage import Storage


@dataclass
class FunnelMetrics:
    total: int = 0
    qualified: int = 0
    disqualified: int = 0
    needs_review: int = 0
    dropped_off: int = 0
    in_progress: int = 0

    completion_rate: float = 0.0  # completed / total
    qualification_rate: float = 0.0  # qualified / completed (excludes in_progress)
    drop_off_by_stage: dict[str, int] = field(default_factory=dict)  # "stage_<i>:<job_id>" → count
    avg_messages_per_completed: float = 0.0
    avg_duration_seconds_completed: float = 0.0


def compute(
    storage: Storage,
    since: Optional[datetime] = None,
    job_id: Optional[str] = None,
    limit: int = 1000,
) -> FunnelMetrics:
    rows = storage.list_conversations(job_id=job_id, limit=limit)
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

        if hasattr(metrics, decision.value):
            setattr(metrics, decision.value, getattr(metrics, decision.value) + 1)

        state = ScreeningState.model_validate_json(row["state_json"])
        if decision in (Decision.in_progress, Decision.dropped_off):
            key = f"stage_{state.stage_index}:{state.job_id}"
            metrics.drop_off_by_stage[key] = metrics.drop_off_by_stage.get(key, 0) + 1

        if decision not in (Decision.in_progress, Decision.dropped_off):
            conv = storage.get_conversation(row["id"])
            if conv:
                message_counts.append(len(conv.messages))
                if conv.messages:
                    duration = (
                        conv.messages[-1].timestamp - conv.messages[0].timestamp
                    ).total_seconds()
                    durations.append(duration)

    completed = metrics.qualified + metrics.disqualified + metrics.needs_review
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
    """Return drop-off counts ordered by stage index then job."""
    items = list(metrics.drop_off_by_stage.items())

    def sort_key(item: tuple[str, int]):
        key = item[0]
        try:
            stage_str, _job = key.split(":", 1)
            return (int(stage_str.split("_", 1)[1]), key)
        except (ValueError, IndexError):
            return (10**9, key)

    items.sort(key=sort_key)
    return items

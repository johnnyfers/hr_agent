from datetime import datetime, timedelta, timezone

from hr_agent import analytics
from hr_agent.schema import Conversation, Decision, Message, ScreeningState, Stage


def _conv(id_, decision, stage=Stage.greet, msg_count=4, duration_s=60):
    state = ScreeningState(decision=decision, stage=stage)
    base = datetime.now(timezone.utc)
    messages = [
        Message(role="user" if i % 2 else "assistant", content=f"m{i}", timestamp=base + timedelta(seconds=i * (duration_s / max(msg_count - 1, 1))))
        for i in range(msg_count)
    ]
    return Conversation(id=id_, state=state, messages=messages)


def test_compute_basic_funnel(tmp_storage):
    storage = tmp_storage
    storage.upsert_conversation(_conv("a", Decision.qualified))
    for m in storage.get_conversation("a").messages:
        storage.append_message("a", m)
    storage.upsert_conversation(_conv("b", Decision.disqualified_no_license))
    for m in storage.get_conversation("b").messages:
        storage.append_message("b", m)
    storage.upsert_conversation(_conv("c", Decision.in_progress, stage=Stage.location))
    for m in storage.get_conversation("c").messages:
        storage.append_message("c", m)

    m = analytics.compute(storage)
    assert m.total == 3
    assert m.qualified == 1
    assert m.disqualified_no_license == 1
    assert m.in_progress == 1
    assert m.completion_rate == round(2 / 3, 3)
    assert m.qualification_rate == 0.5  # 1 qualified / 2 completed


def test_drop_off_by_stage(tmp_storage):
    storage = tmp_storage
    storage.upsert_conversation(_conv("a", Decision.in_progress, stage=Stage.location))
    storage.upsert_conversation(_conv("b", Decision.in_progress, stage=Stage.availability))
    storage.upsert_conversation(_conv("c", Decision.in_progress, stage=Stage.location))

    m = analytics.compute(storage)
    table = analytics.stage_drop_off_table(m)
    rendered = dict(table)
    assert rendered.get("2_location") == 2
    assert rendered.get("4_availability") == 1


def test_empty_storage(tmp_storage):
    m = analytics.compute(tmp_storage)
    assert m.total == 0
    assert m.completion_rate == 0.0

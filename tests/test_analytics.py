from datetime import datetime, timedelta, timezone

from hr_agent import analytics
from hr_agent.jobspec import Client
from hr_agent.schema import Conversation, Decision, Message, ScreeningState


JOB_ID = "grupo-sazon/delivery-guy"
CLIENT_ID = "grupo-sazon"


def _conv(id_, decision, stage_index=0, msg_count=4, duration_s=60, job_id=JOB_ID):
    state = ScreeningState(
        job_id=job_id,
        client_id=CLIENT_ID,
        decision=decision,
        stage_index=stage_index,
    )
    base = datetime.now(timezone.utc)
    step = duration_s / max(msg_count - 1, 1)
    messages = [
        Message(
            role="user" if i % 2 else "assistant",
            content=f"m{i}",
            timestamp=base + timedelta(seconds=i * step),
        )
        for i in range(msg_count)
    ]
    return Conversation(id=id_, state=state, messages=messages)


def _save(storage, conv):
    storage.upsert_conversation(conv)
    for m in conv.messages:
        storage.append_message(conv.id, m)


def test_compute_basic_funnel(tmp_storage):
    _save(tmp_storage, _conv("a", Decision.qualified))
    _save(tmp_storage, _conv("b", Decision.disqualified))
    _save(tmp_storage, _conv("c", Decision.in_progress, stage_index=2))

    m = analytics.compute(tmp_storage)
    assert m.total == 3
    assert m.qualified == 1
    assert m.disqualified == 1
    assert m.in_progress == 1
    assert m.completion_rate == round(2 / 3, 3)
    assert m.qualification_rate == 0.5  # 1 qualified / 2 completed


def test_drop_off_by_stage(tmp_storage):
    _save(tmp_storage, _conv("a", Decision.in_progress, stage_index=2))
    _save(tmp_storage, _conv("b", Decision.in_progress, stage_index=4))
    _save(tmp_storage, _conv("c", Decision.in_progress, stage_index=2))

    m = analytics.compute(tmp_storage)
    table = analytics.stage_drop_off_table(m)
    rendered = dict(table)
    assert rendered.get(f"stage_2:{JOB_ID}") == 2
    assert rendered.get(f"stage_4:{JOB_ID}") == 1


def test_segments_by_job(tmp_storage):
    """job_id filter scopes the funnel to one job."""
    # Add a second job/client in storage so we can prove scoping.
    tmp_storage.upsert_client(Client(id="acme", name="Acme"))
    other = tmp_storage.get_job(JOB_ID).model_copy(update={
        "job_id": "acme/courier",
        "client": Client(id="acme", name="Acme"),
    })
    tmp_storage.upsert_job(other)

    _save(tmp_storage, _conv("a", Decision.qualified))
    _save(tmp_storage, _conv("b", Decision.qualified, job_id="acme/courier"))

    sazon = analytics.compute(tmp_storage, job_id=JOB_ID)
    acme = analytics.compute(tmp_storage, job_id="acme/courier")
    assert sazon.total == 1 and sazon.qualified == 1
    assert acme.total == 1 and acme.qualified == 1


def test_empty_storage(tmp_storage):
    m = analytics.compute(tmp_storage)
    assert m.total == 0
    assert m.completion_rate == 0.0

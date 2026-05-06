"""Storage round-trip tests.

The fixture ``tmp_storage`` is pre-seeded with the default jobs, so we
have valid foreign-key targets for the conversation rows.
"""

from hr_agent.jobspec import Client
from hr_agent.schema import Conversation, Decision, Message, ScreeningState


JOB_ID = "grupo-sazon/delivery-guy"
CLIENT_ID = "grupo-sazon"


def _state(**overrides) -> ScreeningState:
    base = dict(job_id=JOB_ID, client_id=CLIENT_ID)
    base.update(overrides)
    return ScreeningState(**base)


def test_seed_inserts_job_and_client(tmp_storage):
    clients = tmp_storage.list_clients()
    jobs = tmp_storage.list_jobs()
    assert any(c.id == CLIENT_ID for c in clients)
    assert any(j.job_id == JOB_ID for j in jobs)


def test_get_job_returns_full_spec(tmp_storage):
    job = tmp_storage.get_job(JOB_ID)
    assert job is not None
    assert job.client.name == "Grupo Sazón"
    assert any(f.name == "has_license" for f in job.fields)
    assert job.service_areas is not None
    assert "Madrid" in job.service_areas.countries["ES"]


def test_upsert_conversation_round_trip(tmp_storage):
    state = _state(
        fields={
            "has_license": True,
            "city": {"raw": "Madrid", "canonical": "Madrid", "country": "ES", "in_service_area": True},
            "full_name": "Ana López",
        }
    )
    conv = Conversation(id="c1", state=state)
    tmp_storage.upsert_conversation(conv)
    tmp_storage.append_message("c1", Message(role="user", content="hola"))
    tmp_storage.append_message("c1", Message(role="assistant", content="¡hola!"))

    fetched = tmp_storage.get_conversation("c1")
    assert fetched is not None
    assert fetched.state.fields["full_name"] == "Ana López"
    assert fetched.state.fields["has_license"] is True
    assert len(fetched.messages) == 2


def test_upsert_idempotent(tmp_storage):
    conv = Conversation(id="c1", state=_state())
    tmp_storage.upsert_conversation(conv)
    conv.state.fields["full_name"] = "Updated Name"
    tmp_storage.upsert_conversation(conv)
    fetched = tmp_storage.get_conversation("c1")
    assert fetched.state.fields["full_name"] == "Updated Name"


def test_list_filtered_by_decision(tmp_storage):
    for i, decision in enumerate([Decision.qualified, Decision.qualified, Decision.disqualified]):
        conv = Conversation(id=f"c{i}", state=_state(decision=decision))
        tmp_storage.upsert_conversation(conv)
    qualified = tmp_storage.list_conversations(decision=Decision.qualified)
    assert len(qualified) == 2


def test_list_filtered_by_job(tmp_storage):
    # Add a second client+job so we can verify scoping.
    tmp_storage.upsert_client(Client(id="acme", name="Acme"))
    other_job = tmp_storage.get_job(JOB_ID).model_copy(update={
        "job_id": "acme/courier",
        "client": Client(id="acme", name="Acme"),
    })
    tmp_storage.upsert_job(other_job)

    tmp_storage.upsert_conversation(Conversation(id="c1", state=_state()))
    tmp_storage.upsert_conversation(
        Conversation(id="c2", state=_state(job_id="acme/courier", client_id="acme"))
    )

    sazon_only = tmp_storage.list_conversations(job_id=JOB_ID)
    assert {r["id"] for r in sazon_only} == {"c1"}
    acme_only = tmp_storage.list_conversations(job_id="acme/courier")
    assert {r["id"] for r in acme_only} == {"c2"}


def test_get_missing_returns_none(tmp_storage):
    assert tmp_storage.get_conversation("nope") is None
    assert tmp_storage.get_job("nope/nope") is None

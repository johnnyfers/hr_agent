"""Storage round-trip tests."""

from hr_agent.schema import Availability, Conversation, Decision, Experience, Message, ScreeningState


def test_upsert_and_fetch(tmp_storage):
    state = ScreeningState(
        full_name="Ana López",
        has_license=True,
        city="Madrid",
        country="ES",
        city_in_service_area=True,
        availability=Availability.full_time,
        experience=Experience(years=3, platforms=["Glovo"]),
    )
    conv = Conversation(id="c1", state=state)
    tmp_storage.upsert_conversation(conv)
    tmp_storage.append_message("c1", Message(role="user", content="hola"))
    tmp_storage.append_message("c1", Message(role="assistant", content="¡hola!"))

    fetched = tmp_storage.get_conversation("c1")
    assert fetched is not None
    assert fetched.state.full_name == "Ana López"
    assert fetched.state.experience.years == 3
    assert len(fetched.messages) == 2


def test_upsert_idempotent(tmp_storage):
    conv = Conversation(id="c1")
    tmp_storage.upsert_conversation(conv)
    conv.state.full_name = "Updated"
    tmp_storage.upsert_conversation(conv)
    assert tmp_storage.get_conversation("c1").state.full_name == "Updated"


def test_list_filtered_by_decision(tmp_storage):
    for i, decision in enumerate([Decision.qualified, Decision.qualified, Decision.disqualified_no_license]):
        conv = Conversation(id=f"c{i}", state=ScreeningState(decision=decision))
        tmp_storage.upsert_conversation(conv)

    qualified = tmp_storage.list_conversations(decision=Decision.qualified)
    assert len(qualified) == 2


def test_export_json(tmp_storage, tmp_path):
    conv = Conversation(id="c1", state=ScreeningState(full_name="Test User"))
    tmp_storage.upsert_conversation(conv)
    tmp_storage.append_message("c1", Message(role="user", content="hi"))

    path = tmp_storage.export_json("c1", out_dir=str(tmp_path))
    assert path.exists()
    data = path.read_text()
    assert "Test User" in data
    assert "hi" in data


def test_get_missing_returns_none(tmp_storage):
    assert tmp_storage.get_conversation("nope") is None

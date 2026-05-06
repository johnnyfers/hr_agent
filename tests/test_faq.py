"""FAQ retrieval tests against the seeded Grupo Sazón JobSpec."""

from hr_agent import faq


def test_pay_question_es(grupo_sazon_spec):
    r = faq.search("¿cuánto pagan?", grupo_sazon_spec.faq, language="es")
    assert r is not None and r.id == "pay"


def test_pay_question_en(grupo_sazon_spec):
    r = faq.search("how much do you pay per hour", grupo_sazon_spec.faq, language="en")
    assert r is not None and r.id == "pay"


def test_vehicle_question(grupo_sazon_spec):
    r = faq.search("¿necesito moto propia?", grupo_sazon_spec.faq, language="es")
    assert r is not None and r.id == "vehicle"


def test_unrelated_returns_none(grupo_sazon_spec):
    r = faq.search("what's the weather like in tokyo", grupo_sazon_spec.faq)
    assert r is None


def test_empty_query(grupo_sazon_spec):
    assert faq.search("", grupo_sazon_spec.faq) is None


def test_threshold_filters_weak_match(grupo_sazon_spec):
    r = faq.search("a", grupo_sazon_spec.faq, min_score=0.5)
    assert r is None


def test_empty_faq_list():
    assert faq.search("anything", [], language="es") is None

from hr_agent import faq


def test_pay_question_es():
    r = faq.search("¿cuánto pagan?", language="es")
    assert r is not None and r.id == "pay"


def test_pay_question_en():
    r = faq.search("how much do you pay per hour", language="en")
    assert r is not None and r.id == "pay"


def test_vehicle_question():
    r = faq.search("¿necesito moto propia?", language="es")
    assert r is not None and r.id == "vehicle"


def test_unrelated_returns_none():
    r = faq.search("what's the weather like in tokyo")
    assert r is None


def test_empty_query():
    assert faq.search("") is None


def test_threshold_filters_weak_match():
    r = faq.search("a", min_score=0.5)
    assert r is None

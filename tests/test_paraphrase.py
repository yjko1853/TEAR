from tear.data import paraphrase as para


def _mentions(text):
    names = {"france": "France", "olugbenga ashiru": "Olugbenga Ashiru"}
    t = text.lower()
    return [v for k, v in names.items() if k in t]


def test_population_excludes_dates_and_months():
    base = {"qtype": "after_first"}
    assert para.in_population(dict(base, question="After Olugbenga Ashiru, who was the first to praise France?"))
    assert not para.in_population(dict(base, question="After 2006-01-05, who was the first to visit Iraq?"))
    assert not para.in_population(dict(base, question="Who visited Iraq first after March?"))
    assert not para.in_population({"qtype": "equal", "question": "Who visited Iraq?"})


def test_draw_sample_is_deterministic_and_stratified():
    qs = [{"quid": i, "qtype": t, "question": f"Before X{i}, who was last?"}
          for i, t in enumerate(para.MULTI * 10)]
    a, b = para.draw_sample(qs, 3, 5), para.draw_sample(qs, 3, 5)
    assert a == b and len(a) == 9 and len(set(a)) == 9


def test_clean_output():
    assert para.clean_output('Rewritten question: "Who praised France first?"\nextra') == "Who praised France first?"
    assert para.clean_output("") == ""


def test_check_filter():
    orig = "After Olugbenga Ashiru, who was the first to praise France?"
    ok = para.check(orig, "Who praised France first once Olugbenga Ashiru had done so?", _mentions)
    assert ok["valid"]
    lost = para.check(orig, "Who praised the French Republic first after Olugbenga Ashiru?", _mentions)
    assert not lost["valid"] and lost["lost_entities"] == ["France"]
    assert "unchanged" in para.check(orig, orig.upper(), _mentions)["reasons"]
    assert "format" in para.check(orig, "Who praised France first", _mentions)["reasons"]

from tear.text import exact_match, lenient_match, match, normalize


def test_normalize_drops_case_punctuation_articles():
    assert normalize("The Ministry (Brazil)!") == "ministry brazil"
    assert normalize("  a  Citizen  ") == "citizen"


def test_match_is_substring_of_prediction():
    assert match("Citizen (North Korea)", "North Korea")
    assert not match("North Korea", "Citizen (North Korea)")


def test_exact_vs_lenient():
    assert exact_match("Jack Straw", ["jack straw", "Tony Blair"])
    assert not exact_match("2012-05-17", ["2012-05"])
    assert lenient_match("2012-05-17", ["2012-05"])

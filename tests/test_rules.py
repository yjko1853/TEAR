import pytest

from tear.labeling.rules import build_mentioner, build_sector_matcher, main_clause, rule_slot, rule_topic

# Hand-judged cases (grammatical answer slot) collected while designing and validating the rules.
CASES = [
    ("After 2015-05-12, who would be the first to want to negotiate with the United States' Cabinet Council of Advisors?", "subject"),
    ("Who would want to negotiate with Angela Merkel in the same month as 2014-09-17 Kuwait?", "subject"),
    ("Could you tell me the exact month when Malaysia used conventional military force against the criminal of South Korea?", "timestamp"),
    ("Could you tell me the specific month when did China last blame Chuck Hagel?", "timestamp"),
    ("Which was the last country Kuwait made negative comments about?", "object"),
    ("Who was the first country that Ethiopia expressed optimism about?", "object"),
    ("Which country was first praised by Iraq after the militant of Syria on 2007-02-04?", "object"),
    ("Before Cambodia, with whom did Catherine Ashton last wish to meet?", "object"),
    ("Who was the last country to host a visit from Malaysia after 2014-03-17?", "subject"),
    ("Which was the last country to sign an agreement with Swaziland?", "subject"),
    ("To which country did Iraq first offer cooperation?", "object"),
    ("Who made Thailand suffer from unconventional violence for the first time?", "subject"),
    ("In the same month as the African government, who did China wish to engage in diplomatic cooperation with in 2007-02-06?", "object"),
    ("After 2011-07-07, who was the first to visit China?", "subject"),
    ("In which month did China accuse the envoy of the United States?", "timestamp"),
    ("Before Alan Peter Cayetano, who had a telephone conversation with the Inquirer?", "subject"),
    ("After 12 May 2011, which country did Cambodia denounce?", "object"),
    ("When did Muhammad Naji al-Otari praise Kuwait?", "timestamp"),
]


@pytest.mark.parametrize("question,slot", CASES)
def test_rule_slot(question, slot):
    assert rule_slot(question) == slot


def test_main_clause_strips_leading_constraints():
    assert main_clause("After 2006-01-05, who was the first to visit Iraq?") == "who was the first to visit Iraq?"
    assert main_clause("In 2014, before Japan, who blamed China?") == "who blamed China?"


ENTS = ["Iraq", "Japan", "China", "Catherine Ashton", "Cambodia", "Men (Argentina)", "Citizen (Nigeria)"]


def test_rule_topic_main_clause_and_sector_names():
    mentions, sector = build_mentioner(ENTS), build_sector_matcher(ENTS)
    assert rule_topic("Before Cambodia, with whom did Catherine Ashton last wish to meet?", mentions, None, sector) == \
        ("Catherine Ashton", "rule_main_clause")
    assert rule_topic("Who did the men of Argentina use unconventional force against in the same year as China?",
                      mentions, None, sector) == ("Men (Argentina)", "rule_main_clause")
    assert rule_topic("Who was the first to visit Iraq, after Japan?", mentions, None, sector)[0] == "Iraq"


def test_rule_topic_fallbacks():
    mentions, sector = build_mentioner(ENTS), build_sector_matcher(ENTS)
    assert rule_topic("Who denounced the Lebanese military?", mentions, "Iraq", sector) == ("Iraq", "ner_fallback")
    assert rule_topic("Who denounced the Lebanese military?", mentions, None, sector) == ("", "none")


# Rule v2: cases from the human check of the labels, then a fixed random sample of questions whose slot changed
# between v1 and v2 (seed 31), each judged by reading the question. v1 gets every one of them wrong.
V2_CASES = [
    ("After 2015-10-14, which country first expressed its intention to negotiate with China after the citizens of Tanzania did so?", "subject"),
    ("In which country did the villager of Thailand criticize or denounce in the same month of Thailand?", "object"),
    ("Which was the last country with which Cambodia formally signed an agreement?", "object"),
    ("Who was the last to want to meet the military of Thailand, before Cambodia did 2008-08-05?", "subject"),
    ("After 2005-12-29, which country first received a visit from Ethiopia after Abu Bakr Abdullah al-Qirbi did?", "subject"),
    ("Who investigated Iraq before the Energy Department/Ministry of Kuwait did?", "subject"),
    ("Who receive the visit from María Ángela Holguín after Cambodia did?", "subject"),
    ("Who formally signed an agreement with Burundi before the rebels of Libération did?", "subject"),
    ("After 2015-09-04, which country expressed its intention to cooperate with Thailand first, after the company - owner or operator of Thailand did?", "subject"),
    ("After 2014-12-01, what was the first country with which Iraq sought diplomatic cooperation?", "object"),
    ("Before 2006-11-27, which country wished to engage in diplomatic cooperation with China last before Joon Young Woo did?", "subject"),
    ("Before 2009-03-14, who was the last country with which the Cabinet of Ministers of the United States expressed interest in cooperation?", "object"),
    ("Which country receive Thongsing Thammavong's visit from Thongsing Thammavong before China did?", "subject"),
    ("Who fought Malaysia with small arms and light weapons after the Malaysian police did?", "subject"),
    ("Who condemned the Philippine military after the Inquirer did?", "subject"),
    ("Before 2005-06-15, which was the last country with which China expressed an intention to meet?", "object"),
    ("After 2012-01-28, which country was the last to accuse Ethiopia before Afar Revolutionary Democratic Unity Front did?", "subject"),
    # second blind check: (d) date with a comma in the leading clause, (e) 'suffer from' asks for the victim
    ("After Sep, 2011, which country did Thailand negotiate with?", "object"),
    ("After Nov, 2012, with whom did the Inquirer have a telephone conversation?", "object"),
    ("Which country suffered from Thailand's conventional military forces before 2005-07?", "object"),
    ("Before 2006-05-07, who was the last to suffer from the conventional military forces of the Israeli police?", "object"),
]


@pytest.mark.parametrize("question,slot", V2_CASES)
def test_rule_slot_v2_fixes(question, slot):
    assert rule_slot(question) == slot              # default version is v2
    assert rule_slot(question, "v1") != slot        # the v1 error this case documents


def test_rule_v2_keeps_passive_and_unknown_version():
    assert rule_slot("Before 2012-06-06, which country was rejected by China last?") == "object"
    with pytest.raises(ValueError):
        rule_slot("Who visited Iraq?", "v3")


def test_rule_v2_keeps_correct_v1_answers():
    # the date-comma fix must not change questions v1 already got right; causative 'made X suffer' asks for the attacker
    assert rule_slot("In Aug, 2006, who wanted to negotiate with Kuwait?") == "subject"
    assert rule_slot("Who made Thailand suffer from unconventional violence for the first time?") == "subject"
    assert main_clause("After Sep, 2011, which country did Thailand negotiate with?") == "which country did Thailand negotiate with?"
    assert main_clause("After Sep, 2011, which country did Thailand negotiate with?", "v1").startswith("2011,")


def test_fix_sets_bracket_the_versions():
    from tear.labeling.rules import FIXES, rule_slot_fixes
    q = "After Sep, 2011, which country did Thailand negotiate with?"
    assert rule_slot_fixes(q, FIXES["v1"]) == rule_slot(q, "v1") == "subject"
    assert rule_slot_fixes(q, frozenset("abc")) == "subject"          # needs fix (d)
    assert rule_slot_fixes(q, frozenset("abcd")) == rule_slot(q) == "object"

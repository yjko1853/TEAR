from tear.data.multitq import add_query_field, query_text, rewrite_is_valid, stage_query

Q = "After the Danish Ministry of Defence and Security, who was the first to visit Iraq?"


def test_rewrite_validity():
    assert rewrite_is_valid("After 2006-01-05, who was the first to visit Iraq?", Q)
    assert not rewrite_is_valid("", Q) and not rewrite_is_valid("   ", Q) and not rewrite_is_valid(None, Q)
    assert not rewrite_is_valid("After 2012-11-21, who was", Q)              # truncated: no '?', < 60% of the words
    assert rewrite_is_valid("After 2006-01-05, who was the first to visit Iraq", Q)   # long enough without '?'


def test_query_text_and_stage_query():
    good = {"question": Q, "rewrite": "After 2006-01-05, who was the first to visit Iraq?"}
    empty = {"question": Q, "rewrite": ""}
    assert query_text(good) == good["rewrite"] and query_text(empty) == Q
    assert stage_query(empty) == ""                                          # v1 files: the rewrite, unchanged
    v2 = add_query_field([good, empty])
    assert [stage_query(q) for q in v2] == [good["rewrite"], Q]
    assert [q["query_source"] for q in v2] == ["rewrite", "question"]
    assert "query" not in empty                                              # input not mutated

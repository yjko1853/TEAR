import numpy as np

from tear.rerank.atc import ContainMasks, atc_mask_exact, atc_mask_substring, resolve_labels
from tear.rerank.baselines import chrono_direction, timer4_phi
from tear.rerank.core import rerank
from tear.rerank.tr import before_after_validity, group_keys, group_representatives, non_representative_mask

ENTS = ["Iraq", "Citizen (Iraq)", "Japan"]
RELS = ["Make a visit"]
TSS = ["2006-01-07"]


def test_atc_exact_vs_substring():
    s_c = np.array([0, 1, 2]); o_c = np.array([2, 2, 0]); r_c = np.zeros(3, int); ts_c = np.zeros(3, int)
    cand = np.arange(3)
    sents = ["iraq make a visit japan in 2006-01-07", "citizen (iraq) make a visit japan in 2006-01-07",
             "japan make a visit iraq in 2006-01-07"]
    exact = atc_mask_exact("subject", "Iraq", s_c, o_c, r_c, ENTS, RELS)
    sub = atc_mask_substring("subject", "Iraq", cand, s_c, o_c, r_c, ts_c, ContainMasks(ENTS, RELS, TSS), sents)
    assert exact.tolist() == [True, False, False]
    assert sub.tolist() == [True, True, False]      # the original rule also hits "Citizen (Iraq)"
    assert atc_mask_exact("timestamp", "Iraq", s_c, o_c, r_c, ENTS, RELS) is None


def test_resolve_labels():
    slots = {"1": {"answer_slot": "object", "topic_entity": "Iraq"}, "2": {"answer_slot": "duration"}}
    assert resolve_labels("1", slots) == ("object", "Iraq")
    assert resolve_labels("2", slots) == (None, None)
    assert resolve_labels("1", slots, {"1": {"topic_entity": "Japan"}}) == ("object", "Japan")


def test_group_representatives_first_and_last():
    # answer slot = subject -> groups by (r, o); candidates 0,1,2 share (r, o) with dates 30, 10, 20; 3 is alone
    s_c = np.array([5, 6, 7, 8]); r_c = np.array([0, 0, 0, 0]); o_c = np.array([1, 1, 1, 2])
    keys = group_keys("subject", s_c, r_c, o_c, E=10, R=1)
    dates = np.array([30, 10, 20, 15])
    valid = np.ones(4, bool)
    assert sorted(group_representatives(keys, valid, dates, ["first"]).tolist()) == [1, 3]
    assert sorted(group_representatives(keys, valid, dates, ["last"]).tolist()) == [0, 3]
    # the before/after constraint removes candidate 1 -> earliest valid member of the group is 2
    valid = before_after_validity([("after", 12)], dates, dates, dates, interval=False)
    assert sorted(group_representatives(keys, valid, dates, ["first"]).tolist()) == [2, 3]
    assert non_representative_mask(4, np.array([2, 3])).tolist() == [True, True, False, False]


def test_timer4_phi_and_chrono_direction():
    ph = timer4_phi([("before", 100)], "before", np.array([90, 50, 120]))
    assert np.allclose(ph, [1 - 10 / 50, 0.0, -100.0])
    assert timer4_phi([("after", 100)], "before", np.array([90])) is None
    assert chrono_direction(["first", "after"]) == 1 and chrono_direction(["last"]) == -1 and chrono_direction([]) == 0


def test_tear_scores_rank_penalised_last():
    sims = np.array([[0.9, 0.8, 0.7]], np.float32)
    F = {"atc": np.array([[True, False, False]]), "tr": np.array([[False, True, False]]),
         "window": np.ones((1, 3), bool)}
    assert rerank("none", sims, F).tolist() == [[0, 1, 2]]
    assert rerank("atc", sims, F).tolist() == [[1, 0, 2]]     # 0.9*0.8 = 0.72: below 0.8, still above 0.7
    assert rerank("tear", sims, F, lam1=0.5, lam2=0.5).tolist() == [[2, 0, 1]]


def test_representative_of_and_promote():
    from tear.rerank.core import PROMOTE_EPS, promote_scores
    from tear.rerank.tr import representative_of
    # groups: candidates 0,1,2 share a key, 3 is alone, 4 has a group without representative (invalid)
    keys = np.array([7, 7, 7, 9, 11]); reps = np.array([2, 3])
    assert representative_of(keys, reps).tolist() == [2, 2, 2, 3, -1]
    assert representative_of(keys, np.array([], np.int64)).tolist() == [-1] * 5
    sims = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]], np.float32)
    tr_rep = np.array([[2, 2, 2, 3, -1]])
    sc = promote_scores(sims, tr_rep)
    assert np.isclose(sc[0, 2], 0.9 + PROMOTE_EPS) and sc[0, 0] == np.float32(0.9)   # representative lifted to the group max
    assert np.allclose(sc[0, [1, 4]], [0.8, 0.5]) and np.isclose(sc[0, 3], 0.6 + PROMOTE_EPS)
    assert sims[0, 2] == np.float32(0.7)                                             # input untouched


def test_tr_promote_keeps_other_members_in_place():
    sims = np.array([[0.9, 0.8, 0.7, 0.6]], np.float32)
    F = {"atc": np.zeros((1, 4), bool), "tr": np.array([[True, True, False, False]]),
         "tr_rep": np.array([[2, 2, 2, 3]]), "window": np.array([[True, True, True, False]])}
    assert rerank("tr", sims, F).tolist() == [[0, 2, 1, 3]]                  # original TR: 0.72, 0.64, 0.70, 0.60
    assert rerank("tr_promote", sims, F).tolist() == [[2, 0, 1, 3]]          # 2 lifted above 0; 0 and 1 keep their order
    assert rerank("tear_window_promote", sims, F).tolist() == [[2, 0, 1, 3]]
    F["window"] = np.array([[False, True, True, True]])
    assert rerank("tear_window_promote", sims, F).tolist() == [[2, 1, 3, 0]]  # window applied after the promotion

"""Point-in-time event KG with the indexes the question procedures need.

Facts are (s, r, o, day). A question asks for one side of a fact given the other side and the relation, so the
central index is the *event list* of (relation, topic, answer side): every (partner, day, fact id) with
  answer side 'subject': facts (partner, r, topic, day)
  answer side 'object' : facts (topic, r, partner, day)
sorted by day. The (s, r, o) groups (same triple on several days) are indexed separately.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ...temporal import date_to_days, days_to_date
from ...utils.io import load_json, save_json

SIDES = ("subject", "object")


class EventList:
    """Events of one (relation, topic, answer side), sorted by day: partner ids, day numbers, fact ids."""
    __slots__ = ("partners", "days", "fids")

    def __init__(self, partners, days, fids):
        self.partners, self.days, self.fids = partners, days, fids

    def __len__(self):
        return len(self.days)

    def partner_set(self) -> set:
        return set(self.partners.tolist())

    def in_range(self, lo: int, hi: int) -> slice:
        """Slice of the events with lo <= day <= hi."""
        return slice(int(np.searchsorted(self.days, lo, "left")), int(np.searchsorted(self.days, hi, "right")))

    def first_day_of(self, partner: int) -> int | None:
        m = self.partners == partner
        return int(self.days[m][0]) if m.any() else None


class TemporalKG:
    def __init__(self, s, r, o, days, entities, relations):
        self.S = np.asarray(s, np.int64)
        self.R = np.asarray(r, np.int64)
        self.O = np.asarray(o, np.int64)
        self.D = np.asarray(days, np.int64)
        self.entities, self.relations = list(entities), list(relations)
        self.eid = {e: i for i, e in enumerate(self.entities)}
        self.rid = {x: i for i, x in enumerate(self.relations)}
        self.E, self.NR = len(self.entities), len(self.relations)
        self._index = {}
        self._sro = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_cache(cls, d) -> "TemporalKG":
        """cache/<ds>/facts.npz + vocab.json (the format of scripts/01_prepare_data.py)."""
        d = Path(d)
        f = np.load(d / "facts.npz")
        v = load_json(d / "vocab.json")
        return cls(f["s_id"], f["r_id"], f["o_id"], f["t_days"], v["entities"], v["relations"])

    @property
    def n_facts(self) -> int:
        return len(self.D)

    # ------------------------------------------------------------------ event lists
    def _build(self, side: str):
        topic = self.O if side == "subject" else self.S
        partner = self.S if side == "subject" else self.O
        key = self.R * self.E + topic
        order = np.lexsort((np.arange(len(key)), self.D, key))
        k = key[order]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        ends = np.r_[starts[1:], len(k)]
        self._index[side] = (dict(zip(k[starts].tolist(), zip(starts.tolist(), ends.tolist()))),
                             partner[order], self.D[order], order)

    def events(self, side: str, r: int, topic: int) -> EventList:
        if side not in self._index:
            self._build(side)
        idx, P, D, F = self._index[side]
        a, b = idx.get(r * self.E + topic, (0, 0))
        return EventList(P[a:b], D[a:b], F[a:b])

    def topics(self, side: str, r: int) -> np.ndarray:
        """Topic entities that have at least one event of relation r in this layout."""
        if side not in self._index:
            self._build(side)
        keys = np.array([k for k in self._index[side][0] if k // self.E == r], np.int64)
        return keys % self.E

    def facts_of(self, r: int) -> np.ndarray:
        return np.flatnonzero(self.R == r)

    # ------------------------------------------------------------------ (s, r, o) groups
    def _build_sro(self):
        key = (self.S * self.NR + self.R) * self.E + self.O
        order = np.lexsort((np.arange(len(key)), self.D, key))
        k = key[order]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        ends = np.r_[starts[1:], len(k)]
        size = np.empty(len(key), np.int64)
        ndays = np.empty(len(key), np.int64)
        for a, b in zip(starts, ends):
            size[order[a:b]] = b - a
            ndays[order[a:b]] = len(np.unique(self.D[order[a:b]]))
        self._sro = (dict(zip(k[starts].tolist(), zip(starts.tolist(), ends.tolist()))), order, size, ndays)

    def sro_days(self, s: int, r: int, o: int) -> np.ndarray:
        """Sorted days of the facts (s, r, o, .)."""
        if self._sro is None:
            self._build_sro()
        idx, order, _, _ = self._sro
        a, b = idx.get((s * self.NR + r) * self.E + o, (0, 0))
        return self.D[order[a:b]]

    def sro_fact_ids(self, s: int, r: int, o: int) -> np.ndarray:
        """Fact ids of (s, r, o, .) in day order."""
        if self._sro is None:
            self._build_sro()
        idx, order, _, _ = self._sro
        a, b = idx.get((s * self.NR + r) * self.E + o, (0, 0))
        return order[a:b]

    def sro_n_days(self) -> np.ndarray:
        """Per fact: number of distinct days of its (s, r, o) group."""
        if self._sro is None:
            self._build_sro()
        return self._sro[3]

    def name(self, e: int) -> str:
        return self.entities[int(e)]


# ---------------------------------------------------------------------- RE-NET release (ICEWS18, GDELT, ...)
RENET_START = {"ICEWS18": date(2018, 1, 1), "ICEWS14": date(2014, 1, 1)}


def _read_dict(p: Path) -> list:
    names = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            name, i = line.rstrip("\n").rsplit("\t", 1)
            names[int(i)] = name.strip()
    assert sorted(names) == list(range(len(names))), p
    return [names[i] for i in range(len(names))]


def prepare_renet(raw_dir, out_dir, start: date, hours_per_step: int = 24,
                  splits=("train.txt", "valid.txt", "test.txt")) -> dict:
    """RE-NET quadruple files (s_id, r_id, o_id, t_hours[, 0]) -> cache/<ds>/ in the MultiTQ format of
    tear.data.multitq.prepare: facts.npz (s_id, r_id, o_id, t_id, t_days), vocab.json (entities, relations,
    timestamps in order of first appearance), sentences.txt ('s r o in YYYY-MM-DD'). The background KG is the union
    of the three splits (as MultiTQ uses all of ICEWS05-15)."""
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ents, rels = _read_dict(raw / "entity2id.txt"), _read_dict(raw / "relation2id.txt")
    s_id, r_id, o_id, t_str = [], [], [], []
    per_split = {}
    for name in splits:
        n0 = len(s_id)
        with open(raw / name, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if not p:
                    continue
                s, r, o, t = int(p[0]), int(p[1]), int(p[2]), int(p[3])
                assert t % hours_per_step == 0, line
                s_id.append(s); r_id.append(r); o_id.append(o)
                t_str.append((start + timedelta(days=t // hours_per_step)).isoformat())
        per_split[name] = len(s_id) - n0
    tss = {}
    t_id = [tss.setdefault(t, len(tss)) for t in t_str]
    ts_list = list(tss)
    t_days = np.array([date_to_days(t) for t in ts_list], np.int64)
    np.savez(out / "facts.npz", s_id=np.array(s_id, np.int32), r_id=np.array(r_id, np.int32),
             o_id=np.array(o_id, np.int32), t_id=np.array(t_id, np.int32),
             t_days=t_days[np.array(t_id)].astype(np.int32))
    save_json({"entities": ents, "relations": rels, "timestamps": ts_list}, out / "vocab.json")
    with open(out / "sentences.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(f"{ents[s]} {rels[r]} {ents[o]} in {t}" for s, r, o, t in zip(s_id, r_id, o_id, t_str)) + "\n")
    info = {"facts": len(s_id), "per_split": per_split, "entities": len(ents),
            "entities_used": len(set(s_id) | set(o_id)), "relations": len(rels), "timestamps": len(ts_list),
            "first_day": min(ts_list), "last_day": max(ts_list)}
    save_json(info, out / "kg_info.json")
    return info


def day_str(n: int) -> str:
    return days_to_date(int(n))

"""MultiTQ-style question generation over a point-in-time event KG (ICEWS05-15, ICEWS18, ...).

MultiTQ (Chen et al., ACL 2023) built its questions from templates over ICEWS05-15: 22 frequent relations, six
question types (equal, before_after, first_last; equal_multi, after_first, before_last), answers computed by a
procedure over the KG, three time granularities, questions with more than 20 answers removed. Its generation code is
not public. This package re-implements the procedure from the released questions:

  kg.py         TemporalKG: fact arrays and the (relation, topic, answer side) event index; RE-NET loader (ICEWS18)
  timeexpr.py   date expressions in MultiTQ's formats: parse (span, granularity, day range, format) and render
  aliases.py    surface forms of KG entities ('Government (Nigeria)' -> 'the government of Nigeria', ...)
  semantics.py  the answer procedure of each question type (shared by alignment and generation)
  align.py      alignment of MultiTQ train/dev questions to the KG -> delexicalised templates (frozen before use)
  generate.py   the generator: template -> sampled fact -> answers -> filters -> question text
Fidelity statistics and the rule-label check live in tear/analysis/qgen_fidelity.py (they use tear.labeling).
"""

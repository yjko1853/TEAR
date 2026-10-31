# Frozen MultiTQ templates

Each entry is written by `scripts/17_generate_questions.py extract`; `load_frozen` checks the SHA-256 before generation.

## v1 — 2026-10-18T18:09:40

- file: `resources/qgen/multitq_templates_v1.json.gz`
- SHA-256: `820c4cab789c4f681558e3be8d9444053f4d7d9efe826910315553e3105c7606`
- sources: MultiTQ train (386787 questions, sha256 89ce10b4a1ca9b53...), dev (57979, ad0dfe7c94dfdd7c...); KG cache/multitq/facts.npz (712b5c6e8333d47f...). The MultiTQ test split was not read.
- templates: 46923 (46902 usable), relations: 22, demonyms: 82
- frozen before any ICEWS18 question was generated and before any retrieval, reranking or reader result on generated questions existed.

**Superseded before use (2026-10-18 18:20).** Inspection of the build above, before any question was generated from
it, found three wrong demonyms ('South Korea' -> 'Grand', 'United States' -> 'Mahdi', 'Africa' -> 'Constitutional';
each is the first word of another entity name, e.g. Grand National Party, Mahdi Army). The mining rule now also
requires that an adjective that does not share its first two letters with the country is not the first word of a
multi-word entity name; the relation-phrase table (documentation only) was refined. The build is kept as
`superseded/multitq_templates_v1_build1.json.gz` (same SHA-256 as above) and was never used for generation.

## v1 — 2026-10-18T18:19:38

- file: `resources/qgen/multitq_templates_v1.json.gz`
- SHA-256: `a886f3b884a7622f72f26959dc6e55044ee028f6cb9ade70019fb5fe64d47878`
- sources: MultiTQ train (386787 questions, sha256 89ce10b4a1ca9b53...), dev (57979, ad0dfe7c94dfdd7c...); KG cache/multitq/facts.npz (712b5c6e8333d47f...). The MultiTQ test split was not read.
- templates: 46923 (46902 usable), relations: 22, demonyms: 79
- frozen before any ICEWS18 question was generated and before any retrieval, reranking or reader result on generated questions existed.

## generator v2 config — 2026-10-18T22:14:20

- file: `resources/qgen/generator_v2.json`
- SHA-256: `802176385fa792c055ff13f6192c9eb7413d6a5b5344b4bc4b4d981bf950a5fb`
- rules from MultiTQ dev: single answer for after_first/anchor/entity, before_last/anchor/entity, first_last/extreme/entity, equal_multi/period_fl/entity, after_first/date/entity, before_last/date/entity (dev shares {'after_first|anchor|entity': 1.0, 'before_after|anchor|entity': 0.1493, 'before_after|date|entity': 0.1036, 'before_last|anchor|entity': 1.0, 'equal_multi|period_fl|entity': 1.0, 'equal_multi|same|entity': 0.3744, 'equal|period|entity': 0.4774, 'equal|unique|time': 1.0, 'first_last|extreme|entity': 1.0, 'first_last|extreme|time': 1.0}); anchor events on one day (dev 0.987 of 13737 aligned anchored questions); time seed alpha = 0.5 (first/last time questions in one period: dev 0.568, regenerated 0.573).
- fixed before any retrieval, reranking or reader run on generated questions.

**Generated sets (2026-10-18 22:14).** `cache/icews0515_regen/questions.json`, `cache/icews18/questions.json` and
`questions_dev.json` were generated at 18:21 with the first generator (v1 rules) and regenerated with the v2 config
above; the v1 files were only used for the question-form / KG-side fidelity statistics that led to v2 and never for
retrieval, reranking or reading. Each question file has a `.meta.json` with its SHA-256, the seed, the config and its
SHA-256.

**Generator config superseded before use for retrieval (2026-10-18 22:18).** The fidelity and rule checks of the sets
generated with the config above (SHA-256 `802176385fa792c055ff13f6192c9eb7413d6a5b5344b4bc4b4d981bf950a5fb`, kept as `superseded/generator_v2_build1.json`) showed that 35% of
their topic entities were sector names against 9-10% in MultiTQ, and two surface-form errors ('the citizens of the
Spain': 'of the' used for every country; 'foreign affairses'). MultiTQ dev never names a sector entity as the topic of
an entity question without an anchor and never names two sector entities in one question; the next config adds these
role signatures (shares read from MultiTQ dev), 'of the' follows the country's own article rate, and words already
ending in s are not pluralised again. No retrieval, reranking or reading was run on the superseded sets.

## generator v2 config — 2026-10-18T22:19:11

- file: `resources/qgen/generator_v2.json`
- SHA-256: `a7f13805033e4a4d187b565a880aa9c27cc209f7cd35e0c879cd671df0fb9c7f`
- rules from MultiTQ dev: single answer for after_first/anchor/entity, before_last/anchor/entity, first_last/extreme/entity, equal_multi/period_fl/entity, after_first/date/entity, before_last/date/entity (dev shares {'after_first|anchor|entity': 1.0, 'before_after|anchor|entity': 0.1493, 'before_after|date|entity': 0.1036, 'before_last|anchor|entity': 1.0, 'equal_multi|period_fl|entity': 1.0, 'equal_multi|same|entity': 0.3744, 'equal|period|entity': 0.4774, 'equal|unique|time': 1.0, 'first_last|extreme|entity': 1.0, 'first_last|extreme|time': 1.0}); anchor events on one day (dev 0.987 of 13737 aligned anchored questions); time seed alpha = 0.5 (first/last time questions in one period: dev 0.568, regenerated 0.534); role signatures (sector / other entity per named role) {'entity|anchor': {'FF': 0.7191526534177768, 'FT': 0.16218970663172455, 'TF': 0.11865763995049865}, 'entity|plain': {'F': 1.0}, 'time': {'FF': 0.721032316657837, 'FT': 0.08756527662150912, 'TF': 0.1914024067206539}}.
- fixed before any retrieval, reranking or reader run on generated questions.

**Clarification (2026-10-18 22:35).** "The MultiTQ test split was not read" refers to the extraction input. Before the
v1 freeze (at about 18:00) the test split's question lengths and type proportions, without answers, were compared with
train and dev once; that comparison chose dev as the split that weights the templates. The test questions were first
aligned to the KG at 18:22 (`align-test`), after the freeze.

# Plan for investigating complementary accounts of coherence

Revised 2026-09-10. This revision supersedes the earlier requirement that a
classifier improvement must precede representation analysis or human review.

## Aim and scope

Determine whether PCC, GUM, RST-DT, and NLDT offer distinct, overlapping, and
linguistically defensible descriptions of the same passages. The automated
work prepares evidence and cases for human assessment. Faithful reproduction
of the original paper and maximizing detector performance are not objectives.

Maintain three separate questions:

1. Do discourse motifs carry human/generated label information across inventories?
2. Does combining inventories improve detection under the tested models?
3. Do inventory-dependent analyses highlight different defensible aspects of coherence?

Questions 1 and 2 are useful auxiliary experiments. A positive answer to
question 2 is neither necessary nor sufficient for question 3. Two descriptions
can express different discourse distinctions while predicting the same binary
label. Conversely, classifier gains can reflect parser errors or stylistic cues.
This is the conceptual reason for correcting the previous stopping rule.

The four outputs come from shared UniRST machinery under intentionally shared
segmentation. Findings initially concern these parser outputs conditional on
that design. Attributing differences to the standards themselves requires
comparison with annotation definitions and human judgments.

## Progress and what remains unresolved

| Question | Existing evidence | Defensible conclusion |
|---|---|---|
| Motif utility | All eight inventory/classifier M3 additions to structural and relation-frequency controls have positive identity-clean estimates and pass the reported numerical thresholds. | M3 carries incremental HC3 label information across inventories. |
| Predictive complementarity | With 2,000 bootstrap resamples, none of 14 selected-reference fusion contrasts reaches the required +0.01 AUC increment. Pair gains are about +0.0003–+0.0057. | No addition satisfies the chosen practical criterion under this fusion procedure. |
| Representation overlap | Shared segmentation; reported graph-edge Jaccard 0.970–0.987 and shared-edge label agreement 0.723–0.930. | The exported representations overlap strongly; metric semantics and structural-edge contributions still require verification. |
| Linguistic interpretation | Training relation-correspondence tables, motif mappings, and per-document differences exist. | Renaming, finer/coarser distinctions, alternative interpretations, and parser errors have not yet been separated. |
| Complementary coherence accounts | No blinded linguistic judgments have been collected. | Unresolved; the detector result does not close this question. |

All-inventory fusion versus validation-best PCC has identity-clean AUC gains
of +0.00415 for LR (nominal 95% interval +0.00056 to +0.00791) and +0.00260 for
XGBoost (-0.00056 to +0.00572). These small estimates support ending further
HC3 detector optimization. They do not establish general equivalence or absence
of linguistic complementarity. The +0.01 cutoff is an operational criterion,
not a universal scientific boundary.

The move from 500 to 2,000 bootstrap draws fixed a Monte Carlo resolution
problem: the earlier minimum Holm-adjusted sign value was about 0.0798 across
20 tests. More draws improve numerical resolution; they do not increase the
number of independent documents or establish adequate statistical power.

## Inputs and constraints

Reuse existing artifacts whenever valid:

- Parsed data: `data/unirst2`; catalogs: `data/motifs2`.
- Matrices and models: `artifacts/unirst2_feature_matrices` and
  `artifacts/unirst2_model_comparison`.
- Motif associations and mappings: `artifacts/unirst2_motif_associations`,
  `artifacts/unirst2_robustness`, and
  `artifacts/unirst2_cross_standard_mapping`.
- Cohort audit, prediction comparisons, representation alignment, and fusion
  results: `artifacts/unirst2_complementarity`.

Preserve the released HC3 split membership and intentional segmentation.
Use the existing identity/near-duplicate audits for sensitivities and sampling;
their coverage does not prove removal of all prompt or template dependence.
All analyses on the previously inspected HC3 test outcomes remain exploratory.
Do not require new parsing, M6/M9 extraction, retraining, or OOD collection for
the remaining automated work.

Write new outputs under `artifacts/unirst2_complementarity`, recording source
hashes, parameters, software versions, counts, and exclusions. Reuse verified
results; recompute only measurements shown to be invalid. Keep numerical
results separate from their interpretation and preserve a record of corrections.

## Remaining automated steps

### 1. Verify measurement and reporting before freezing conclusions

Audit the existing summaries against scripts 20 and 21, using targeted tests
where a defect could change the interpretation.

- Verify that graph node IDs refer to the same EDU spans across inventories.
  Report structural/scaffolding edges (including the observed `/` category)
  separately from discourse-relation edges. The current exported-edge metric
  must not be presented as constituent-span agreement without verification.
- Check prediction-bundle identities, labels, lengths, seed metadata, and source
  hashes against the feature matrices before accepting reused predictions.
- Describe the inference actually implemented: nominal percentile intervals;
  Holm-adjusted bootstrap-sign values over 20 contrasts per model/cohort/metric;
  then selection of seven stronger-reference comparisons per model. These
  are not simultaneous corrected intervals or a newly preregistered family.
- The inferential table uses the first seed; two seeds have stored aggregate
  scores, and deterministic LR/GroupKFold repetitions are not independent
  replications. The current gate does not programmatically enforce stability
  across seeds. Verify any retained stability claim from available artifacts.
- Correct language such as "powered rerun", "equivalent", and "redundant" where
  it exceeds the evidence. Preserve the failed +0.01 gate without treating it
  as a linguistic exclusion rule.

Output: `measurement_audit.md` and corrected, versioned summaries as needed.
If an alignment or provenance defect affects a downstream case, exclude it or
repair the affected measurement before proceeding.

**Completed 2026-09-10.** `artifacts/unirst2_complementarity/measurement_audit.md`
and `.json` verify all matrix/prediction identities and source hashes, confirm
valid shared EDU/constituent spans, and provide discourse-only graph metrics in
`representation_metric_audit.csv`. The incremental report and generator now
describe nominal intervals, exploratory seed checks, and the 0/14 primary gate
without implying equivalence or redundancy.

### 2. Identify recurring differences between representations

Reuse the training correspondence table, existing family mappings, and aligned
parses. Freeze rules using training/validation before applying them to test.

- Quantify approximately one-to-one renamings, asymmetric many-to-one mappings
  suggesting different granularity, residual label disagreements, and actual
  attachment/nuclearity differences. Record support and ambiguity by domain.
- Compare native-label and empirically mapped agreement, retaining unmatched
  relations. An empirical mapping is a statistical correspondence, not a claim
  that two relation definitions mean the same thing.
- Locate M3 occurrences on the shared EDU spans for a small shortlist of
  recurrent differences. Recover occurrences from existing parses when the
  aggregate motif vectors cannot identify a passage.
- Keep motif presence, frequency, and conditional depth distinct. Existing
  motif counts pool nuclearity variants; use local trees for nuclearity
  diagnostics and recount only shortlisted occurrences if required.

Output: `representation_contrasts.csv` and `candidate_passages.jsonl`, linking
each candidate to its text spans, both analyses, motif identity, support, and
unresolved explanations. Do not require a classifier improvement to retain a
representational candidate.

**Completed 2026-09-10.** The validation-frozen extraction produced
`artifacts/unirst2_complementarity/representation_contrasts/representation_contrasts.csv`
with 1,428 pair/category summaries and a 104-passage
`candidate_passages.jsonl` pool. All six inventory pairs are represented;
selection is capped and domain-balanced, excludes `/` scaffolding links, and
uses no authorship labels, model scores, or motif coefficients. Exact span and
nuclearity differences were absent from the retained pool; attachment,
one-to-one renaming, many-to-one granularity, residual, and unmapped relation
cases remain available for review.

### 3. Prepare a small blinded review package

Before reading candidate passage content, freeze a deterministic sampling rule:
target at most 60 unique passages, covering all six inventory pairs and five
domains where eligible cases exist. Use quotas across recurring residual
differences, granularity candidates, simple-renaming controls, and agreement
controls. Flag suspected parse failures and report shortages rather than
filling quotas with duplicates or lowering criteria after inspection.

Select from the audited cohort without consulting authorship labels, model
scores, prediction correctness, or motif coefficient signs. Cap duplicate
clusters and repeated prompts/templates where identifiable. Record pair/domain
coverage and inclusion rules; this purposive sample estimates no population
prevalence without an appropriate sampling design.

Show passages, aligned EDUs, and randomly ordered analyses under neutral IDs.
Hide inventory identity, authorship label, and predictions; retain these in a
separate key. Native relation terminology may reveal inventory identity, so
describe blinding as partial and do not rename labels in ways that erase the
distinction being studied.

Prepare a codebook for judgments of textual support, well-formedness, whether
both analyses are defensible, what distinction each highlights, and whether the
difference is mere renaming, granularity, alternative interpretation, or error.
Allow "neither", "uncertain", and "no meaningful difference". Record relevant
annotation-definition references where available and flag definitions requiring
expert clarification.

Output: `human_review/blinded_cases.jsonl`, `blinded_cases.html`,
`codebook.md`, `sampling_manifest.json`, and `sealed_answer_key.jsonl`.
Generation is automated; actual annotation and adjudication require human input.

**Completed 2026-09-10.** The package under
`artifacts/unirst2_complementarity/human_review` contains 60 unique cases,
covering all six inventory pairs and all five domains. Three candidate
identities in near-duplicate clusters were excluded; the sealed key is mode
0600. Analyses were randomly ordered with deterministic IDs, while native
relation labels remain a documented partial-blinding risk.

### 4. Synthesize the evidence and hand off

Update `UNIRST2_RESEARCH_REPORT.md` and create
`artifacts/unirst2_complementarity/report.md` with separate conclusions about
motif utility, fusion performance, representation differences, and unresolved
linguistic validity. Document this change of research procedure as exploratory.

Advance cases to review only when their representations are aligned, traceable,
and supported by enough recurrent examples for the frozen sampling rule.
If all differences reduce to correspondence conventions, or eligible cases
cannot be verified, report that outcome and stop case generation. If auditable
differences remain, hand off the review package regardless of detector gains.

## Deferred work

Further classifier searches, leave-one-inventory-out fusion, performance
ablations, nuclearity-wide catalog regeneration, and fresh OOD data are outside
the necessary next steps. A later confirmatory study needs prospectively fixed
hypotheses and suitable independent evaluation; it is not required to prepare
the present exploratory linguistic review.

Human judgments may ultimately support complementary interpretations, mere
annotation conventions, parser errors, or no meaningful difference. The aim is
to distinguish these outcomes rather than to assume that complementarity will
be found.

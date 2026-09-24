# Research archive

This directory preserves material that is not part of the active canonical
four-standard experiment. Active scripts and paper-facing reports must not read
from this tree.

## Contents

- `code/original_acl2024`: original-repository preprocessing and splitting
  entry points.
- `code/superseded`: analyses replaced by the matched-cohort or fixed-tree
  workflow.
- `code/side_studies`: M6/M9, Longformer, fusion, external-evaluation, and
  other research branches outside the current contribution.
- `code/one_off_utilities`: ad-hoc conversion and merge helpers.
- `artifacts/superseded`: earlier six-condition results, cross-parser
  comparisons, and preliminary figures/reports.
- `artifacts/exploratory`: complementarity, localized ablation, qualitative,
  and minimal-contribution investigations.
- `artifacts/external_evaluation`: OOD and Human Detectors outputs excluded from
  the experimental outcomes.
- `artifacts/legacy_detector`: reserved for legacy detector bundles.
- `docs`: superseded plans and pre-refactor flow audits.

`manifest.json` is a local index of archived files, including generated
artifacts that are excluded from version control. It is intentionally ignored;
regenerate it after an intentional local archive change with:

```bash
python -m scripts.archive_manifest
```

The archived source code and documentation are versioned. Large analysis
artifacts and raw parse
intermediates should be stored in external cold storage and described by a
separate restoration manifest rather than added to Git.

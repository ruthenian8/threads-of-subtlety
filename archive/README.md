# Research archive

The active workflow and paper-facing reports do not read from this directory.
Use the [repository guide](../docs/REPOSITORY_GUIDE.md) for the active code
map and the [canonical pipeline guide](../docs/UNIRST2_PIPELINE.md) for the
supported experiment.

## Versioned contents

The following source and historical documents are tracked in Git:

- `code/original_acl2024/`: original preprocessing, graph, motif-distribution,
  and split entry points.
- `code/superseded/`: analyses replaced by the matched-cohort or fixed-tree
  workflows, plus one-time analysis utilities.
- `code/side_studies/`: M6/M9, Longformer, fusion, OOD, Human Detectors, and
  other investigations outside the current reported experiment.
- `code/one_off_utilities/`: ad-hoc conversion and merge helpers.
- `docs/`: pre-refactor flow and research-plan documents.

These files are retained for provenance and selective reuse; they are not
maintained as part of the supported pipeline. Their outputs may have been
produced with different cohorts, parser versions, or experimental definitions.
Check the source and accompanying report before comparing an archived result
with the canonical report.

## Local-only generated material

The following paths may be present in an existing research working copy, but
are ignored by Git and are not included in a fresh clone:

- `artifacts/superseded/`: earlier model runs and preliminary figures/reports.
- `artifacts/exploratory/`: complementarity, localized ablation, qualitative,
  and minimal-contribution materials.
- `artifacts/external_evaluation/`: OOD and Human Detectors outputs, excluded
  from the current experimental outcomes.
- `artifacts/legacy_detector/`: legacy Longformer outputs and M6/M9 bundles.

Large raw parses, model bundles, and generated analysis outputs should be kept
in external research storage with a separate restoration record. They should
not be added to Git merely because an archive directory exists locally.

## Local archive index

`archive/manifest.json` is an optional, ignored inventory of files present in
the current local archive, including ignored generated outputs. It is not
checked in, does not restore data, and does not describe external storage.
Regenerate it after an intentional local archive change with:

```bash
python -m scripts.archive_manifest
```

The command reports local paths, sizes, and SHA-256 digests. To see exactly
which archive files are included in a clone, use:

```bash
git ls-files archive
```

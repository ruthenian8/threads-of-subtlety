# Threads of Subtlety

This repository studies how four RST relation inventories—PCC, full eRST,
full RST-DT, and NLDT—change motif-level measurements on the same HC3
documents. The canonical experiment compares structural, relation-frequency,
and M3 motif representations; tests fixed-tree fine-to-coarse projections with
LR MF and Kim-style random-forest MF+WAD features; and traces stable motif
associations across domains.

The reported experiment uses a shared EDU segmentation and an identity-matched
cohort of 43,322 training, 968 validation, and 1,932 test documents. PCC and
NLDT define the segmentation target; the eRST and RST-DT inputs match every
recoverable target scene exactly. OOD and Human Detectors analyses are archived
side studies and are not included in the reported outcomes.

## Canonical run

Create the environment from `requirements.txt`, restore `data/unirst2` and
`data/motifs2`, then run:

```bash
python run_canonical.py resume --config configs/canonical.json
python run_canonical.py verify --config configs/canonical.json
```

`resume` validates and reuses matching stage outputs. To refit every model and
recompute every comparison, use:

```bash
python run_canonical.py build --config configs/canonical.json
```

Regenerate only the paper-facing Markdown files with:

```bash
python run_canonical.py report --config configs/canonical.json
```

The configuration centralizes inventory names, input roots, mapping files,
seeds, bootstrap settings, worker counts, and expected output dimensions.

## Active layout

```text
configs/canonical.json       canonical experiment configuration
run_canonical.py             build/resume/verify/report entry point
scripts/prepare/             parsing, alignment, graph, and M3 preparation
scripts/canonical/           quantitative experiment stages
scripts/review/              optional qualitative review workflow
tos/                         parsing, dataset, and shared analysis code
data/unirst2/                aligned graph-added M3 records
data/motifs2/                training-only M3 catalogs and provenance
artifacts/canonical/         verified paper-facing evidence
artifacts/work/              regenerable intermediate matrices and caches
archive/                     obsolete code, side studies, and their artifacts
```

The canonical configuration resolves inputs through
`data/unirst2/shared_segmentation` and `data/motifs2/shared_segmentation`.
These views select the verified shared-segmentation eRST/RST-DT records while
retaining the established PCC/NLDT records and catalogs.

The active workflow does not read from `archive/`. Its contents are indexed by
`archive/manifest.json`; each entry includes its size and SHA-256 digest.

## Input preparation

The full preparation workflow is expensive and normally unnecessary when the
aligned inputs have been restored. Its active modules are:

```bash
python -m scripts.prepare.prepare_and_parse_datasets_unirst --help
python -m scripts.prepare.align_to_paper_splits --help
python -m scripts.prepare.add_graphs --help
python -m scripts.prepare.extract_m3 --help
python -m scripts.prepare.add_motif_distributions --help
```

M3 catalogs must be selected from training data only. Final input records use
the suffix `.graph_added.m3_motif_dists.jsonl` and preserve the released paper
split identities.

## Outputs

The main human-readable outputs are:

- `UNIRST2_RESEARCH_REPORT.md`
- `PAPER_RELEVANT_RESULTS_TABLES.md`
- `artifacts/canonical/report.md`
- `docs/figures/canonical/` tracked paper-facing figures
- `docs/figures/relation_frequencies/` tracked pooled relation-frequency histograms
- `docs/tables/overall_relation_frequencies.csv` pooled counts and edge shares

Every canonical stage writes provenance alongside its outputs. The root
`artifacts/canonical/manifest.json` records the configuration, environment,
headline dimensions, sizes, and hashes used by `run_canonical.py verify`.

## Qualitative review

The optional review modules are under `scripts/review/`. They default to
`artifacts/review` and use the canonical matrices and mapping. Historical
filled review cases remain in the exploratory archive.

## Tests

Run the active suite with:

```bash
pytest -q
```

The canonical verifier is also required because it checks data identities,
inventory membership, result dimensions, report scope, and artifact hashes.

## Archive and migration

See `REPOSITORY_REFACTOR_PLAN.md` for the executed migration design and
`archive/README.md` for disposition rules. Large raw and intermediate parse
files should be stored outside Git with a restoration manifest; generated data
and artifacts remain ignored by default.

## Citation

This codebase extends the discourse-motif workflow released with the original
Threads of Subtlety project. Preserve the original paper citation and license
when redistributing the legacy implementation or derived artifacts.

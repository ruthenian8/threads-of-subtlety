# Repository guide

This page is a practical map for inspecting the project or picking up work. The
root [README](../README.md) summarizes the study; the
[research report](../UNIRST2_RESEARCH_REPORT.md) and
[results tables](../PAPER_RELEVANT_RESULTS_TABLES.md) are the fastest route to
the findings.

## Where to look

| If you want to… | Start with… |
|---|---|
| Understand the claim and its limits | [Research report](../UNIRST2_RESEARCH_REPORT.md), then [results tables](../PAPER_RELEVANT_RESULTS_TABLES.md) |
| Understand the run from inputs to report | [Canonical pipeline](UNIRST2_PIPELINE.md) and [`run_canonical.py`](../run_canonical.py) |
| Change inventories, input roots, seeds, or expected dimensions | [`configs/canonical.json`](../configs/canonical.json) |
| Inspect parsing/alignment and motif preparation | [`scripts/prepare/`](../scripts/prepare/) |
| Inspect feature construction, modeling, and inference | [`scripts/canonical/`](../scripts/canonical/) |
| Inspect the shared fine/coarse helpers | [`tos/analysis/granularity.py`](../tos/analysis/granularity.py) |
| Inspect qualitative review utilities | [`scripts/review/`](../scripts/review/) |
| Find superseded scripts and side studies | [Archive index](../archive/README.md) |
| See what changed in the repository reorganization | [Refactor record](../REPOSITORY_REFACTOR_PLAN.md) |

The active Python code is grouped by purpose: `scripts/prepare` creates the
analysis inputs, `scripts/canonical` runs the quantitative experiment, and
`scripts/review` contains optional qualitative review tools. `tos/` contains
shared parsing, dataset, and analysis code. The one supported full analysis
entry point is `run_canonical.py`; stage modules can also be run individually
when debugging.

## Inputs and local-only outputs

The canonical configuration expects the following restored local inputs:

```text
data/unirst2/shared_segmentation/rel-<inventory>/*.jsonl
data/motifs2/shared_segmentation/<inventory>/*
```

where `<inventory>` is one of `deu.rst.pcc`, `eng.erst.gum.new`,
`eng.rst.rstdt.new`, or `nld.rst.nldt`. These corpora and catalogs are ignored
by Git and are not included in a fresh clone. There is no in-repository data
download command: obtain the project input bundle from the project maintainer
or the research storage used by your team, preserving this relative layout.
Do not substitute the legacy `data/motifs/` catalog for the four inventory-
specific catalogs.

The pipeline writes its complete local run under `artifacts/canonical/` and
rebuildable raw matrices under `artifacts/work/`; these directories are also
ignored. The repository does track the report, selected figures under
`docs/figures/`, and the pooled relation table under `docs/tables/`. Thus a
fresh clone can inspect the main results without the raw inputs, but cannot
recompute or independently verify the full run until the input bundle is
restored.

After restoring inputs, install the Python dependencies and run the usual
sequence from the repository root:

```bash
python -m pip install -r requirements.txt
python run_canonical.py resume --config configs/canonical.json
python run_canonical.py verify --config configs/canonical.json
python -m pytest -q
```

Use `build` instead of `resume` to force a fresh run. Use the command
`report` to regenerate only the Markdown report and tables from existing local
canonical artifacts. The verification command checks artifact provenance and
expected row counts; it does not recreate missing inputs or model outputs.

## Results and their provenance

- [`UNIRST2_RESEARCH_REPORT.md`](../UNIRST2_RESEARCH_REPORT.md): narrative
  interpretation, model results, paired comparisons, and figures.
- [`PAPER_RELEVANT_RESULTS_TABLES.md`](../PAPER_RELEVANT_RESULTS_TABLES.md):
  compact research questions and tables for paper drafting.
- [`docs/figures/canonical/`](figures/canonical/): selected tracked result
  figures. The generated, fuller figure set is in local
  `artifacts/canonical/figures/` after a run.
- [`docs/figures/relation_frequencies/`](figures/relation_frequencies/):
  tracked relation histograms.
- [`docs/tables/overall_relation_frequencies.csv`](tables/overall_relation_frequencies.csv):
  pooled edge counts and shares by split and inventory.

Detailed stage-level CSV/JSON outputs and manifests live under
`artifacts/canonical/` when restored or generated locally. Archived results are
not inputs to the canonical reports; consult the [archive README](../archive/README.md)
before using them, since some are exploratory or superseded.

## Common maintenance checks

Run commands from the repository root so the configured relative paths resolve.
Before sharing a code or configuration change, run:

```bash
python -m pytest -q
python run_canonical.py verify --config configs/canonical.json
```

The second command requires the complete local canonical run. If it is not
available, state that limitation rather than treating unit tests as a
substitute for run verification. Keep generated corpora and full run outputs
out of Git; see [`.gitignore`](../.gitignore) and the [archive policy](../archive/README.md).

# threads-of-subtlety
Code repository for the paper: "Threads of Subtlety: Detecting Machine-Generated Texts Through Discourse Motifs" (ACL 2024)

- Code: this repo.
- Models: [zaemyung/ToS-Longformer-Plain](https://huggingface.co/zaemyung/ToS-Longformer-Plain/tree/main), [zaemyung/ToS-Longformer-Motif](https://huggingface.co/zaemyung/ToS-Longformer-Motif/tree/main)
- Dataset: [zaemyung/ToS-Dataset](https://huggingface.co/datasets/zaemyung/ToS-Dataset/tree/main)

Some preprocessing updates were made, which may have contributed to performance improvements on the OOD and OOD-Para test sets.

|                  | HC3    | MAGE-Test | MAGE-OOD | MAGE-OOD-Para |
|------------------|--------|-----------|----------|---------------|
| Longformer-Plain | 96.74% | 88.64%    | 69.48%   | 55.95%        |
| Longformer-Motif | 97.42% | 91.83%    | 82.63%   | 76.15%        |

## UniRST reproduction

The UniRST reproduction replaces the DMRST parsing stage without patching
`isanlp_rst`. Each scene is segmented once with `deu.rst.pcc`; the segmented
EDUs are persisted as validated pickle artifacts and reused through UniRST's
`from_edus()` API. Consequently, all relation-inventory predictions use the
same EDU boundaries.

The four relation inventories are:

`eng.erst.gum`, `eng.rst.rstdt`, `deu.rst.pcc`, and `nld.rst.nldt`.

Install the dependencies, including UniRST and its IsaNLP dependency, from
`requirements.txt`:

```bash
pip install -r requirements.txt
```

Run the preparation and parsing stage:

```bash
python 0_prepare_and_parse_datasets_unirst.py \
  --total-gpus 1 --gpu-id 0 --output-dir data/unirst
```

Then add discourse graphs:

```bash
python 1_add_graphs_to_unirst_datasets.py --root data/unirst
```

The default UniRST motif workflow uses M3 and M6 only. Each RST inventory is
processed independently, with parsed graphs below `data/unirst` and motif
catalogs below `data/motifs`:

```bash
python 2_extract_single_triads.py --data-dir data
python 3_extract_double_triads.py --data-dir data
```

M6 selection can independently require scene support, require cross-shard
support, and retain only a cumulative fraction of the surviving scene hits.
For example, the pruning profile discussed above is:

```bash
python 3_extract_double_triads.py --data-dir data \
  --min-scene-support 10 \
  --min-shard-support 2 \
  --coverage 0.95
```

Support thresholds are applied first. Coverage then keeps the smallest
frequency-ranked prefix accounting for the requested fraction of the
surviving support. Defaults are `1`, `1`, and `1.0`, preserving the unpruned
workflow. Each inventory also receives an `hc3_M6_support.json` audit file.

Both commands discover the `rel-*` directories and process every relation
inventory by default. Pass `--relinventory` to process only one; with that
option, `--motif-dir` names the exact inventory directory. The default dataset
prefix is `hc3`, matching the per-inventory files below. Pass
`--dataset-name hc3-mage` when extracting a combined HC3/MAGE catalog. Input
discovery follows this value: `hc3` reads `hc3_*` shards, `mage` reads
`mage_*`, and `hc3-mage` reads both.

The M6 command writes a matching M3+M6 selection manifest, so M9 extraction is
not a prerequisite for the distribution stage. The resulting layout is:

```text
data/motifs/
├── eng.erst.gum/
│   ├── hc3_M3_motifs.json
│   ├── hc3_M6_motifs.json
│   ├── hc3_M6_support.json
│   └── hc3_selected-motif-hashes.generated.json
├── eng.rst.rstdt/
├── deu.rst.pcc/
└── nld.rst.nldt/
```

```bash
python 5_add_motif_dists_to_unirst_datasets.py --data-dir data
```

Script 5 loads only the `m3` and `m6` manifest entries and writes only those
two distribution groups. Existing manifests that also contain `m9` remain
valid; the extra entry is ignored. The generated manifest selects hashes from
that standard's M3 and M6 files.
Pass `--selected-hashes` to use a curated manifest instead; stale manifests
are rejected with an actionable error.

Distribution filenames end in `.m3_m6_motif_dists.jsonl`. The explicit feature
set prevents an older M3+M6+M9 output from being mistaken for a completed
M3+M6 artifact and skipped.

`4_extract_triple_triads.py` remains available for optional M9 experiments, but
it is deliberately not part of the default UniRST workflow.

Motif distributions are written incrementally to `*.partial` files and moved
atomically to their final paths when complete. Existing final outputs are
skipped; pass `--force` to rebuild them. Use `--workers` and `--chunksize`
(documents per worker task) to tune CPU parallelism and IPC batching.

To compare the optimized exact motif operations with their original reference
implementations on deterministic synthetic graphs, run:

```bash
PYTHONPATH=. python benchmarks/benchmark_motifs.py --nodes 40 --repeats 3
```

The benchmark covers M3 extraction, motif distributions, corpus-driven M6
extraction, and exact parallel M9 composition. Use `--skip-m9` when the
checked-in M3/M6 motif catalogs are unavailable; tune production parallelism
with `--workers` and IPC batching with `--chunksize` on the M6 and M9 scripts.

The preparation script supports `--output-mode separate`, `paired`, or
`both`; `both` is the default. It also supports `--force-segmentation` when a
segmentation cache must be regenerated.

Generated files are organized as follows:

```text
data/unirst/
├── segments-deu.rst.pcc/        # persisted shared EDU segmentation pickles
├── rel-eng.erst.gum/            # pipeline-compatible parsed documents
├── rel-eng.rst.rstdt/
├── rel-deu.rst.pcc/
├── rel-nld.rst.nldt/
└── paired/                      # all four predictions per scene
```

The `rel-*` directories contain the normal `.discourse_parsed.jsonl`
artifacts consumed by the graph and motif stages. The paired artifacts retain
the shared EDU list together with each inventory's prediction and status.

On subsequent runs, a group whose requested output paths already exist is
skipped before segmentation and parsing. If only some separate inventories are
missing, the preparation resumes with those inventories; a missing paired
artifact requires all inventories to be parsed so it can contain the complete
prediction set. Use `--force-parsing` to regenerate outputs, or
`--force-segmentation` when the segmentation cache and all dependent outputs
must be rebuilt.

For a multi-GPU MAGE run, launch the preparation script once per GPU. Each
worker receives a disjoint dataset chunk and writes a GPU-specific pickle and
output shard:

```bash
python 0_prepare_and_parse_datasets_unirst.py \
  --total-gpus 4 --gpu-id 0 --output-dir data/unirst
python 0_prepare_and_parse_datasets_unirst.py \
  --total-gpus 4 --gpu-id 1 --output-dir data/unirst
# repeat for --gpu-id 2 and --gpu-id 3
```

HC3 and MAGE loading, scene splitting, and downstream graph/motif formats are
kept compatible with the original pipeline.

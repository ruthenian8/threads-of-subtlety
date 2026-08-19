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

If the motif definitions are being regenerated, each RST inventory is processed
independently. This creates separate motif files and manifests per standard:

```bash
for relinventory in eng.erst.gum eng.rst.rstdt deu.rst.pcc nld.rst.nldt; do
  motif_dir="data/motifs/${relinventory}"
  python 2_extract_single_triads.py \
    --root data/unirst --relinventory "$relinventory" --motif-dir "$motif_dir"
  python 3_extract_double_triads.py \
    --root data/unirst --relinventory "$relinventory" --motif-dir "$motif_dir"
  python 4_extract_triple_triads.py --motif-dir "$motif_dir"
done
```

Each final triad command writes a matching
`hc3-mage_selected-motif-hashes.generated.json`. The distribution stage can
then be run once per standard:

```text
data/motifs/
├── eng.erst.gum/
├── eng.rst.rstdt/
├── deu.rst.pcc/
└── nld.rst.nldt/
```

```bash
for relinventory in eng.erst.gum eng.rst.rstdt deu.rst.pcc nld.rst.nldt; do
  python 5_add_motif_dists_to_unirst_datasets.py \
    --root data/unirst --relinventory "$relinventory" \
    --motif-dir "data/motifs/${relinventory}"
done
```

The generated manifest selects hashes from that standard's motif files.
Pass `--selected-hashes` to use a curated manifest instead; stale manifests
are rejected with an actionable error.

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

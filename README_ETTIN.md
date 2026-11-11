# Ettin DMRST preprocessing

Branch feature/ettin-dmrst derives from dev at e658d85. The existing script
names and document, scene-tree, graph, and motif formats are retained.

## Approved design

- Shared segmentation: larc-iu/dmrst-ettin-400m-gum-12.1.0.
- Keep Dipo's default forced paragraph/heading boundaries.
- Parse that same cached EDU sequence with both requested checkpoints.
- Preserve native relation labels and predicted NS/SN/NN nuclearity. RST-DT
  coarse has 18 relation labels; GUM 12.1.0 has 32. No new label projection,
  inventory matching, or nuclearity pooling is applied.
- Keep dev's dataset ingestion, labels, scene splitting, output checkpointing,
  graphs, motif extraction/selection, distribution calculation, and split logic.
  Documents fitting one GUM encoder window are processed intact. Longer
  documents fall back to the inherited XLM-R token-budget scene splitter.
  This changes tree boundaries relative to the earlier scene-split setup;
  use a new output root rather than mixing results.

These are two trained parsers with different corpora and relation spaces.
Their score difference is not an inventory-only or fine-versus-coarse effect.

## Installation

Use a separate Python 3.10+ environment (3.12 recommended):
```bash
python -m pip install -r requirements-ettin.txt
```
Dipo and both checkpoint revisions are pinned in tos/ettin.py and the new
requirements file. Transformers 4.49.0 supports Ettin's ModernBERT backbone;
the legacy requirements.txt pin does not. No UniRST weights are loaded.

## Prepare and parse

```bash
python 0_prepare_and_parse_datasets_unirst.py --output-dir data/ettin/unirst
```

All dev input flags remain available: --hc3-dataset, --mage-dataset,
--human-detectors-path, and --storyscope-dataset/--storyscope-splits.
--relinventories can select either or both of:
- eng.erst.gum-12.1.0.ettin
- eng.rst.rstdt-coarse.ettin

The default is both. --segment-relinventory defaults to the approved GUM key;
changing it changes the experiment and requires a separate output directory.
The default cache directory is segments-eng.erst.gum-12.1.0.ettin inside the
output root. Existing UniRST caches do not validate against this backend.

The current pinned Dipo API runs individual documents. --batch-size remains
accepted for bounded scheduling, but emits a warning when greater than one:
it does not provide native multi-document GPU inference. Segmentation calls
the learned segmenter and the same paragraph/heading rules as predict_from_text,
without decoding a tree. Parsing uses predict() with supplied EDUs and validates
that returned leaves retain exactly their text and order.

An ettin_provenance.json file protects existence-based output resumption from
model/policy changes. Output roots containing pre-existing parses without that
manifest are rejected. New model keys also keep the two outputs distinct from
the four original UniRST inventories. Inference failures retain the existing
per-scene statuses and NONE sentinel.

## Continue with the inherited workflow

Use a separate data root to avoid mixing catalogs from different models:
```bash
python 1_add_graphs_to_unirst_datasets.py --root data/ettin/unirst
python 2_extract_single_triads.py --data-dir data/ettin --dataset-name hc3
python 3_extract_double_triads.py --data-dir data/ettin --dataset-name hc3
python 5_add_motif_dists_to_unirst_datasets.py --data-dir data/ettin --dataset-name hc3
python 6_split_hc3_dataset.py --data-dir data/ettin
```
Catalog extraction and distribution generation discover the rel-* directories.
Build new catalogs from the intended training inputs; do not reuse the original
UniRST catalogs. Selection rules are unchanged by this integration. Training
and evaluation scripts accept --data-dir data/ettin and the new --relinventory
keys. Their old classifier artifacts are not trained on these new features.

Sources:
- https://huggingface.co/larc-iu/dmrst-ettin-400m-rstdt-coarse
- https://huggingface.co/larc-iu/dmrst-ettin-400m-gum-12.1.0
- https://github.com/larc-iu/dipo

## Offline verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/ettin_adapter_test.py tests/unirst_batch_preparation_test.py \
  tests/storyscope_preparation_test.py tests/motif_optimization_test.py \
  tests/motif_distribution_workflow_test.py tests/double_motif_selection_test.py \
  tests/triple_inventory_test.py
```

These tests cover pinned loader arguments, segmentation-only inference,
paragraph rules, shared-cache reuse, native-label conversion, and graph/motif
compatibility. A tiny randomly initialized ModernBERT exercises Dipo's actual
segmentation and parsing code without downloading checkpoint weights. This is
an integration check, not validation of the released checkpoints or GPU memory
requirements. Legacy UniRST-specific tests require the legacy environment.

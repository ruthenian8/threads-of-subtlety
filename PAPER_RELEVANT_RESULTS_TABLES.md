# Paper-relevant results tables: canonical four-standard analysis

These tables use the exact same HC3 identities and shared EDU segmentation for PCC, full eRST, full RST-DT, and NLDT. Models are trained on 43,322 documents, selected on 968 validation documents, and evaluated on 1,932 test documents. Alternative-segmentation runs are retained only as sensitivity artifacts and are not counted as additional standards.

## Research questions

1. Do M3 motifs add authorship-related information beyond document structure and marginal relation frequency?
2. Does fine relation resolution retain predictive information that is lost under fixed-tree coarse projection?
3. Are motif associations stable in sign across HC3 domains, and are presence and depth empirically distinct?
4. Which effects overlap across standards and which depend on inventory-specific distinctions?

## Table 1. Canonical cohort and feature dimensions

| Standard | Train | Validation | Test | Relation features | M3 features |
|---|---:|---:|---:|---:|---:|
| PCC | 43,322 | 968 | 1,932 | 26 | 70 |
| eRST | 43,322 | 968 | 1,932 | 33 | 90 |
| RST-DT | 43,322 | 968 | 1,932 | 61 | 161 |
| NLDT | 43,322 | 968 | 1,932 | 27 | 74 |

## Table 2. HC3 test F1

| Standard | Model | Structural | Relation frequency | M3 presence | M3 frequency | M3 depth |
|---|---|---:|---:|---:|---:|---:|
| PCC | LR | 0.678 | 0.624 | 0.723 | 0.768 | 0.703 |
| PCC | XGB | 0.699 | 0.774 | 0.748 | 0.796 | 0.756 |
| eRST | LR | 0.687 | 0.688 | 0.746 | 0.785 | 0.741 |
| eRST | XGB | 0.710 | 0.788 | 0.762 | 0.801 | 0.774 |
| RST-DT | LR | 0.674 | 0.686 | 0.742 | 0.748 | 0.709 |
| RST-DT | XGB | 0.691 | 0.761 | 0.746 | 0.771 | 0.770 |
| NLDT | LR | 0.664 | 0.666 | 0.719 | 0.753 | 0.714 |
| NLDT | XGB | 0.696 | 0.754 | 0.755 | 0.789 | 0.761 |

## Table 3. Paired increment of M3 frequency over structure (F1)

| Standard | Model | M3 frequency − structure F1 | 95% CI | Holm-adjusted value |
|---|---|---:|---:|---:|
| PCC | LR | +0.090 | [+0.062, +0.118] | 0.0070 |
| PCC | XGB | +0.096 | [+0.072, +0.123] | 0.0070 |
| eRST | LR | +0.098 | [+0.070, +0.126] | 0.0070 |
| eRST | XGB | +0.091 | [+0.066, +0.118] | 0.0070 |
| RST-DT | LR | +0.074 | [+0.045, +0.103] | 0.0070 |
| RST-DT | XGB | +0.080 | [+0.053, +0.106] | 0.0070 |
| NLDT | LR | +0.088 | [+0.059, +0.118] | 0.0070 |
| NLDT | XGB | +0.092 | [+0.068, +0.117] | 0.0070 |

## Table 4. Fixed-tree fine-to-coarse RF MF+WAD ablation (F1)

| Standard | Fine MF+WAD F1 | Coarse MF+WAD F1 | Fine − coarse (95% CI) | Holm-adjusted p |
|---|---:|---:|---:|---:|
| PCC | 0.778 | 0.751 | +0.028 [+0.011, +0.047] | 0.0160 |
| eRST | 0.795 | 0.775 | +0.021 [+0.003, +0.039] | 0.0840 |
| RST-DT | 0.780 | 0.771 | +0.009 [-0.008, +0.026] | 0.6457 |
| NLDT | 0.763 | 0.757 | +0.006 [-0.011, +0.024] | 0.6457 |

## Table 5. Fixed-tree fine-to-coarse LR ablation with MF and MF+WAD (F1)

| Standard | Representation | Fine F1 | Coarse F1 | Fine − coarse (95% CI) | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| PCC | MF | 0.768 | 0.687 | +0.081 [+0.061, +0.102] | 0.0040 |
| PCC | MF_WAD | 0.774 | 0.716 | +0.057 [+0.040, +0.076] | 0.0040 |
| eRST | MF | 0.785 | 0.729 | +0.056 [+0.035, +0.078] | 0.0040 |
| eRST | MF_WAD | 0.778 | 0.697 | +0.081 [+0.057, +0.104] | 0.0040 |
| RST-DT | MF | 0.748 | 0.746 | +0.002 [-0.014, +0.019] | 0.8226 |
| RST-DT | MF_WAD | 0.770 | 0.739 | +0.032 [+0.013, +0.050] | 0.0040 |
| NLDT | MF | 0.753 | 0.696 | +0.057 [+0.037, +0.076] | 0.0040 |
| NLDT | MF_WAD | 0.766 | 0.702 | +0.064 [+0.046, +0.084] | 0.0040 |

Paired change in the fine-minus-coarse gain after adding WAD:

| Standard | (MF+WAD gain) − (MF gain), F1 (95% CI) | Holm-adjusted p |
|---|---:|---:|
| PCC | -0.024 [-0.044, -0.003] | 0.0660 |
| eRST | +0.025 [+0.004, +0.044] | 0.0660 |
| RST-DT | +0.029 [+0.012, +0.049] | 0.0080 |
| NLDT | +0.007 [-0.011, +0.027] | 0.4498 |

The paired interaction tests whether adding WAD changes LR's fine-minus-coarse gain, using the same identity-cluster bootstrap resamples for MF and MF+WAD. Coarse WAD is occurrence-weighted when fine motifs merge. RF retains its Kim-style fixed 0.5 threshold; LR uses the established validation-selected threshold.

## Table 6. Stable held-out motif associations

| Standard | Presence | Frequency | Conditional depth |
|---|---:|---:|---:|
| PCC | 3 | 1 | 0 |
| eRST | 4 | 3 | 0 |
| RST-DT | 2 | 1 | 0 |
| NLDT | 3 | 2 | 0 |

## Table 7. Cross-standard mapping summary

The four catalogs contain 395 inventory-specific motifs, grouped into 287 pipeline-equivalent and 14 relation-agnostic structural families. Structural families—not raw hashes—are the defensible unit for broad cross-standard comparison.

## Table 8. Domain-conditioned analysis

Training selected 80 domain–motif hypotheses; 48 replicated on test and 47 also survived the identity-clean sensitivity cohort. 4 exact-label motif families produced recurring robust domain interactions in multiple standards. This supports substantial inventory specificity alongside a smaller shared structural signal.

## Segmentation target

All four inventories use the same EDU segmentation. PCC and NLDT define the shared segmentation target; the eRST and RST-DT inputs were independently verified against it with 49,360 recoverable scenes matching exactly and no boundary mismatches.

## Scope

F1 for label 1 (the human class) is the primary reported predictive metric. The LR/XGBoost representation models and LR granularity fits retain ROC-AUC validation selection and validation-only threshold selection. The RF granularity comparison uses Kim-style fixed 100-tree random forests over MF+WAD with a fixed 0.5 threshold. No test-set tuning was performed. All outcomes are parser-conditional measurements on matched HC3 documents. They compare measurement consequences of relation inventories; they do not rank linguistic correctness or parser quality.

# Version History

Development timeline for the signal peptide prediction study.

## Script Development Progression

| Script | Description | Key Output |
|--------|-------------|------------|
| 01 | Reproduce Grasso et al. RF baseline | MSE = 1.22 (matches paper) |
| 02 | RF hyperparameter search (grid + random) | Best RF config identified |
| 03 | NN regression hyperparameter search | Best NN config: (256,128), dropout=0.4 |
| 04 | Final comparison across PLM embeddings | ESM2-650M best overall |
| 05 | Cross-dataset generalization (Wu, Xue, Zhang) | Transfer learning evaluation |
| 06 | Vector regression (10-bin softmax) | CE vs focal loss comparison |
| 07 | Design task evaluation | Per-gene ranking on 4,911 variants |
| 08 | Bootstrap confidence intervals | 95% CIs for all 15 models |
| 09 | Vector architecture search (4 phases) | Best: deeper-3layer, MSE = 0.973 |
| 10 | Vector ensemble optimization | Best: MSE = 0.9323, beats benchmark |
| 11 | Dropout validation sweep | Optimal dropout = 0.35 |
| 12 | Bimodal distribution figure (Figure 1) | Motivating example |
| 13 | Cross-dataset fine-tuning | Transfer via frozen-layer fine-tuning |
| save_best_model | Save 5-seed ensemble weights | models/ with round-trip validation |
| 14 | ReLU² vs LeakyReLU activation comparison | Sparsity analysis + bootstrap CIs |
| 15 | Regression vs classification comparison | 10-bin vs 5/3/2-class info loss |
| 16 | Standardize parquet schema | Unified 9,963-row dataset |
| 00 | ESM2-650M embedding generation for designs | 4,911 design variant embeddings |

## Key Milestones

- **Baseline reproduction**: Script 01 confirmed Grasso et al. RF at
  MSE = 1.22 on the held-out test set (1,326 samples).

- **Initial improvement**: Script 04 showed NN regression with ESM2-650M
  embeddings achieves MSE = 1.05, a 14% improvement over baseline.

- **Vector regression**: Script 06 introduced 10-bin probability
  prediction via softmax output, achieving MSE ~1.00 with focal loss
  on Ginkgo-AA0 embeddings.

- **Architecture search**: Script 09 systematically searched deeper
  architectures, activation functions, and ensemble sizes. Best
  configuration: 3-layer (256, 256, 128) with LeakyReLU.

- **Final result**: Script 10 optimized dropout and ensemble strategy,
  achieving MSE = 0.9323 [0.823, 1.054] 95% CI, a 2.2% improvement
  over the benchmark (0.953) and 23.6% over baseline (1.22).

- **Model persistence**: save_best_model trains and saves the 5-seed
  ensemble (256,256,128, dropout=0.35, focal loss) to `models/` with
  round-trip validation. Adds `load_ensemble()` and
  `predict_ensemble_wa()` to `src/models.py`.

- **Activation comparison**: Script 14 tests ReLU²(x) = max(0,x)² vs
  LeakyReLU. Hypothesis: squared activation produces sparser
  representations for sparse bin distributions. Includes Hoyer sparsity
  metric and bootstrap CIs.

- **Output formulation**: Script 15 compares 4 output strategies
  (10-bin vector regression, 5-class, 3-class, binary) on identical
  architecture. Quantifies information loss as output dimensionality
  decreases from 10 bins to binary.

- **Data standardization**: Script 16 consolidates all 7 datasets into
  a unified parquet schema (9,963 rows) with consistent column names,
  producing `data/unified/all_datasets_esm2_650M.parquet`.

## Figure Revision History

- **v1** (initial): Figures generated with default matplotlib styling,
  value labels on bars, grid lines, descriptive titles.

- **v2** (current): Tufte-style cleanup applied to all 11 figures.
  Removed chart junk (value labels, grid lines, titles, bar edges),
  thinned reference lines, shortened legend labels. Figure 1 redesigned
  from 3-panel to single-panel (aspB).

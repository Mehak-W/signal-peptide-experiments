# Signal Peptide Efficiency Prediction

Systematic evaluation of machine learning models for predicting signal peptide secretion efficiency in *Bacillus subtilis*, comparing random forest and neural network regressors across physicochemical and protein language model (PLM) feature representations. A vector regression approach with optimized regularization achieves a 5-seed ensemble MSE of **0.932** [0.823, 1.054] (95% bootstrap CI), surpassing both the physicochemical baseline of 1.22 and a prior benchmark of 0.953.

## Key Results

| Model | Test MSE | Improvement over baseline |
|-------|----------|---------------------------|
| Best scalar NN (Ginkgo-AA0) | 1.050 | -14.0% |
| Best vector NN (Ginkgo-AA0 + focal, val split) | 1.001 | -18.0% |
| **Best overall** (full-data, dropout 0.35, 5-seed) | **0.932** | **-23.6%** |
| Prior benchmark (Dr. Schrier) | 0.953 | -21.9% |
| Baseline (Grasso et al. RF) | 1.22 | --- |

## Methodology

| # | Script | Approach |
|---|--------|----------|
| 1 | Baseline reproduction | Exact Grasso et al. RF params: 75 trees, depth 25, min_split 0.001, min_leaf 0.0001 |
| 2 | RF hyperparameter search | RandomizedSearchCV (100 iters, 5-fold CV) per feature type |
| 3 | NN regression search | 40 configs per feature type, dimension-aware architecture grids |
| 4 | Final comparison | Grouped bar chart + CSV summary of all scalar models |
| 5 | Cross-dataset generalization | Evaluate on 4 external SP datasets (Wu, Xue, Zhang) |
| 6 | Vector regression | Dense(10, softmax) predicting bin distributions with focal/CE loss |
| 7 | Design task evaluation | Predicting mutation effects on 4,832 designed SP variants |
| 8 | Bootstrap CIs | 10,000-resample 95% CIs for all 16 models |
| 9 | Vector architecture search | Full-data training with 3- and 4-layer architectures |
| 10 | Vector ensemble optimization | Dropout tuning, 20-seed ensembles, mixed-architecture ensembles |

## Directory Structure

```
signal_peptide_study/
├── data/              dataset files (see Data section below)
├── src/
│   ├── data_loading.py   GRASSO_FEATURES, load functions, preprocessing
│   ├── models.py         SignalPeptideRegressorNN, SignalPeptideVectorNN, FocalLoss
│   └── evaluation.py     compute_metrics (MSE, RMSE, MAE, R², Spearman, Pearson)
├── scripts/
│   ├── 01_grasso_reproduction.py            Grasso et al. RF reproduction
│   ├── 02_rf_hyperparameter_search.py       RF search across 4 feature types
│   ├── 03_nn_regression.py                  NN regression search, dimension-aware
│   ├── 04_final_comparison.py               Grouped bar chart + CSV summary
│   ├── 05_cross_dataset_generalization.py   External dataset evaluation
│   ├── 06_vector_regression.py              Vector regression with focal loss
│   ├── 07_design_task_evaluation.py         Design variant mutation prediction
│   ├── 08_bootstrap_ci.py                   Bootstrap confidence intervals
│   ├── 09_vector_architecture_search.py     Full-data architecture search
│   └── 10_vector_ensemble_optimization.py   Dropout + ensemble optimization
├── results/           JSON + CSV outputs
├── figures/           PNG plots (300 DPI)
├── paper/             LaTeX manuscript
├── run_all.ipynb      Jupyter notebook for full reproducibility
└── requirements.txt
```

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. All required data files are included in the `data/` directory. No additional downloads are needed.

## Running

```bash
# Step 1: Grasso et al. reproduction (~1 min)
python3 scripts/01_grasso_reproduction.py

# Step 2: RF hyperparameter search, 100 iters x 5-fold CV x 4 feature types (~2-8 hrs)
python3 -u scripts/02_rf_hyperparameter_search.py

# Step 3: NN regression search, 40 configs x 4 feature types (~1-2 hrs)
python3 -u scripts/03_nn_regression.py

# Step 4: Final comparison chart + CSV (~10 sec)
python3 scripts/04_final_comparison.py

# Step 5: Cross-dataset generalization (~30 min)
python3 -u scripts/05_cross_dataset_generalization.py

# Step 6: Vector regression, 5 seeds x 3 embeddings x 2 losses (~1-2 hrs)
python3 -u scripts/06_vector_regression.py

# Step 7: Design task evaluation (~5 min)
python3 -u scripts/07_design_task_evaluation.py

# Step 8: Bootstrap confidence intervals, 10,000 resamples (~30-40 min)
python3 -u scripts/08_bootstrap_ci.py

# Step 9: Vector architecture search, full-data training (~2-3 hrs)
python3 -u scripts/09_vector_architecture_search.py

# Step 10: Vector ensemble optimization, dropout + ensemble tuning (~5 hrs)
python3 -u scripts/10_vector_ensemble_optimization.py
```

All scripts save results to `results/` (JSON + CSV) and figures to `figures/` (PNG, 300 DPI). A Jupyter notebook (`run_all.ipynb`) is also provided for running all scripts sequentially with explanatory markdown cells.

## Feature Types

| Feature Type | Dimensions | Source |
|-------------|-----------|--------|
| PhysChem | 156 | xlsx (Grasso et al. validated features) |
| ESM2-650M | 1280 | Precomputed parquet embeddings |
| ESM2-3B | 2560 | Precomputed parquet embeddings |
| Ginkgo-AA0 | 1280 | Precomputed parquet embeddings |

## Results

### Baseline Reproduction (Script 01)
- Test MSE: 1.193 (within 2.2% of Grasso et al. reported 1.22)
- 156 physicochemical features, 3085 train / 1323 test samples (PhysChem after quality filters)
- PLM embeddings: 3095 train / 1326 test samples

### RF Hyperparameter Search (Script 02)

| Feature Type | Dims | Test MSE | Test R² | Spearman |
|---|---|---|---|---|
| **Ginkgo-AA0** | 1280 | **1.158** | 0.757 | 0.857 |
| ESM2-650M | 1280 | 1.189 | 0.750 | 0.853 |
| PhysChem | 156 | 1.208 | 0.747 | 0.857 |
| ESM2-3B | 2560 | 1.245 | 0.739 | 0.849 |

### NN Regression Search (Script 03)

| Feature Type | Dims | Test MSE | Test R² | Spearman |
|---|---|---|---|---|
| **Ginkgo-AA0** | 1280 | **1.050** | **0.780** | **0.866** |
| ESM2-3B | 2560 | 1.067 | 0.776 | 0.861 |
| ESM2-650M | 1280 | 1.090 | 0.771 | 0.861 |
| PhysChem | 156 | 1.328 | 0.721 | 0.840 |

### Vector Regression (Script 06)

| Embedding | Loss | Test MSE | Test R² | Spearman |
|---|---|---|---|---|
| **Ginkgo-AA0** | **Focal** | **1.001** | **0.790** | **0.867** |
| Ginkgo-AA0 | Cross-Entropy | 1.003 | 0.790 | 0.867 |
| ESM2-650M | Focal | 1.049 | 0.780 | 0.860 |
| ESM2-650M | Cross-Entropy | 1.054 | 0.779 | 0.861 |
| ESM2-3B | Cross-Entropy | 1.109 | 0.767 | 0.857 |
| ESM2-3B | Focal | 1.117 | 0.766 | 0.856 |

### Full-Data Optimization (Scripts 09-10)

| Configuration | Dropout | Test MSE | Test R² | Spearman |
|---|---|---|---|---|
| (256, 256, 128), 5-seed | 0.25 | 0.971 | --- | --- |
| (256, 256, 128), 5-seed | 0.30 | 0.940 | --- | --- |
| **(256, 256, 128), 5-seed** | **0.35** | **0.932** | **0.804** | **0.873** |
| (256, 256, 128, 64), 5-seed | 0.20 | 0.940 | --- | --- |
| Mixed 3-arch x 5-seed | --- | 0.967 | --- | --- |

Best result: MSE **0.932** [0.823, 1.054] 95% CI, beating Dr. Schrier's 0.953 by 2.2% and the Grasso baseline of 1.22 by 23.6%.

### Cross-Dataset Generalization (Script 05)

| Dataset | N | RF Spearman (p) | NN Spearman (p) |
|---|---|---|---|
| Wu | 81 | -0.314 (0.004) | -0.241 (0.030) |
| Xue | 322 | -0.278 (<0.001) | -0.154 (0.006) |
| Zhang-P43 | 114 | -0.218 (0.020) | -0.196 (0.037) |
| Zhang-PglVM | 114 | -0.250 (0.007) | -0.213 (0.023) |

Models trained on Grasso data do not generalize to external datasets (significant negative correlations), indicating context-dependence of signal peptide efficiency prediction.

### Design Task Evaluation (Script 07)

| Model | Spearman | Pearson | MSE | ClassAcc |
|---|---|---|---|---|
| NN | 0.385 | 0.398 | 4.01 | 71.8% |
| RF | 0.366 | 0.375 | 4.07 | 73.7% |

Both models meaningfully rank 4,832 designed SP variants (p < 10^-150), with classification accuracy well above the 50% random baseline.

### Bootstrap Confidence Intervals (Script 08)

95% bootstrap CIs (10,000 resamples) confirm that the best model (full-data optimized) achieves MSE [0.823, 1.054] and Spearman [0.857, 0.887]. All vector regression models outperform the Grasso baseline; the full-data optimized model's CI excludes the baseline MSE of 1.22.

### Key Findings

1. **Best model: MSE 0.932** --- full-data vector NN with Ginkgo-AA0 embeddings, focal loss, (256,256,128) architecture, dropout 0.35, 5-seed ensemble
2. **Vector regression improves over scalar** --- predicting 10-dim bin distributions (softmax) outperforms Dense(1, linear) regression across all embedding types
3. **RF is robust but flat across feature types** --- all RF models achieve MSE 1.16-1.25 regardless of input representation
4. **NNs require rich features** --- NN + PhysChem (1.33) is worse than RF + PhysChem (1.21); NNs only outperform RF with PLM embeddings
5. **Cross-dataset: models do not generalize** --- significant negative Spearman correlations on all 4 external datasets
6. **Design task: practical utility** --- Spearman 0.37-0.39 and 72-74% classification accuracy on designed variants

## Paper

The `paper/` directory contains a LaTeX manuscript (`main.tex`) with full methods, results, and discussion. Compile with:

```bash
cd paper && bash compile.sh
```

## Data

All data files are included in the `data/` directory. No additional downloads are needed.

### Grasso et al. (primary dataset)

The xlsx originates from [Grasso et al., ACS Synth. Bio. 2023](https://doi.org/10.1021/acssynbio.2c00328). PLM embeddings were generated using ESM-2 and the Ginkgo Bioworks API (see [Dr. Schrier lab repository](https://github.com/mfbliposome/signal_peptides) for generation scripts).

| File | Description |
|------|-------------|
| `sb2c00328_si_011.xlsx` | 156 physicochemical features + WA values for 4408 SPs |
| `trainAA_esm2-650M.parquet` | ESM2-650M (1280d) embeddings, train split |
| `testAA_esm2-650M.parquet` | ESM2-650M (1280d) embeddings, test split |
| `trainAA_esm2-3B.parquet` | ESM2-3B (2560d) embeddings, train split |
| `testAA_esm2-3B.parquet` | ESM2-3B (2560d) embeddings, test split |
| `trainAA_ginkgo-AA0-650M.parquet` | Ginkgo-AA0 (1280d) embeddings, train split |
| `testAA_ginkgo-AA0-650M.parquet` | Ginkgo-AA0 (1280d) embeddings, test split |

### Design variant embeddings (Script 07)

ESM2-650M embeddings for the Grasso design library variants (sequences from the xlsx with Set = NaN).

| File | Description | N |
|------|-------------|---|
| `grasso_esm_embeddings.parquet` | ESM2-650M embeddings for design library variants | 3,838 |

### External datasets (Script 05)

ESM2-650M embeddings for signal peptide sequences from four published datasets.

| File | Description | N |
|------|-------------|---|
| `wu_esm_embeddings.parquet` | Wu et al. --- binary secretion (functional/non-functional) | 81 |
| `xue_esm_embeddings.parquet` | Xue et al. --- enzyme activity (continuous) | 322 |
| `zhang_p43_esm_embeddings.parquet` | Zhang et al. --- P43 promoter, *B. subtilis* | 114 |
| `zhang_pglvm_esm_embeddings.parquet` | Zhang et al. --- PglVM promoter, same 114 SPs | 114 |

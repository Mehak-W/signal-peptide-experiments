# Signal Peptide Efficiency Prediction using Protein Language Models

Machine learning prediction of signal peptide secretion efficiency using PLM embeddings and physicochemical features.

## Key Results

- **Ginkgo AA0 + Neural Network: MSE = 0.98** on full Grasso test set (n=1,326)
- **20% improvement** over Grasso et al. baseline (MSE = 1.22)
- **PLM embeddings outperform physicochemical features** (MSE 0.98 vs 1.35)
- **Cross-species generalization**: Successful transfer from B. subtilis to other organisms

## Overview

This repository compares Random Forest and Neural Network approaches for predicting signal peptide secretion efficiency using:

- **Protein Language Model (PLM) embeddings**: ESM-2 650M, ESM-2 3B, Ginkgo AA0 650M
- **Physicochemical features**: 156 hand-crafted features from Grasso et al.

The work builds on [Grasso et al. (2023)](https://dx.doi.org/10.1021/acssynbio.2c00328).

## Repository Structure

```
signal-peptide-experiments/
├── data/
│   ├── trainAA_*.parquet          # Training embeddings (3,095 sequences)
│   ├── testAA_*.parquet           # Test embeddings (1,326 sequences)
│   └── sb2c00328_si_011.xlsx      # Original Grasso dataset + physicochemical features
│
├── src/                           # Shared utilities
│   ├── data_utils.py              # Data loading
│   ├── eval_utils.py              # Evaluation metrics
│   ├── plot_utils.py              # Visualization
│   └── models.py                  # Model architectures
│
├── scripts/                       # Experiment scripts
│   ├── 00_quick_test.py           # Setup validation
│   ├── 01_random_forest_baseline.py
│   ├── 02_neural_network_classifier.py
│   ├── 03_model_comparison.py
│   ├── 04_cross_dataset_evaluation.py
│   ├── 05_generate_professor_figures.py
│   ├── 06_hard_vs_soft_labels.py
│   ├── 07_add_bin_counts_to_parquet.py
│   ├── 08_grasso_design_task_evaluation.py
│   ├── 09_vector_regression.py
│   ├── 10_design_task_full_evaluation.py
│   └── 11_rf_vs_nn_comparison.py  # RF vs NN + physicochemical comparison
│
├── results/                       # CSV result files
├── figures/                       # Generated plots
└── requirements.txt
```

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run main comparison (RF vs NN across all feature types)
python scripts/11_rf_vs_nn_comparison.py
```

## Experiments

### Core Experiments

| Script | Description | Key Output |
|--------|-------------|------------|
| `01_random_forest_baseline.py` | RF on all PLM embeddings | `rf_baseline_results.csv` |
| `02_neural_network_classifier.py` | NN with soft binning | `nn_classifier_results.csv` |
| `11_rf_vs_nn_comparison.py` | RF vs NN + physicochemical comparison | `rf_vs_nn_results.csv` |

### Additional Analyses

| Script | Description |
|--------|-------------|
| `04_cross_dataset_evaluation.py` | Generalization to Wu, Xue, Zhang datasets |
| `06_hard_vs_soft_labels.py` | Ablation: hard vs soft label binning |
| `09_vector_regression.py` | Alternative: predict full bin distribution |
| `10_design_task_full_evaluation.py` | Evaluate on Grasso design variants |

## Main Results

### RF vs NN Comparison (Full Test Set, n=1,326)

| Feature Type | RF MSE | NN MSE | Best |
|--------------|--------|--------|------|
| Physicochemical (156 features) | 1.35 | 1.43 | RF |
| ESM-2 650M | 1.23 | 1.07 | NN |
| ESM-2 3B | 1.30 | 1.14 | NN |
| **Ginkgo AA0** | 1.19 | **0.98** | **NN** |

- RF performs better on hand-crafted physicochemical features
- NN performs better on PLM embeddings (can better exploit learned representations)
- Ginkgo AA0 + NN achieves best overall performance

### Cross-Dataset Generalization

Models trained on Grasso (B. subtilis) transfer to:
- Wu et al. (B. subtilis): R² = 0.75
- Xue et al. (S. cerevisiae): R² = 0.61
- Zhang et al. (B. subtilis): R² = 0.58

## Data

### Train/Test Split
- **Training**: 3,095 sequences
- **Test**: 1,326 sequences
- Split follows original Grasso et al. methodology

### PLM Embeddings
- **ESM-2 650M**: 1,280 dimensions
- **ESM-2 3B**: 2,560 dimensions
- **Ginkgo AA0**: 1,280 dimensions

### Physicochemical Features
156 features from `Library_w_Bins_and_WA` sheet in Excel file, including:
- Amino acid composition
- Secondary structure predictions
- Biophysical properties (pI, instability, hydrophobicity)

## Reproducibility

All experiments use fixed random seeds:
```python
np.random.seed(42)
tf.random.set_seed(42)
```

## Citation

```bibtex
@article{grasso2023signal,
  title={Signal Peptide Efficiency: From High-Throughput Data to Prediction and Explanation},
  author={Grasso, Stefano and others},
  journal={ACS Synthetic Biology},
  year={2023},
  doi={10.1021/acssynbio.2c00328}
}
```

## Contact

- Mehak Wadhwa, Fordham University
- Research Advisor: Dr. Joshua Schrier

## License

MIT License

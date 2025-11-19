# Signal Peptide Efficiency Prediction using Protein Language Models

Machine learning prediction of signal peptide efficiency using pre-computed PLM embeddings from ESM-2 and Ginkgo models.

## Key Results

- **Neural Network MSE = 0.9847** on Grasso benchmark dataset (test set, n=1,326)
- **19.3% improvement** over baseline (Grasso et al. 2023: MSE = 1.22)
- **Cross-species generalization**: Successful transfer from B. subtilis → S. cerevisiae (R² = 0.745)

## Overview

This repository contains experiments comparing Random Forest and Neural Network approaches for predicting signal peptide secretion efficiency using Protein Language Model (PLM) embeddings. The work builds on [Grasso et al. (2023)](https://dx.doi.org/10.1021/acssynbio.2c00328) and uses pre-computed embeddings from three PLMs:

- **ESM-2 650M**: 1280-dimensional embeddings
- **ESM-2 3B**: 2560-dimensional embeddings
- **Ginkgo AA0 650M**: 1280-dimensional embeddings (best performance)

## Repository Structure

```
signal_peptide_experiments/
├── data/                          # Pre-computed PLM embeddings (parquet files)
│   ├── trainAA_esm2-650M.parquet
│   ├── testAA_esm2-650M.parquet
│   ├── trainAA_esm2-3B.parquet
│   ├── testAA_esm2-3B.parquet
│   ├── trainAA_ginkgo-AA0-650M.parquet
│   ├── testAA_ginkgo-AA0-650M.parquet
│   └── sb2c00328_si_011.xlsx     # Original Grasso et al. dataset
│
├── src/                           # Shared utility code
│   ├── __init__.py
│   ├── data_utils.py             # Data loading and processing
│   ├── eval_utils.py             # Model evaluation and metrics
│   ├── plot_utils.py             # Visualization functions
│   └── models.py                 # Model architectures (RF, NN)
│
├── scripts/                       # Experiment scripts (run these!)
│   ├── 01_random_forest_baseline.py
│   ├── 02_neural_network_classifier.py
│   └── 03_model_comparison.py
│
├── results/                       # Experiment results (CSV files)
├── figures/                       # Generated plots
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

Each script is self-contained and can be run independently:

```bash
# Random Forest baseline on all PLM embeddings
python scripts/01_random_forest_baseline.py

# Neural Network classifier on all PLM embeddings
python scripts/02_neural_network_classifier.py

# Comprehensive comparison of all models
python scripts/03_model_comparison.py
```

## Experiments

### Experiment 1: Random Forest Baseline
**Script**: `scripts/01_random_forest_baseline.py`

Trains Random Forest regressors on all three PLM embeddings with standard hyperparameters:
- n_estimators: 100
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2

**Outputs**:
- `results/rf_baseline_results.csv`: Performance metrics
- `figures/rf_*_predictions.png`: Prediction scatter plots
- `figures/rf_embedding_comparison.png`: Embedding comparison

### Experiment 2: Neural Network Classifier
**Script**: `scripts/02_neural_network_classifier.py`

Trains neural network classifiers using soft binning approach:
- Architecture: Single hidden layer (256 units)
- Dropout: 0.3
- L2 regularization: 0.001
- Soft binning: 10 bins with Gaussian weighting

**Outputs**:
- `results/nn_classifier_results.csv`: Performance metrics
- `figures/nn_*_training.png`: Training curves
- `figures/nn_*_predictions.png`: Prediction scatter plots
- `figures/nn_embedding_comparison.png`: Embedding comparison

### Experiment 3: Model Comparison
**Script**: `scripts/03_model_comparison.py`

Comprehensive comparison of all model-embedding combinations:
- Random Forest vs Neural Network
- All three PLM embeddings
- Multiple evaluation metrics (MSE, R², Spearman correlation)

**Outputs**:
- `results/comprehensive_comparison.csv`: Complete results table
- `figures/model_comparison_*.png`: Comparison plots
- `figures/model_comparison_heatmap.png`: Performance heatmap

## Data Format

### PLM Embeddings (Parquet Files)
Each parquet file contains three columns:
- `sequence`: Amino acid sequence of signal peptide
- `embedding`: Dense vector representation from PLM (numpy array)
- `WA`: Weighted average efficiency score (1=best, 10=worst)

### Train/Test Split
Data is pre-split into training and test sets based on the original Grasso et al. study:
- Training: ~3,095 sequences
- Test: ~1,326 sequences

## Performance Metrics

All experiments report:
- **MSE** (Mean Squared Error): Primary metric for comparison
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **Spearman ρ**: Rank correlation coefficient
- **Pearson r**: Linear correlation coefficient

## Customization

### Modifying Hyperparameters

Edit the hyperparameters in each script's `main()` function:

```python
# In 02_neural_network_classifier.py
hyperparams = {
    'architecture': 'two_layer',    # Change to two layers
    'units': [512, 256],            # Increase hidden units
    'dropout_rates': [0.4, 0.3],
    'epochs': 150,
    # ... other parameters
}
```

### Adding New Experiments

1. Create a new script in `scripts/`
2. Import utilities from `src/`
3. Use helper functions for data loading, evaluation, and plotting

Example:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_utils import load_plm_embeddings
from eval_utils import evaluate_model
from plot_utils import plot_predictions_vs_actual
```

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- NumPy: `np.random.seed(42)`
- TensorFlow: `tf.random.set_seed(42)`
- Scikit-learn: `random_state=42`

## Citation

If you use this code or data, please cite:

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

For questions or issues, please open a GitHub issue or contact:
- Mehak Wadhwa
- Fordham University
- Research Mentor: Dr. Joshua Schrier

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PLM embeddings computed using the Ginkgo BioWorks API
- Original dataset from Grasso et al. (2023)
- ESM-2 models from Meta AI

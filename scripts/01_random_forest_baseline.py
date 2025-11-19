#!/usr/bin/env python3
"""
Random Forest Baseline for Signal Peptide Prediction
======================================================

Trains Random Forest models on PLM embeddings (ESM-650M, ESM-3B, Ginkgo-AA0)
to predict signal peptide efficiency scores.

This serves as a baseline to compare against more sophisticated neural network approaches.

Usage:
    python scripts/01_random_forest_baseline.py

Outputs:
    - Model performance metrics for all three PLM embeddings
    - Prediction plots saved to figures/
    - Results summary saved to results/
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data_utils import load_plm_embeddings, get_model_info
from eval_utils import evaluate_model, create_results_dataframe
from plot_utils import plot_predictions_vs_actual, plot_embedding_comparison


def train_random_forest_on_embedding(
    model_name: str,
    n_estimators: int = 100,
    max_depth: int = 20,
    random_state: int = 42
):
    """
    Train Random Forest on a specific PLM embedding.

    Parameters
    ----------
    model_name : str
        Name of PLM model ('esm2-650M', 'esm2-3B', or 'ginkgo-AA0-650M')
    n_estimators : int
        Number of trees in forest
    max_depth : int
        Maximum depth of trees
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    results : dict
        Dictionary containing metrics and predictions
    """
    print(f"\n{'='*70}")
    print(f"Training Random Forest on {model_name}")
    print(f"{'='*70}\n")

    # Load embeddings
    X_train, X_test, y_train, y_test = load_plm_embeddings(model_name)

    # Build and train model
    print(f"Building Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    print("Training...")
    model.fit(X_train, y_train)

    # Evaluate
    metrics, y_train_pred, y_test_pred = evaluate_model(
        model, X_train, y_train, X_test, y_test,
        model_name=f"Random Forest ({model_name})"
    )

    # Create prediction plots
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    plot_predictions_vs_actual(
        y_test, y_test_pred,
        title=f"Random Forest: {model_name}",
        save_path=figures_dir / f"rf_{model_name}_predictions.png"
    )

    return {
        'model_name': model_name,
        'metrics': metrics,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }


def main():
    """Run Random Forest baseline experiments on all PLM embeddings."""
    print("\n" + "="*70)
    print("Random Forest Baseline Experiments")
    print("="*70)

    # Get available models
    model_info = get_model_info()
    model_names = list(model_info.keys())

    # Train on each embedding
    all_results = {}
    results_list = []

    for model_name in model_names:
        result = train_random_forest_on_embedding(model_name)
        all_results[model_name] = result

        # Collect metrics for comparison
        results_df = create_results_dataframe(
            result['metrics'],
            model_name='Random Forest',
            embedding_model=model_name
        )
        results_list.append(results_df)

    # Combine results
    combined_results = pd.concat(results_list, ignore_index=True)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Random Forest Performance Across PLM Embeddings")
    print("="*70 + "\n")
    print(combined_results[['embedding_model', 'test_mse', 'test_r2', 'test_spearman']].to_string(index=False))
    print()

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    combined_results.to_csv(results_dir / 'rf_baseline_results.csv', index=False)
    print(f"Results saved to {results_dir / 'rf_baseline_results.csv'}")

    # Create embedding comparison plot
    figures_dir = Path(__file__).parent.parent / 'figures'
    metrics_dict = {
        result['model_name']: result['metrics']
        for result in all_results.values()
    }

    plot_embedding_comparison(
        metrics_dict,
        metric='test_mse',
        title='Random Forest: PLM Embedding Comparison (Test MSE)',
        save_path=figures_dir / 'rf_embedding_comparison.png'
    )

    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

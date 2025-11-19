#!/usr/bin/env python3
"""
Comprehensive Model Comparison
===============================

Compares Random Forest and Neural Network approaches across all PLM embeddings.
Generates comprehensive comparison figures and tables.

Usage:
    python scripts/03_model_comparison.py

Outputs:
    - Combined comparison plots
    - Performance comparison table
    - Best model identification
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

from data_utils import load_plm_embeddings, get_model_info
from eval_utils import compute_regression_metrics
from models import ProteinClassifierNN


def train_and_evaluate_all_models():
    """
    Train and evaluate all model-embedding combinations.

    Returns
    -------
    results_df : pd.DataFrame
        Comprehensive results table
    """
    model_info = get_model_info()
    model_names = list(model_info.keys())

    results_list = []

    for embedding_model in model_names:
        print(f"\n{'='*70}")
        print(f"Evaluating on {embedding_model}")
        print(f"{'='*70}\n")

        # Load embeddings
        X_train, X_test, y_train, y_test = load_plm_embeddings(embedding_model)

        # 1. Random Forest
        print("Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_test_pred_rf = rf.predict(X_test)

        rf_metrics = compute_regression_metrics(y_test, y_test_pred_rf)
        rf_metrics['model'] = 'Random Forest'
        rf_metrics['embedding'] = embedding_model
        results_list.append(rf_metrics)

        # 2. Neural Network
        print("Training Neural Network...")
        nn = ProteinClassifierNN(n_bins=10, bin_strategy='quantile')
        nn.fit(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            architecture='single_layer',
            units=[256],
            dropout_rates=[0.3],
            epochs=100,
            verbose=0
        )
        y_test_pred_nn = nn.predict(X_test)

        nn_metrics = compute_regression_metrics(y_test, y_test_pred_nn)
        nn_metrics['model'] = 'Neural Network'
        nn_metrics['embedding'] = embedding_model
        results_list.append(nn_metrics)

    # Create DataFrame
    results_df = pd.DataFrame(results_list)

    # Reorder columns
    cols = ['model', 'embedding', 'mse', 'rmse', 'mae', 'r2', 'spearman', 'pearson']
    results_df = results_df[cols]

    return results_df


def plot_comprehensive_comparison(results_df, save_dir):
    """
    Create comprehensive comparison plots.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table
    save_dir : Path
        Directory to save figures
    """
    save_dir.mkdir(exist_ok=True)

    # Set up style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # 1. MSE Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    embeddings = results_df['embedding'].unique()
    models = results_df['model'].unique()

    x = np.arange(len(embeddings))
    width = 0.35

    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        values = [model_data[model_data['embedding'] == emb]['mse'].values[0]
                  for emb in embeddings]
        ax.bar(x + i*width, values, width, label=model)

    ax.set_xlabel('PLM Embedding', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Model Performance Comparison: Test MSE', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(embeddings, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison_mse.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. R² Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        values = [model_data[model_data['embedding'] == emb]['r2'].values[0]
                  for emb in embeddings]
        ax.bar(x + i*width, values, width, label=model)

    ax.set_xlabel('PLM Embedding', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Model Performance Comparison: Test R²', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(embeddings, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison_r2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap of all metrics
    fig, ax = plt.subplots(figsize=(14, 8))

    # Pivot data for heatmap
    pivot_data = results_df.pivot_table(
        values='mse',
        index='embedding',
        columns='model'
    )

    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Test MSE'}, ax=ax)
    ax.set_title('Test MSE Heatmap: Model × Embedding', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('PLM Embedding', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nComparison plots saved to {save_dir}/")


def main():
    """Run comprehensive model comparison."""
    print("\n" + "="*70)
    print("Comprehensive Model Comparison")
    print("="*70)

    # Train and evaluate all models
    results_df = train_and_evaluate_all_models()

    # Print results table
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70 + "\n")
    print(results_df.to_string(index=False))
    print()

    # Identify best models
    print("\n" + "="*70)
    print("BEST MODELS")
    print("="*70 + "\n")

    best_mse = results_df.loc[results_df['mse'].idxmin()]
    best_r2 = results_df.loc[results_df['r2'].idxmax()]
    best_spearman = results_df.loc[results_df['spearman'].idxmax()]

    print(f"Best MSE: {best_mse['model']} on {best_mse['embedding']} (MSE: {best_mse['mse']:.3f})")
    print(f"Best R²: {best_r2['model']} on {best_r2['embedding']} (R²: {best_r2['r2']:.3f})")
    print(f"Best Spearman: {best_spearman['model']} on {best_spearman['embedding']} "
          f"(ρ: {best_spearman['spearman']:.3f})")
    print()

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_df.to_csv(results_dir / 'comprehensive_comparison.csv', index=False)
    print(f"Results saved to {results_dir / 'comprehensive_comparison.csv'}")

    # Create comparison plots
    figures_dir = Path(__file__).parent.parent / 'figures'
    plot_comprehensive_comparison(results_df, figures_dir)

    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

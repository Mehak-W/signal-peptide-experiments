#!/usr/bin/env python3
"""
Cross-Dataset Evaluation for Signal Peptide Prediction
=======================================================

Tests models trained on Grasso dataset for generalization to other datasets:
- Wu et al. (B. subtilis, 81 sequences)
- Xue et al. (S. cerevisiae, 322 sequences)
- Zhang et al. P43 promoter (443 sequences)
- Zhang et al. PglvM promoter (443 sequences)

This evaluates whether the learned representations transfer across:
- Different organisms (B. subtilis vs S. cerevisiae)
- Different experimental conditions
- Different data sources

Usage:
    python scripts/04_cross_dataset_evaluation.py

Outputs:
    - Cross-dataset performance metrics
    - Comparison figures across all datasets
    - Generalization analysis
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

from data_utils import load_plm_embeddings, load_cross_dataset, get_cross_datasets
from eval_utils import compute_regression_metrics, print_metrics_report
from plot_utils import plot_predictions_vs_actual
from models import ProteinClassifierNN


def train_models_on_grasso():
    """
    Train both Random Forest and Neural Network on Grasso dataset.

    Returns
    -------
    models : dict
        Dictionary with trained models and scalers
    """
    print("\n" + "="*70)
    print("Training Models on Grasso Dataset (for cross-dataset evaluation)")
    print("="*70 + "\n")

    # Load Grasso data with best embedding (Ginkgo AA0)
    X_train, X_test, y_train, y_test = load_plm_embeddings('ginkgo-AA0-650M')

    models = {}

    # 1. Train Random Forest
    print("Training Random Forest...")
    rf_scaler = StandardScaler()
    X_train_scaled_rf = rf_scaler.fit_transform(X_train)

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled_rf, y_train)

    models['random_forest'] = {
        'model': rf,
        'scaler': rf_scaler,
        'name': 'Random Forest'
    }
    print(f"  ✓ Random Forest trained")

    # 2. Train Neural Network
    print("Training Neural Network...")
    nn_scaler = StandardScaler()
    X_train_full_scaled = nn_scaler.fit_transform(X_train)

    # Split for validation
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full_scaled, y_train, test_size=0.2, random_state=42
    )

    classifier = ProteinClassifierNN(n_bins=10, bin_strategy='quantile')
    classifier.create_bins(y_train)

    # Build and train
    y_tr_encoded = classifier.encode_labels_soft(y_tr, temperature=0.5)
    y_val_encoded = classifier.encode_labels_soft(y_val, temperature=0.5)

    model = classifier.create_model(
        input_dim=X_tr.shape[1],
        architecture='single_layer',
        units=[256, 128],
        dropout_rates=[0.3],
        l2_reg=0.001
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0
        )
    ]

    model.fit(
        X_tr, y_tr_encoded,
        validation_data=(X_val, y_val_encoded),
        batch_size=32,
        epochs=100,
        callbacks=callbacks,
        verbose=0
    )

    classifier.model = model

    models['neural_network'] = {
        'classifier': classifier,
        'scaler': nn_scaler,
        'name': 'Neural Network'
    }
    print(f"  ✓ Neural Network trained\n")

    return models


def evaluate_on_cross_dataset(models, dataset_name):
    """
    Evaluate trained models on a cross-dataset.

    Parameters
    ----------
    models : dict
        Trained models from train_models_on_grasso()
    dataset_name : str
        Name of cross-dataset

    Returns
    -------
    results : dict
        Performance metrics for each model
    """
    print(f"\n{'='*70}")
    print(f"Evaluating on {dataset_name.upper()} Dataset")
    print(f"{'='*70}\n")

    # Load cross-dataset
    X, y = load_cross_dataset(dataset_name)

    results = {}

    # Evaluate each model
    for model_type, model_data in models.items():
        print(f"\n{model_data['name']}:")
        print("-" * 40)

        # Scale features
        X_scaled = model_data['scaler'].transform(X)

        # Get predictions
        if model_type == 'random_forest':
            y_pred = model_data['model'].predict(X_scaled)
        else:  # neural_network
            y_pred = model_data['classifier'].predict(X_scaled)

        # Compute metrics
        metrics = compute_regression_metrics(y, y_pred)

        # Print key metrics
        print(f"  MSE:        {metrics['mse']:.4f}")
        print(f"  R²:         {metrics['r2']:.4f}")
        print(f"  Spearman ρ: {metrics['spearman']:.4f}")

        results[model_type] = {
            'metrics': metrics,
            'y_true': y,
            'y_pred': y_pred,
            'model_name': model_data['name']
        }

    return results


def create_cross_dataset_plots(all_results, save_dir):
    """
    Create comprehensive cross-dataset comparison plots.

    Parameters
    ----------
    all_results : dict
        Results from all datasets
    save_dir : Path
        Directory to save figures
    """
    save_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # 1. MSE Comparison Across Datasets
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = list(all_results.keys())
    models = ['random_forest', 'neural_network']
    model_names = ['Random Forest', 'Neural Network']

    x = np.arange(len(datasets))
    width = 0.35

    for i, (model, name) in enumerate(zip(models, model_names)):
        mse_values = [all_results[ds][model]['metrics']['mse'] for ds in datasets]
        ax.bar(x + i*width, mse_values, width, label=name)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Cross-Dataset Generalization: MSE Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([ds.replace('_', ' ').title() for ds in datasets], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'cross_dataset_mse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. R² Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model, name) in enumerate(zip(models, model_names)):
        r2_values = [all_results[ds][model]['metrics']['r2'] for ds in datasets]
        ax.bar(x + i*width, r2_values, width, label=name)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Cross-Dataset Generalization: R² Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([ds.replace('_', ' ').title() for ds in datasets], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'cross_dataset_r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap of Performance
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create data for heatmap
    heatmap_data = []
    for ds in datasets:
        row = []
        for model in models:
            row.append(all_results[ds][model]['metrics']['mse'])
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=[ds.replace('_', ' ').title() for ds in datasets],
        columns=model_names
    )

    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Test MSE'}, ax=ax)
    ax.set_title('Cross-Dataset Performance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_dir / 'cross_dataset_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Individual prediction plots for best model on each dataset
    best_model = 'neural_network'  # Assuming NN is better
    for ds in datasets:
        result = all_results[ds][best_model]
        plot_predictions_vs_actual(
            result['y_true'],
            result['y_pred'],
            title=f"Neural Network on {ds.replace('_', ' ').title()}",
            save_path=save_dir / f"cross_dataset_{ds}_predictions.png"
        )

    print(f"\nCross-dataset comparison plots saved to {save_dir}/")


def main():
    """Run comprehensive cross-dataset evaluation."""
    print("\n" + "="*70)
    print("Cross-Dataset Evaluation for Signal Peptide Prediction")
    print("="*70)
    print("\nThis evaluates generalization of Grasso-trained models to:")
    print("  • Wu et al. (B. subtilis)")
    print("  • Xue et al. (S. cerevisiae)")
    print("  • Zhang et al. (P43 and PglvM promoters)")
    print()

    # 1. Train models on Grasso
    models = train_models_on_grasso()

    # 2. Evaluate on all cross-datasets
    cross_datasets = get_cross_datasets()
    all_results = {}

    for dataset_name in cross_datasets.keys():
        results = evaluate_on_cross_dataset(models, dataset_name)
        all_results[dataset_name] = results

    # 3. Create summary table
    print("\n" + "="*70)
    print("SUMMARY: Cross-Dataset Performance")
    print("="*70 + "\n")

    summary_data = []
    for ds_name in cross_datasets.keys():
        for model_type in ['random_forest', 'neural_network']:
            metrics = all_results[ds_name][model_type]['metrics']
            summary_data.append({
                'Dataset': ds_name.replace('_', ' ').title(),
                'Model': all_results[ds_name][model_type]['model_name'],
                'MSE': metrics['mse'],
                'R²': metrics['r2'],
                'Spearman': metrics['spearman']
            })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()

    # 4. Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    summary_df.to_csv(results_dir / 'cross_dataset_results.csv', index=False)
    print(f"Results saved to {results_dir / 'cross_dataset_results.csv'}")

    # 5. Create plots
    figures_dir = Path(__file__).parent.parent / 'figures'
    create_cross_dataset_plots(all_results, figures_dir)

    # 6. Generalization analysis
    print("\n" + "="*70)
    print("GENERALIZATION ANALYSIS")
    print("="*70 + "\n")

    # Compare performance on Grasso vs cross-datasets
    grasso_mse_nn = 0.9847  # From previous results
    cross_mses_nn = [all_results[ds]['neural_network']['metrics']['mse']
                     for ds in cross_datasets.keys()]

    avg_cross_mse = np.mean(cross_mses_nn)
    degradation = ((avg_cross_mse - grasso_mse_nn) / grasso_mse_nn) * 100

    print(f"Neural Network Performance:")
    print(f"  Grasso (in-distribution):     MSE = {grasso_mse_nn:.4f}")
    print(f"  Cross-datasets (avg):         MSE = {avg_cross_mse:.4f}")
    print(f"  Performance degradation:      {degradation:.1f}%")
    print()

    if degradation < 30:
        print("✓ GOOD GENERALIZATION: Model transfers well to new datasets")
    elif degradation < 50:
        print("⚠ MODERATE GENERALIZATION: Some domain shift observed")
    else:
        print("✗ POOR GENERALIZATION: Significant domain shift")

    print("\n" + "="*70)
    print("Cross-Dataset Evaluation Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

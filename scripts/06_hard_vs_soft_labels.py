#!/usr/bin/env python3
"""
Ablation Study: Hard Labels vs Soft Labels
===========================================

Addresses Professor Schrier's concern:
"I'm not sold on this Gaussian functional form for the soft labels...
why not just predict bin probabilities directly?"

Tests two approaches with IDENTICAL architecture:
1. Hard labels (one-hot encoding) - simpler
2. Soft labels (Gaussian weighting, sigma=1.0) - current approach

Usage:
    python scripts/06_hard_vs_soft_labels.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

from data_utils import load_plm_embeddings
from models import ProteinClassifierNN
from eval_utils import compute_regression_metrics


def train_with_hard_labels():
    """Train model with hard (one-hot) labels."""
    print("\n" + "="*70)
    print("Training with HARD LABELS (one-hot encoding)")
    print("="*70 + "\n")

    # Load best embedding
    X_train, X_test, y_train, y_test = load_plm_embeddings('ginkgo-AA0-650M')

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # Create classifier
    classifier = ProteinClassifierNN(n_bins=10, bin_strategy='quantile')

    # Temporarily replace encode_labels_soft with hard encoding
    original_encode = classifier.encode_labels_soft
    classifier.encode_labels_soft = classifier.encode_labels  # Use hard encoding

    # Train
    print("Training with dropout=0.3...")
    history = classifier.fit(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        architecture='single_layer',
        units=[256, 128],
        dropout_rates=[0.3],
        l2_reg=0.001,
        learning_rate=0.0005,
        batch_size=32,
        epochs=100,
        verbose=0
    )

    # Restore original method
    classifier.encode_labels_soft = original_encode

    # Evaluate
    y_train_pred = classifier.predict(X_train_scaled)
    y_val_pred = classifier.predict(X_val)
    y_test_pred = classifier.predict(X_test_scaled)

    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    val_metrics = compute_regression_metrics(y_val, y_val_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    print(f"\nResults:")
    print(f"  Train MSE: {train_metrics['mse']:.4f}, R²: {train_metrics['r2']:.4f}")
    print(f"  Val   MSE: {val_metrics['mse']:.4f}, R²: {val_metrics['r2']:.4f}")
    print(f"  Test  MSE: {test_metrics['mse']:.4f}, R²: {test_metrics['r2']:.4f}")

    return {
        'approach': 'hard_labels',
        'train_mse': train_metrics['mse'],
        'val_mse': val_metrics['mse'],
        'test_mse': test_metrics['mse'],
        'train_r2': train_metrics['r2'],
        'val_r2': val_metrics['r2'],
        'test_r2': test_metrics['r2'],
        'train_spearman': train_metrics['spearman'],
        'val_spearman': val_metrics['spearman'],
        'test_spearman': test_metrics['spearman'],
        'final_epoch': len(history['loss'])
    }


def train_with_soft_labels():
    """Train model with soft (Gaussian) labels."""
    print("\n" + "="*70)
    print("Training with SOFT LABELS (Gaussian, sigma=1.0)")
    print("="*70 + "\n")

    # Load best embedding
    X_train, X_test, y_train, y_test = load_plm_embeddings('ginkgo-AA0-650M')

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # Create classifier (uses soft labels by default)
    classifier = ProteinClassifierNN(n_bins=10, bin_strategy='quantile')

    # Train
    print("Training with dropout=0.3...")
    history = classifier.fit(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        architecture='single_layer',
        units=[256, 128],
        dropout_rates=[0.3],
        l2_reg=0.001,
        learning_rate=0.0005,
        batch_size=32,
        epochs=100,
        verbose=0
    )

    # Evaluate
    y_train_pred = classifier.predict(X_train_scaled)
    y_val_pred = classifier.predict(X_val)
    y_test_pred = classifier.predict(X_test_scaled)

    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    val_metrics = compute_regression_metrics(y_val, y_val_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    print(f"\nResults:")
    print(f"  Train MSE: {train_metrics['mse']:.4f}, R²: {train_metrics['r2']:.4f}")
    print(f"  Val   MSE: {val_metrics['mse']:.4f}, R²: {val_metrics['r2']:.4f}")
    print(f"  Test  MSE: {test_metrics['mse']:.4f}, R²: {test_metrics['r2']:.4f}")

    return {
        'approach': 'soft_labels',
        'train_mse': train_metrics['mse'],
        'val_mse': val_metrics['mse'],
        'test_mse': test_metrics['mse'],
        'train_r2': train_metrics['r2'],
        'val_r2': val_metrics['r2'],
        'test_r2': test_metrics['r2'],
        'train_spearman': train_metrics['spearman'],
        'val_spearman': val_metrics['spearman'],
        'test_spearman': test_metrics['spearman'],
        'final_epoch': len(history['loss'])
    }


def plot_comparison(results_df):
    """Plot comparison of hard vs soft labels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ('test_mse', 'Test MSE', 'lower is better'),
        ('test_r2', 'Test R²', 'higher is better'),
        ('test_spearman', 'Test Spearman ρ', 'higher is better')
    ]

    colors = {'hard_labels': '#3498db', 'soft_labels': '#e74c3c'}
    labels = {'hard_labels': 'Hard Labels\n(one-hot)', 'soft_labels': 'Soft Labels\n(Gaussian σ=1.0)'}

    for idx, (metric, title, direction) in enumerate(metrics):
        ax = axes[idx]

        # Bar plot
        x_pos = [0, 1]
        values = [results_df[results_df['approach'] == 'hard_labels'][metric].values[0],
                  results_df[results_df['approach'] == 'soft_labels'][metric].values[0]]

        bars = ax.bar(x_pos, values, color=[colors['hard_labels'], colors['soft_labels']],
                      alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels['hard_labels'], labels['soft_labels']])
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n({direction})', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Highlight better approach
        if direction == 'lower is better':
            better_idx = np.argmin(values)
        else:
            better_idx = np.argmax(values)
        bars[better_idx].set_edgecolor('green')
        bars[better_idx].set_linewidth(3)

    plt.tight_layout()

    # Save
    output_path = Path('figures/ablation_hard_vs_soft_labels.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def main():
    """Run ablation study."""
    print("\n" + "="*70)
    print("ABLATION STUDY: Hard Labels vs Soft Labels")
    print("="*70)
    print("\nAddressing Professor Schrier's concern:")
    print('"I\'m not sold on this Gaussian functional form..."')
    print("\nTesting whether soft labels actually improve performance")
    print("over simpler one-hot encoding.\n")

    results = []

    # Test hard labels
    hard_results = train_with_hard_labels()
    results.append(hard_results)

    # Test soft labels
    soft_results = train_with_soft_labels()
    results.append(soft_results)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save results
    output_csv = Path('results/ablation_hard_vs_soft_labels.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved results: {output_csv}")

    # Plot comparison
    plot_comparison(results_df)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    hard_mse = hard_results['test_mse']
    soft_mse = soft_results['test_mse']
    hard_r2 = hard_results['test_r2']
    soft_r2 = soft_results['test_r2']

    print(f"\nTest MSE:")
    print(f"  Hard labels: {hard_mse:.4f}")
    print(f"  Soft labels: {soft_mse:.4f}")
    print(f"  Difference:  {abs(hard_mse - soft_mse):.4f} ({100*abs(hard_mse - soft_mse)/hard_mse:.2f}%)")

    print(f"\nTest R²:")
    print(f"  Hard labels: {hard_r2:.4f}")
    print(f"  Soft labels: {soft_r2:.4f}")
    print(f"  Difference:  {abs(hard_r2 - soft_r2):.4f}")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if abs(hard_mse - soft_mse) < 0.05:  # Less than 5% difference
        print("\n✓ Soft labels provide MINIMAL benefit over hard labels")
        print("  Professor Schrier is RIGHT - simpler approach works just as well")
        print("  Recommendation: Use hard labels for simplicity")
    elif soft_mse < hard_mse:
        improvement = 100 * (hard_mse - soft_mse) / hard_mse
        print(f"\n✓ Soft labels provide {improvement:.1f}% improvement in MSE")
        print("  The Gaussian functional form is justified")
    else:
        print("\n✓ Hard labels actually perform BETTER than soft labels")
        print("  Professor Schrier is RIGHT - the Gaussian form adds complexity without benefit")
        print("  Recommendation: Switch to hard labels")

    print("\n")


if __name__ == '__main__':
    main()

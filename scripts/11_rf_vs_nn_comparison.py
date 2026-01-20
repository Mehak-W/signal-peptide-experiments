#!/usr/bin/env python3
"""
RF vs NN Comparison - Compares Random Forest and Neural Network across all feature types.

Author: Mehak Wadhwa
Advisor: Joshua Schrier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# TensorFlow setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent / 'data'
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models import ProteinClassifierNN


def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    spearman = spearmanr(y_true, y_pred)[0]
    return {'mse': mse, 'r2': r2, 'spearman': spearman}


def train_nn(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train neural network and return predictions and metrics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classifier = ProteinClassifierNN(n_bins=10, bin_strategy='quantile')
    classifier.create_bins(y_train)

    classifier.fit(
        X_train_scaled, y_train,
        X_val=X_val_scaled, y_val=y_val,
        architecture='single_layer',
        units=[256, 128],
        dropout_rates=[0.4],
        l2_reg=0.001,
        learning_rate=0.0005,
        batch_size=32,
        epochs=100,
        verbose=0
    )

    y_pred = classifier.predict(X_test_scaled)
    return y_pred, compute_metrics(y_test, y_pred)


def main():
    print("="*70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*70)

    # Load data
    print("\n[1/6] Loading data...")

    # Physicochemical from correct sheet
    excel_path = DATA_DIR / 'sb2c00328_si_011.xlsx'
    df_lib = pd.read_excel(excel_path, sheet_name='Library_w_Bins_and_WA')

    meta_cols = ['Unnamed: 0', 'index', '#', 'ID', 'Locus', 'gene', 'Library',
                 'SP_nt', 'SP_aa', 'N_nt', 'H_nt', 'C_nt', 'Ac_nt',
                 'N_aa', 'H_aa', 'C_aa', 'Ac_aa', 'WA', 'Set']
    bin_cols = [c for c in df_lib.columns if 'BIN' in c or 'bin' in c.lower()]
    exclude_cols = meta_cols + bin_cols
    feature_cols = [c for c in df_lib.columns if c not in exclude_cols]

    physchem_dict = {}
    for _, row in df_lib.iterrows():
        seq = row['SP_aa']
        features = row[feature_cols].values.astype(float)
        if not np.isnan(features).any():
            physchem_dict[seq] = features

    # Load PLM embeddings
    embeddings = {}
    for emb in ['esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M']:
        df_tr = pd.read_parquet(DATA_DIR / f'trainAA_{emb}.parquet')
        df_te = pd.read_parquet(DATA_DIR / f'testAA_{emb}.parquet')
        embeddings[emb] = {
            'train': {r['sequence']: r['embedding'] for _, r in df_tr.iterrows()},
            'test': {r['sequence']: r['embedding'] for _, r in df_te.iterrows()}
        }

    df_train = pd.read_parquet(DATA_DIR / 'trainAA_esm2-650M.parquet')
    df_test = pd.read_parquet(DATA_DIR / 'testAA_esm2-650M.parquet')

    # Find matched sequences
    train_seqs = list(set(df_train['sequence']) & set(physchem_dict.keys()) &
                      set(embeddings['ginkgo-AA0-650M']['train'].keys()))
    test_seqs = list(set(df_test['sequence']) & set(physchem_dict.keys()) &
                     set(embeddings['ginkgo-AA0-650M']['test'].keys()))

    wa_train = {r['sequence']: r['WA'] for _, r in df_train.iterrows()}
    wa_test = {r['sequence']: r['WA'] for _, r in df_test.iterrows()}
    y_train_full = np.array([wa_train[s] for s in train_seqs])
    y_test = np.array([wa_test[s] for s in test_seqs])

    # Split for validation
    train_idx, val_idx = train_test_split(range(len(train_seqs)), test_size=0.2, random_state=42)
    y_train = y_train_full[train_idx]
    y_val = y_train_full[val_idx]

    print(f"    Train: {len(train_seqs)} sequences")
    print(f"    Test: {len(test_seqs)} sequences")
    print(f"    Features: {len(feature_cols)} physicochemical")

    # Store all results and predictions
    results = {}
    predictions = {}

    feature_configs = [
        ('Physicochemical', physchem_dict, None),
        ('ESM-2 650M', None, 'esm2-650M'),
        ('ESM-2 3B', None, 'esm2-3B'),
        ('Ginkgo AA0', None, 'ginkgo-AA0-650M')
    ]

    # Train all models
    for i, (name, phys_dict, emb_key) in enumerate(feature_configs):
        print(f"\n[{i+2}/6] Training on {name}...")

        if phys_dict:
            X_train_full = np.array([phys_dict[train_seqs[j]] for j in range(len(train_seqs))])
            X_test_data = np.array([phys_dict[s] for s in test_seqs])
        else:
            X_train_full = np.array([embeddings[emb_key]['train'][s] for s in train_seqs])
            X_test_data = np.array([embeddings[emb_key]['test'][s] for s in test_seqs])

        X_train_split = X_train_full[train_idx]
        X_val_split = X_train_full[val_idx]

        # Random Forest
        print(f"    Training RF...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_full, y_train_full)
        rf_pred = rf.predict(X_test_data)
        results[(name, 'RF')] = compute_metrics(y_test, rf_pred)
        predictions[(name, 'RF')] = rf_pred
        print(f"    RF MSE: {results[(name, 'RF')]['mse']:.4f}")

        # Neural Network
        print(f"    Training NN...")
        nn_pred, nn_metrics = train_nn(X_train_split, y_train, X_val_split, y_val, X_test_data, y_test)
        results[(name, 'NN')] = nn_metrics
        predictions[(name, 'NN')] = nn_pred
        print(f"    NN MSE: {results[(name, 'NN')]['mse']:.4f}")

    # =========================================================================
    # Create Figures
    # =========================================================================
    print("\n[6/6] Creating figures...")

    feature_names = ['Physicochemical', 'ESM-2 650M', 'ESM-2 3B', 'Ginkgo AA0']

    # Figure 1: Main comparison bar chart (RF vs NN)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(feature_names))
    width = 0.35

    for ax, (metric, title, better) in zip(axes, [
        ('mse', 'Mean Squared Error (MSE)', 'lower'),
        ('r2', 'R² Score', 'higher'),
        ('spearman', 'Spearman Correlation (ρ)', 'higher')
    ]):
        rf_vals = [results[(ft, 'RF')][metric] for ft in feature_names]
        nn_vals = [results[(ft, 'NN')][metric] for ft in feature_names]

        bars1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest',
                       color='#ff7f0e', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, nn_vals, width, label='Neural Network',
                       color='#1f77b4', edgecolor='black', linewidth=1)

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

        # Highlight best
        all_vals = rf_vals + nn_vals
        best_val = min(all_vals) if better == 'lower' else max(all_vals)
        for bars in [bars1, bars2]:
            for bar in bars:
                if abs(bar.get_height() - best_val) < 0.001:
                    bar.set_edgecolor('green')
                    bar.set_linewidth(3)

        ax.set_ylabel(f'{metric.upper()} ({better} is better)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Model Comparison: RF vs NN Across Feature Types\n(Fair comparison on {len(test_seqs)} matched test sequences)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rf_vs_nn_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: rf_vs_nn_comparison.png")
    plt.close()

    # Figure 2: Prediction scatter plots (best models)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, ft in enumerate(feature_names):
        for row, model in enumerate(['RF', 'NN']):
            ax = axes[row, col]
            pred = predictions[(ft, model)]
            metrics = results[(ft, model)]

            ax.scatter(y_test, pred, alpha=0.5, s=20, c='steelblue')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   'r--', linewidth=2, label='Perfect prediction')

            ax.set_xlabel('True WA' if row == 1 else '', fontsize=9)
            ax.set_ylabel('Predicted WA' if col == 0 else '', fontsize=9)
            ax.set_title(f'{ft}\n{model}: MSE={metrics["mse"]:.3f}, R²={metrics["r2"]:.3f}', fontsize=10)
            ax.grid(True, alpha=0.3)

    fig.suptitle('Prediction Scatter Plots: True vs Predicted WA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: prediction_scatter_plots.png")
    plt.close()

    # Figure 3: Summary heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (metric, title, cmap) in zip(axes, [
        ('mse', 'MSE (lower is better)', 'Reds_r'),
        ('r2', 'R² (higher is better)', 'Greens'),
        ('spearman', 'Spearman ρ (higher is better)', 'Blues')
    ]):
        data = np.array([[results[(ft, m)][metric] for m in ['RF', 'NN']] for ft in feature_names])
        im = ax.imshow(data, cmap=cmap, aspect='auto')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['RF', 'NN'])
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_title(title, fontsize=11, fontweight='bold')

        for i in range(len(feature_names)):
            for j in range(2):
                ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                       fontsize=10, fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Performance Heatmap Summary', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: performance_heatmap.png")
    plt.close()

    # Save results to CSV
    results_list = []
    for (ft, model), metrics in results.items():
        results_list.append({
            'feature_type': ft,
            'model': model,
            'test_mse': metrics['mse'],
            'test_r2': metrics['r2'],
            'test_spearman': metrics['spearman']
        })
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_DIR / 'comprehensive_comparison.csv', index=False)
    print(f"    Saved: comprehensive_comparison.csv")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTest set: {len(test_seqs)} sequences (72% of full Grasso test set)")
    print(f"\n{'Feature Type':<20} {'RF MSE':<10} {'NN MSE':<10} {'Best Model':<12} {'Best MSE':<10}")
    print("-"*62)
    for ft in feature_names:
        rf_mse = results[(ft, 'RF')]['mse']
        nn_mse = results[(ft, 'NN')]['mse']
        best = 'RF' if rf_mse < nn_mse else 'NN'
        best_mse = min(rf_mse, nn_mse)
        print(f"{ft:<20} {rf_mse:<10.4f} {nn_mse:<10.4f} {best:<12} {best_mse:<10.4f}")

    # Find overall best
    best_key = min(results.keys(), key=lambda k: results[k]['mse'])
    print(f"\n✓ Overall best: {best_key[0]} + {best_key[1]} (MSE = {results[best_key]['mse']:.4f})")


if __name__ == '__main__':
    main()

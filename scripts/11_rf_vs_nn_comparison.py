#!/usr/bin/env python3
"""
Correct RF vs NN Comparison - Uses all 1326 test rows (no deduplication).

Author: Mehak Wadhwa
Advisor: Joshua Schrier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

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
        dropout_rates=[0.3],  # Match original
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
    print("CORRECT MODEL COMPARISON - ALL 1326 TEST ROWS")
    print("="*70)

    # Load Excel data (has physicochemical features)
    print("\n[1/6] Loading data...")
    excel = pd.read_excel(DATA_DIR / 'sb2c00328_si_011.xlsx',
                          sheet_name='Library_w_Bins_and_WA')

    excel_train = excel[excel['Set'] == 'Train'].reset_index(drop=True)
    excel_test = excel[excel['Set'] == 'Test'].reset_index(drop=True)

    # Get physicochemical feature columns
    meta_cols = ['Unnamed: 0', 'index', '#', 'ID', 'Locus', 'gene', 'Library',
                 'SP_nt', 'SP_aa', 'N_nt', 'H_nt', 'C_nt', 'Ac_nt',
                 'N_aa', 'H_aa', 'C_aa', 'Ac_aa', 'WA', 'Set']
    bin_cols = [c for c in excel.columns if 'BIN' in c or 'bin' in c.lower() or 'Tot_' in c or 'Perc_' in c]
    exclude_cols = meta_cols + bin_cols
    feature_cols = [c for c in excel.columns if c not in exclude_cols]

    print(f"    Train: {len(excel_train)} rows")
    print(f"    Test: {len(excel_test)} rows")
    print(f"    Physicochemical features: {len(feature_cols)}")

    # Extract physicochemical features (row-aligned, no deduplication!)
    X_train_phys = excel_train[feature_cols].values.astype(float)
    X_test_phys = excel_test[feature_cols].values.astype(float)
    y_train = excel_train['WA'].values
    y_test = excel_test['WA'].values

    # Handle any NaN in physicochemical features
    X_train_phys = np.nan_to_num(X_train_phys, nan=0.0)
    X_test_phys = np.nan_to_num(X_test_phys, nan=0.0)

    # Create validation split from training data (20% like original)
    val_size = int(0.20 * len(X_train_phys))
    indices = np.random.permutation(len(X_train_phys))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    # Load PLM embeddings (already row-aligned with Excel)
    embeddings = {}
    for emb_name in ['esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M']:
        train_df = pd.read_parquet(DATA_DIR / f'trainAA_{emb_name}.parquet')
        test_df = pd.read_parquet(DATA_DIR / f'testAA_{emb_name}.parquet')

        X_train_emb = np.stack(train_df['embedding'].values)
        X_test_emb = np.stack(test_df['embedding'].values)

        embeddings[emb_name] = {
            'X_train': X_train_emb,
            'X_test': X_test_emb
        }

    # Results storage
    results = []
    predictions = {}

    # Feature types to compare
    feature_types = {
        'Physicochemical': {'X_train': X_train_phys, 'X_test': X_test_phys},
        'ESM-2 650M': embeddings['esm2-650M'],
        'ESM-2 3B': embeddings['esm2-3B'],
        'Ginkgo AA0': embeddings['ginkgo-AA0-650M']
    }

    # Train and evaluate each feature type
    for i, (feat_name, data) in enumerate(feature_types.items(), 2):
        print(f"\n[{i}/6] Training on {feat_name}...")

        X_tr = data['X_train']
        X_te = data['X_test']

        # Split for validation
        X_train_split = X_tr[train_idx]
        X_val_split = X_tr[val_idx]
        y_train_split = y_train[train_idx]
        y_val_split = y_train[val_idx]

        # Random Forest
        print(f"    Training RF...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_test_scaled = scaler.transform(X_te)

        rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train_split)
        rf_pred = rf.predict(X_test_scaled)
        rf_metrics = compute_metrics(y_test, rf_pred)
        print(f"    RF MSE: {rf_metrics['mse']:.4f}")

        results.append({
            'feature_type': feat_name,
            'model': 'RF',
            'test_mse': rf_metrics['mse'],
            'test_r2': rf_metrics['r2'],
            'test_spearman': rf_metrics['spearman']
        })
        predictions[f'{feat_name}_RF'] = rf_pred

        # Neural Network
        print(f"    Training NN...")
        nn_pred, nn_metrics = train_nn(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            X_te, y_test
        )
        print(f"    NN MSE: {nn_metrics['mse']:.4f}")

        results.append({
            'feature_type': feat_name,
            'model': 'NN',
            'test_mse': nn_metrics['mse'],
            'test_r2': nn_metrics['r2'],
            'test_spearman': nn_metrics['spearman']
        })
        predictions[f'{feat_name}_NN'] = nn_pred

    # Create figure
    print("\n[6/6] Creating figures...")

    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    feature_order = ['Physicochemical', 'ESM-2 650M', 'ESM-2 3B', 'Ginkgo AA0']
    x = np.arange(len(feature_order))
    width = 0.35

    metrics = [('test_mse', 'MSE (lower is better)'),
               ('test_r2', 'R² (higher is better)'),
               ('test_spearman', 'Spearman ρ (higher is better)')]

    for ax, (metric, title) in zip(axes, metrics):
        rf_vals = [results_df[(results_df['feature_type']==f) & (results_df['model']=='RF')][metric].values[0]
                   for f in feature_order]
        nn_vals = [results_df[(results_df['feature_type']==f) & (results_df['model']=='NN')][metric].values[0]
                   for f in feature_order]

        bars1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, nn_vals, width, label='Neural Network', color='coral', alpha=0.8)

        ax.set_xlabel('Feature Type')
        ax.set_ylabel(metric.replace('test_', '').upper())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(['Physchem', 'ESM-2\n650M', 'ESM-2\n3B', 'Ginkgo\nAA0'])
        ax.legend()

        # Add value labels
        for bar in bars1:
            ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    plt.suptitle(f'RF vs NN Comparison on Full Test Set (n={len(y_test)})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rf_vs_nn_comparison_CORRECTED.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: rf_vs_nn_comparison_CORRECTED.png")

    # Save results
    results_df.to_csv(RESULTS_DIR / 'rf_vs_nn_results_CORRECTED.csv', index=False)
    print(f"    Saved: rf_vs_nn_results_CORRECTED.csv")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTest set: {len(y_test)} rows (100% of Grasso test set)")
    print(f"\n{'Feature Type':<20} {'RF MSE':<10} {'NN MSE':<10} {'Best':<10} {'Best MSE':<10}")
    print("-"*60)

    for feat in feature_order:
        rf_mse = results_df[(results_df['feature_type']==feat) & (results_df['model']=='RF')]['test_mse'].values[0]
        nn_mse = results_df[(results_df['feature_type']==feat) & (results_df['model']=='NN')]['test_mse'].values[0]
        best = 'RF' if rf_mse < nn_mse else 'NN'
        best_mse = min(rf_mse, nn_mse)
        print(f"{feat:<20} {rf_mse:<10.4f} {nn_mse:<10.4f} {best:<10} {best_mse:<10.4f}")

    # Find overall best
    best_row = results_df.loc[results_df['test_mse'].idxmin()]
    print(f"\n✓ Overall best: {best_row['feature_type']} + {best_row['model']} (MSE = {best_row['test_mse']:.4f})")


if __name__ == '__main__':
    main()

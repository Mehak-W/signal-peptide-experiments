#!/usr/bin/env python3
"""
Script 04: Final Comparison

Load results from Scripts 01-03 and produce a grouped bar chart comparing
all models across all feature types on the same test set.

Outputs:
  - results/final_comparison.csv
  - figures/final_comparison.png
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'
GRASSO_BASELINE_MSE = 1.22


def load_results():
    """Load all result JSON files."""
    grasso_path = RESULTS_DIR / 'grasso_reproduction.json'
    rf_path = RESULTS_DIR / 'rf_search_results.json'
    nn_path = RESULTS_DIR / 'nn_search_results.json'

    results = {}

    # Grasso reproduction (Script 01)
    if grasso_path.exists():
        with open(grasso_path) as f:
            results['grasso'] = json.load(f)
    else:
        print(f"[WARN] Missing {grasso_path}")

    # RF search (Script 02)
    if rf_path.exists():
        with open(rf_path) as f:
            results['rf_search'] = json.load(f)
    else:
        print(f"[WARN] Missing {rf_path}")

    # NN search (Script 03)
    if nn_path.exists():
        with open(nn_path) as f:
            results['nn_search'] = json.load(f)
    else:
        print(f"[WARN] Missing {nn_path}")

    return results


def build_comparison_table(results):
    """Build a comparison DataFrame from all results."""
    rows = []

    feature_types = ['PhysChem', 'ESM2-650M', 'ESM2-3B', 'Ginkgo-AA0']

    # Grasso reproduction (PhysChem only)
    if 'grasso' in results:
        g = results['grasso']
        rows.append({
            'Feature Type': 'PhysChem',
            'Model': 'Grasso RF (exact)',
            'Test MSE': g['test_metrics']['mse'],
            'Test RMSE': g['test_metrics']['rmse'],
            'Test MAE': g['test_metrics']['mae'],
            'Test R²': g['test_metrics']['r2'],
            'Spearman ρ': g['test_metrics']['spearman_rho'],
            'Pearson r': g['test_metrics']['pearson_r'],
            'N Test': g['test_metrics']['n_samples'],
        })

    # RF search results (all feature types)
    if 'rf_search' in results:
        for ft in feature_types:
            if ft in results['rf_search']:
                r = results['rf_search'][ft]
                rows.append({
                    'Feature Type': ft,
                    'Model': 'RF (tuned)',
                    'Test MSE': r['test_metrics']['mse'],
                    'Test RMSE': r['test_metrics']['rmse'],
                    'Test MAE': r['test_metrics']['mae'],
                    'Test R²': r['test_metrics']['r2'],
                    'Spearman ρ': r['test_metrics']['spearman_rho'],
                    'Pearson r': r['test_metrics']['pearson_r'],
                    'N Test': r['test_metrics']['n_samples'],
                })

    # NN search results (all feature types)
    if 'nn_search' in results:
        for ft in feature_types:
            if ft in results['nn_search']:
                r = results['nn_search'][ft]
                rows.append({
                    'Feature Type': ft,
                    'Model': 'NN (regression)',
                    'Test MSE': r['test_metrics']['mse'],
                    'Test RMSE': r['test_metrics']['rmse'],
                    'Test MAE': r['test_metrics']['mae'],
                    'Test R²': r['test_metrics']['r2'],
                    'Spearman ρ': r['test_metrics']['spearman_rho'],
                    'Pearson r': r['test_metrics']['pearson_r'],
                    'N Test': r['test_metrics']['n_samples'],
                })

    return pd.DataFrame(rows)


def plot_grouped_bar(df, save_path):
    """
    Grouped bar chart: feature type (x) x model type (groups), MSE on y-axis.
    Horizontal line at Grasso baseline 1.22.
    """
    feature_types = ['PhysChem', 'ESM2-650M', 'ESM2-3B', 'Ginkgo-AA0']
    models = df['Model'].unique().tolist()
    n_models = len(models)

    # Color palette
    colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66']

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.8 / n_models
    x = np.arange(len(feature_types))

    for i, model in enumerate(models):
        mse_values = []
        for ft in feature_types:
            subset = df[(df['Feature Type'] == ft) & (df['Model'] == model)]
            if len(subset) > 0:
                mse_values.append(subset['Test MSE'].values[0])
            else:
                mse_values.append(0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, mse_values, bar_width * 0.9,
                      label=model, color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, mse_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=0)

    # Grasso baseline line
    ax.axhline(y=GRASSO_BASELINE_MSE, color='red', linestyle='--', linewidth=1.5,
               label=f'Grasso baseline (MSE={GRASSO_BASELINE_MSE})')

    ax.set_xlabel('Feature Type', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Signal Peptide Prediction: Model Comparison Across Feature Types', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_types, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    # Start y-axis at 0.95 to emphasize meaningful differences
    ax.set_ylim(0.95, max(df['Test MSE'].max() * 1.05, GRASSO_BASELINE_MSE * 1.15))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results from Scripts 01-03...")
    results = load_results()

    print("\nBuilding comparison table...")
    df = build_comparison_table(results)

    if df.empty:
        print("[ERROR] No results found. Run Scripts 01-03 first.")
        return

    # Print comparison table
    print("\n" + "="*80)
    print("  FINAL COMPARISON")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(df.to_string(index=False, float_format='%.4f'))
    print(f"\n  Grasso baseline MSE: {GRASSO_BASELINE_MSE}")

    # Save CSV
    csv_path = RESULTS_DIR / 'final_comparison.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nCSV saved to {csv_path}")

    # Plot
    fig_path = FIGURES_DIR / 'final_comparison.png'
    plot_grouped_bar(df, fig_path)

    return df


if __name__ == '__main__':
    main()

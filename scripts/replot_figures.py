#!/usr/bin/env python3
"""
Regenerate figures from saved JSON results (no retraining needed).

Fixes: "Dr. Schrier" labels, professional titles.
Regenerates: vector_regression_comparison.png, bootstrap_forest_plot.png,
             vector_architecture_search.png, vector_ensemble_optimization.png
"""
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'

SCHRIER_MSE = 0.953
SCRIPT09_BEST = 0.9732


def replot_06():
    """Regenerate vector_regression_comparison.png from Script 06 results."""
    with open(RESULTS_DIR / 'vector_regression_results.json') as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6))

    emb_names = ['ESM2-650M', 'ESM2-3B', 'Ginkgo-AA0']
    x = np.arange(len(emb_names))
    width = 0.3

    ce_mses, focal_mses = [], []
    for emb_name in emb_names:
        ce_mses.append(results[f"{emb_name}_categorical_crossentropy"]['test_metrics']['mse'])
        focal_mses.append(results[f"{emb_name}_focal"]['test_metrics']['mse'])

    bars_ce = ax.bar(x - width/2, ce_mses, width, label='Cross-Entropy', color='steelblue', alpha=0.85)
    bars_focal = ax.bar(x + width/2, focal_mses, width, label='Focal Loss', color='darkorange', alpha=0.85)

    ax.axhline(y=1.0497, color='gray', linewidth=1.0, linestyle='--', alpha=0.8,
               label='NN Regression Best (1.05)')
    ax.axhline(y=0.95, color='green', linewidth=1.0, linestyle=':', alpha=0.8,
               label='Benchmark (0.953)')

    ax.set_ylabel('Test MSE')
    ax.set_xticks(x)
    ax.set_xticklabels(emb_names)
    ax.legend(loc='upper right')

    ax.set_ylim(0, max(ce_mses + focal_mses) * 1.15)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'vector_regression_comparison.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path.name}")


def replot_08():
    """Regenerate bootstrap_forest_plot.png from Script 08 results."""
    with open(RESULTS_DIR / 'bootstrap_ci_results.json') as f:
        data = json.load(f)

    results = data['results']
    model_order = data['model_order']

    fig, ax = plt.subplots(figsize=(10, 8))

    labels, points, ci_los, ci_his, colors = [], [], [], [], []
    for name in model_order:
        r = results[name]
        labels.append(name)
        points.append(r['mse']['point'])
        ci_los.append(r['mse']['ci_lo'])
        ci_his.append(r['mse']['ci_hi'])
        if name.startswith('Vec'):
            colors.append('darkorange')
        else:
            colors.append('steelblue')

    y_pos = np.arange(len(labels))

    for i in range(len(labels)):
        ax.plot([ci_los[i], ci_his[i]], [y_pos[i], y_pos[i]],
                color=colors[i], linewidth=2, zorder=1)
        ax.plot(points[i], y_pos[i], 'o', color=colors[i],
                markersize=7, zorder=2)

    ax.axvline(x=1.22, color='red', linestyle='--', linewidth=1.0, alpha=0.8,
               label='Baseline (1.22)', zorder=0)
    ax.axvline(x=0.95, color='green', linestyle=':', linewidth=1.0, alpha=0.8,
               label='Benchmark (0.953)', zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Test MSE')

    legend_elements = [
        Line2D([0], [0], marker='o', color='steelblue', label='Scalar models',
               markersize=7, linewidth=2),
        Line2D([0], [0], marker='o', color='darkorange', label='Vector models',
               markersize=7, linewidth=2),
        Line2D([0], [0], color='red', linestyle='--', label='Baseline (1.22)'),
        Line2D([0], [0], color='green', linestyle=':', label='Benchmark (0.953)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'bootstrap_forest_plot.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path.name}")


def replot_09():
    """Regenerate vector_architecture_search.png from Script 09 results."""
    with open(RESULTS_DIR / 'vector_architecture_search_results.json') as f:
        all_results = json.load(f)

    fig, ax = plt.subplots(figsize=(14, 7))

    labels, mses, colors = [], [], []

    p1 = all_results.get('phase1', {})
    for name, r in p1.items():
        labels.append(f"P1: {name}")
        mses.append(r['mse'])
        colors.append('steelblue')

    p2 = all_results.get('phase2', {})
    if '5_seed' in p2:
        labels.append('P2: 5-seed')
        mses.append(p2['5_seed']['mse'])
        colors.append('darkorange')
    if '10_seed' in p2:
        labels.append('P2: 10-seed')
        mses.append(p2['10_seed']['mse'])
        colors.append('darkorange')

    p3 = all_results.get('phase3', {})
    sorted_p3 = sorted(p3.items(), key=lambda x: x[1]['mse'])
    for name, r in sorted_p3[:5]:
        labels.append(f"P3: {name}")
        mses.append(r['mse'])
        colors.append('mediumseagreen')

    p4 = all_results.get('phase4', {})
    for name, r in p4.items():
        labels.append(f"P4: {name}")
        mses.append(r['mse'])
        colors.append('crimson')

    x = np.arange(len(labels))
    bars = ax.bar(x, mses, color=colors, alpha=0.85)

    ax.axhline(y=SCHRIER_MSE, color='green', linestyle=':', linewidth=1.0, alpha=0.8,
               label=f'Benchmark ({SCHRIER_MSE})')
    ax.axhline(y=1.001, color='gray', linestyle='--', linewidth=1.0, alpha=0.8,
               label='Val-split best (1.001)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Test MSE (all 1326 samples)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'vector_architecture_search.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path.name}")


def replot_10():
    """Regenerate vector_ensemble_optimization.png from Script 10 results."""
    with open(RESULTS_DIR / 'vector_ensemble_optimization_results.json') as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(14, 7))

    labels, mses, colors = [], [], []

    for name, r in results.get('phase1_dropout', {}).items():
        labels.append(f"P1: {name}")
        mses.append(r['mse'])
        colors.append('steelblue')

    for name, r in results.get('phase2_4layer', {}).items():
        labels.append(f"P2: {name}")
        mses.append(r['mse'])
        colors.append('darkorange')

    for name, r in results.get('phase3_cosine', {}).items():
        labels.append(f"P3: {name}")
        mses.append(r['mse'])
        colors.append('mediumseagreen')

    for name, r in results.get('phase4_large_ensemble', {}).items():
        labels.append(f"P4: {name}")
        mses.append(r['mse'])
        colors.append('crimson')

    for name, r in results.get('phase5_mixed', {}).items():
        labels.append(f"P5: {name}")
        mses.append(r['mse'])
        colors.append('purple')

    x = np.arange(len(labels))
    bars = ax.bar(x, mses, color=colors, alpha=0.85)

    ax.axhline(y=SCHRIER_MSE, color='green', linestyle=':', linewidth=1.0, alpha=0.8,
               label=f'Benchmark ({SCHRIER_MSE})')
    ax.axhline(y=SCRIPT09_BEST, color='gray', linestyle='--', linewidth=1.0, alpha=0.8,
               label=f'Arch search best ({SCRIPT09_BEST})')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Test MSE (all 1326 samples)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'vector_ensemble_optimization.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path.name}")


if __name__ == '__main__':
    print("Regenerating figures from saved results...")
    replot_06()
    replot_08()
    replot_09()
    replot_10()
    print("Done! All 4 figures updated.")

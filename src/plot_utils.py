"""
Plotting Utilities
Shared functions for creating publication-quality figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None,
    show_metrics: bool = True
) -> plt.Figure:
    """
    Create scatter plot of predictions vs actual values.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    show_metrics : bool
        Whether to display metrics on plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    from sklearn.metrics import r2_score, mean_squared_error

    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

    # Labels and title
    ax.set_xlabel('Actual WA Score', fontsize=11)
    ax.set_ylabel('Predicted WA Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add metrics text box
    if show_metrics:
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        textstr = f'$R^2$ = {r2:.3f}\nMSE = {mse:.3f}\nRMSE = {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create residual plot to assess model fit.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)

    ax.set_xlabel('Predicted WA Score', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'test_mse',
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar plot comparing multiple models.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns ['model', metric]
    metric : str
        Metric to plot
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if title is None:
        title = f"Model Comparison: {metric.upper()}"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by metric value
    results_df = results_df.sort_values(metric)

    # Create bar plot
    bars = ax.barh(range(len(results_df)), results_df[metric])

    # Color bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_df)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Labels
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df['model'])
    ax.set_xlabel(metric.upper(), fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add value labels on bars
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax.text(row[metric], i, f' {row[metric]:.3f}',
                va='center', fontsize=9)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_training_history(
    history: dict,
    metrics: List[str] = ['loss'],
    title: str = "Training History",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history for neural networks.

    Parameters
    ----------
    history : dict
        Training history from Keras model.fit()
    metrics : list
        List of metrics to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # Plot training metric
        ax.plot(history[metric], label=f'Train {metric}', linewidth=2)

        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(f'{metric.upper()} over Epochs', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_embedding_comparison(
    results_dict: dict,
    metric: str = 'test_mse',
    title: str = "PLM Embedding Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare performance across different PLM embeddings.

    Parameters
    ----------
    results_dict : dict
        Dictionary with embedding names as keys and metrics as values
    metric : str
        Metric to compare
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    embeddings = list(results_dict.keys())
    values = [results_dict[emb][metric] for emb in embeddings]

    # Create bar plot
    bars = ax.bar(range(len(embeddings)), values, width=0.6)

    # Color bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)

    # Labels
    ax.set_xticks(range(len(embeddings)))
    ax.set_xticklabels(embeddings, rotation=45, ha='right')
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add value labels on bars
    for i, (emb, val) in enumerate(zip(embeddings, values)):
        ax.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig

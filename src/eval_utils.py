"""
Model Evaluation Utilities
Shared functions for computing metrics and generating evaluation reports
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Tuple, Optional


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    prefix : str
        Prefix for metric names (e.g., 'train_' or 'test_')

    Returns
    -------
    metrics : dict
        Dictionary of metric names and values
    """
    metrics = {}

    # Core regression metrics
    metrics[f'{prefix}mse'] = mean_squared_error(y_true, y_pred)
    metrics[f'{prefix}rmse'] = np.sqrt(metrics[f'{prefix}mse'])
    metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
    metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)

    # Correlation metrics
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)

    metrics[f'{prefix}spearman'] = spearman_corr
    metrics[f'{prefix}spearman_pval'] = spearman_p
    metrics[f'{prefix}pearson'] = pearson_corr
    metrics[f'{prefix}pearson_pval'] = pearson_p

    return metrics


def print_metrics_report(
    metrics: Dict[str, float],
    title: str = "Model Performance"
) -> None:
    """
    Print a formatted metrics report.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values
    title : str
        Title for the report
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    # Group metrics by type
    for key, value in sorted(metrics.items()):
        if 'pval' in key:
            print(f"  {key:25s}: {value:.4e}")
        else:
            print(f"  {key:25s}: {value:.4f}")

    print(f"{'='*60}\n")


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a trained model on train and test sets.

    Parameters
    ----------
    model : sklearn estimator
        Trained model with predict method
    X_train, y_train : np.ndarray
        Training features and labels
    X_test, y_test : np.ndarray
        Test features and labels
    model_name : str
        Name of the model for reporting

    Returns
    -------
    metrics : dict
        Combined train and test metrics
    y_train_pred : np.ndarray
        Training predictions
    y_test_pred : np.ndarray
        Test predictions
    """
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute metrics
    train_metrics = compute_regression_metrics(y_train, y_train_pred, prefix='train_')
    test_metrics = compute_regression_metrics(y_test, y_test_pred, prefix='test_')

    # Combine metrics
    metrics = {**train_metrics, **test_metrics}

    # Print report
    print_metrics_report(metrics, title=f"{model_name} Performance")

    return metrics, y_train_pred, y_test_pred


def check_reproduction_accuracy(
    test_mse: float,
    target_mse: float = 1.22,
    tolerance: float = 0.1
) -> bool:
    """
    Check if reproduction meets target accuracy.

    Parameters
    ----------
    test_mse : float
        Achieved test MSE
    target_mse : float
        Target MSE from original paper (default: 1.22 from Grasso et al.)
    tolerance : float
        Acceptable deviation from target

    Returns
    -------
    success : bool
        Whether reproduction was successful
    """
    deviation = abs(test_mse - target_mse)
    reproduction_accuracy = (1 - deviation / target_mse) * 100

    print(f"\n{'='*60}")
    print(f"{'Reproduction Validation':^60}")
    print(f"{'='*60}")
    print(f"  Target MSE (Grasso et al.):  {target_mse:.3f}")
    print(f"  Achieved MSE:                {test_mse:.3f}")
    print(f"  Deviation:                   {deviation:.3f}")
    print(f"  Reproduction Accuracy:       {reproduction_accuracy:.1f}%")

    if deviation <= tolerance:
        print(f"  Status:                      ✓ SUCCESS")
        success = True
    else:
        print(f"  Status:                      ✗ NEEDS IMPROVEMENT")
        success = False

    print(f"{'='*60}\n")

    return success


def create_results_dataframe(
    metrics: Dict[str, float],
    model_name: str,
    embedding_model: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a formatted DataFrame from metrics for saving results.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics
    model_name : str
        Name of the model
    embedding_model : str, optional
        Name of the embedding model if applicable

    Returns
    -------
    df : pd.DataFrame
        Results DataFrame
    """
    results = {'model': model_name}

    if embedding_model:
        results['embedding_model'] = embedding_model

    results.update(metrics)

    df = pd.DataFrame([results])

    return df

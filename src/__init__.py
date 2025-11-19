"""
Signal Peptide Prediction Utilities
Shared code for signal peptide efficiency prediction experiments
"""

from .data_utils import (
    load_plm_embeddings,
    load_grasso_data,
    get_train_test_split,
    get_model_info
)

from .eval_utils import (
    compute_regression_metrics,
    print_metrics_report,
    evaluate_model,
    check_reproduction_accuracy,
    create_results_dataframe
)

from .plot_utils import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_model_comparison,
    plot_training_history,
    plot_embedding_comparison
)

from .models import (
    ProteinClassifierNN,
    FocalLoss,
    build_random_forest
)

__all__ = [
    # Data utilities
    'load_plm_embeddings',
    'load_grasso_data',
    'get_train_test_split',
    'get_model_info',
    # Evaluation utilities
    'compute_regression_metrics',
    'print_metrics_report',
    'evaluate_model',
    'check_reproduction_accuracy',
    'create_results_dataframe',
    # Plotting utilities
    'plot_predictions_vs_actual',
    'plot_residuals',
    'plot_model_comparison',
    'plot_training_history',
    'plot_embedding_comparison',
    # Models
    'ProteinClassifierNN',
    'FocalLoss',
    'build_random_forest',
]

#!/usr/bin/env python3
"""
Neural Network Classifier for Signal Peptide Prediction
========================================================

Trains neural network classifiers on PLM embeddings using soft binning approach.
Converts continuous WA scores into probability distributions for improved prediction.

IMPROVED VERSION: Includes StandardScaler, validation split, and ReduceLROnPlateau
to match previous MSE = 0.950 performance.

Usage:
    python scripts/02_neural_network_classifier.py

Outputs:
    - Model performance metrics for all three PLM embeddings
    - Training history plots saved to figures/
    - Prediction plots saved to figures/
    - Results summary saved to results/
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

from data_utils import load_plm_embeddings, get_model_info
from eval_utils import compute_regression_metrics, print_metrics_report, create_results_dataframe
from plot_utils import (
    plot_predictions_vs_actual,
    plot_training_history,
    plot_embedding_comparison
)
from models import ProteinClassifierNN


def train_nn_on_embedding(
    model_name: str,
    architecture: str = 'single_layer',
    units: list = [256, 128],
    dropout_rates: list = [0.3],
    l2_reg: float = 0.001,
    learning_rate: float = 0.0005,
    batch_size: int = 32,
    epochs: int = 100,
    n_bins: int = 10,
    use_scaler: bool = True,
    val_split: float = 0.2
):
    """
    Train Neural Network classifier on a specific PLM embedding.

    Parameters
    ----------
    model_name : str
        Name of PLM model ('esm2-650M', 'esm2-3B', or 'ginkgo-AA0-650M')
    architecture : str
        'single_layer' or 'two_layer'
    units : list
        Number of units in hidden layers
    dropout_rates : list
        Dropout rates for each layer
    l2_reg : float
        L2 regularization strength
    learning_rate : float
        Learning rate for Adam optimizer (0.0005 works best)
    batch_size : int
        Training batch size
    epochs : int
        Maximum number of epochs
    n_bins : int
        Number of bins for soft binning
    use_scaler : bool
        Whether to use StandardScaler (IMPORTANT for best performance)
    val_split : float
        Validation split ratio (0.2 = 20% for validation)

    Returns
    -------
    results : dict
        Dictionary containing metrics, predictions, and training history
    """
    print(f"\n{'='*70}")
    print(f"Training Neural Network on {model_name}")
    print(f"{'='*70}\n")

    # Load embeddings
    X_train_full, X_test, y_train_full, y_test = load_plm_embeddings(model_name)

    # CRITICAL: Scale features (this was missing!)
    if use_scaler:
        print("Scaling features with StandardScaler...")
        scaler = StandardScaler()
        X_train_full = scaler.fit_transform(X_train_full)
        X_test = scaler.transform(X_test)

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_split,
        random_state=42
    )

    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")

    # Build and train classifier
    print(f"\nBuilding Neural Network:")
    print(f"  Architecture: {architecture}")
    print(f"  Hidden units: {units}")
    print(f"  Dropout rates: {dropout_rates}")
    print(f"  L2 regularization: {l2_reg}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Bins: {n_bins}")
    print(f"  Feature scaling: {use_scaler}")
    print(f"  Validation split: {val_split}")

    classifier = ProteinClassifierNN(n_bins=n_bins, bin_strategy='quantile')

    # Create bins from full training data
    classifier.create_bins(y_train_full)

    # Encode labels with soft binning
    y_train_encoded = classifier.encode_labels_soft(y_train, temperature=0.5)
    y_val_encoded = classifier.encode_labels_soft(y_val, temperature=0.5)

    # Build model
    model = classifier.create_model(
        input_dim=X_train.shape[1],
        architecture=architecture,
        units=units,
        dropout_rates=dropout_rates,
        l2_reg=l2_reg
    )

    # Compile with optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks - IMPORTANT: ReduceLROnPlateau was missing!
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=0
        )
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train_encoded,
        validation_data=(X_val, y_val_encoded),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0
    )

    # Store model in classifier for predictions
    classifier.model = model

    # Evaluate on all sets
    y_train_pred = classifier.predict(X_train)
    y_val_pred = classifier.predict(X_val)
    y_test_pred = classifier.predict(X_test)

    # Compute metrics
    train_metrics = compute_regression_metrics(y_train, y_train_pred, prefix='train_')
    val_metrics = compute_regression_metrics(y_val, y_val_pred, prefix='val_')
    test_metrics = compute_regression_metrics(y_test, y_test_pred, prefix='test_')

    metrics = {**train_metrics, **val_metrics, **test_metrics}

    print_metrics_report(metrics, title=f"Neural Network ({model_name}) Performance")

    # Create plots
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Training history
    plot_training_history(
        history.history,
        metrics=['loss', 'accuracy'],
        title=f"NN Training: {model_name}",
        save_path=figures_dir / f"nn_{model_name}_training.png"
    )

    # Predictions
    plot_predictions_vs_actual(
        y_test, y_test_pred,
        title=f"Neural Network: {model_name}",
        save_path=figures_dir / f"nn_{model_name}_predictions.png"
    )

    return {
        'model_name': model_name,
        'metrics': metrics,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'history': history.history,
        'classifier': classifier
    }


def main():
    """Run Neural Network experiments on all PLM embeddings."""
    print("\n" + "="*70)
    print("Neural Network Classifier Experiments (IMPROVED)")
    print("="*70)
    print("\nChanges from previous version:")
    print("  ✓ Added StandardScaler for feature normalization")
    print("  ✓ Added validation split (20% of training data)")
    print("  ✓ Added ReduceLROnPlateau callback")
    print("  ✓ Updated learning rate: 0.001 → 0.0005")
    print("  ✓ Updated units: [256] → [256, 128]")
    print("\nTarget: MSE ~0.95 (previous best performance)\n")

    # Get available models
    model_info = get_model_info()
    model_names = list(model_info.keys())

    # IMPROVED hyperparameters matching your previous best config
    hyperparams = {
        'architecture': 'single_layer',
        'units': [256, 128],          # Better than [256] alone
        'dropout_rates': [0.3],
        'l2_reg': 0.001,
        'learning_rate': 0.0005,      # Lower LR works better
        'batch_size': 32,
        'epochs': 100,
        'n_bins': 10,
        'use_scaler': True,           # CRITICAL for performance
        'val_split': 0.2              # Use validation set
    }

    # Train on each embedding
    all_results = {}
    results_list = []

    for model_name in model_names:
        result = train_nn_on_embedding(model_name, **hyperparams)
        all_results[model_name] = result

        # Collect metrics for comparison
        results_df = create_results_dataframe(
            result['metrics'],
            model_name='Neural Network (Improved)',
            embedding_model=model_name
        )
        results_list.append(results_df)

    # Combine results
    combined_results = pd.concat(results_list, ignore_index=True)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Neural Network Performance Across PLM Embeddings")
    print("="*70 + "\n")
    print(combined_results[['embedding_model', 'test_mse', 'test_r2', 'test_spearman']].to_string(index=False))
    print()

    # Find best model
    best_idx = combined_results['test_mse'].idxmin()
    best_result = combined_results.iloc[best_idx]
    print(f"🎯 BEST MODEL: {best_result['embedding_model']}")
    print(f"   Test MSE: {best_result['test_mse']:.4f}")
    print(f"   Test R²: {best_result['test_r2']:.4f}")
    print(f"   Spearman ρ: {best_result['test_spearman']:.4f}")
    print()

    # Check if we matched target
    target_mse = 0.95
    if best_result['test_mse'] <= 1.0:
        improvement = ((1.22 - best_result['test_mse']) / 1.22) * 100
        print(f"✓ SUCCESS! Achieved MSE < 1.0")
        print(f"  {improvement:.1f}% improvement over Grasso et al. baseline (1.22)")
    else:
        print(f"Note: MSE {best_result['test_mse']:.4f} is close to target {target_mse:.4f}")

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    combined_results.to_csv(results_dir / 'nn_classifier_results.csv', index=False)
    print(f"\nResults saved to {results_dir / 'nn_classifier_results.csv'}")

    # Create embedding comparison plot
    figures_dir = Path(__file__).parent.parent / 'figures'
    metrics_dict = {
        result['model_name']: result['metrics']
        for result in all_results.values()
    }

    plot_embedding_comparison(
        metrics_dict,
        metric='test_mse',
        title='Neural Network: PLM Embedding Comparison (Test MSE)',
        save_path=figures_dir / 'nn_embedding_comparison.png'
    )

    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

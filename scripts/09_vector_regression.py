#!/usr/bin/env python3
"""
Vector Regression Architecture for Signal Peptide Prediction
=============================================================

New approach suggested by Professor Schrier:
Instead of predicting WA directly (or via soft labels), predict the 10-dimensional
bin probability vector directly, then convert to WA via dot product.

Advantages:
- Simpler and matches problem structure better
- Directly models the experimental bin distribution
- WA is just a weighted average: probs @ bins (bins = range(1, 11))
- Enables lightweight fine-tuning for other dataset output types

Reference: https://github.com/mfbliposome/signal_peptides
Professor's baseline: MSE = 0.95 with unoptimized hyperparameters

Usage:
    python scripts/09_vector_regression.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

from data_utils import load_plm_embeddings
from eval_utils import compute_regression_metrics, print_metrics_report


def load_data_with_bin_distributions(model_name: str):
    """
    Load PLM embeddings along with bin probability distributions.

    Returns:
    --------
    X_train, X_test : embeddings
    y_train_bins, y_test_bins : 10-dimensional bin probability vectors
    y_train_wa, y_test_wa : weighted average scores (for evaluation)
    """
    print(f"\nLoading data for {model_name}...")

    # Load embeddings (same as before)
    X_train_full, X_test, y_train_wa_full, y_test_wa = load_plm_embeddings(model_name)

    # Load bin distributions from parquet (we just added these!)
    data_dir = Path(__file__).parent.parent / 'data'
    train_df = pd.read_parquet(data_dir / f'trainAA_{model_name}.parquet')
    test_df = pd.read_parquet(data_dir / f'testAA_{model_name}.parquet')

    # Extract bin columns
    bin_cols = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]

    y_train_bins_full = train_df[bin_cols].values
    y_test_bins = test_df[bin_cols].values

    print(f"  Train: {X_train_full.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Bin distribution shape: {y_train_bins_full.shape[1]}-dimensional")

    # Check for NaN values in bin distributions
    train_nan_count = np.isnan(y_train_bins_full).any(axis=1).sum()
    test_nan_count = np.isnan(y_test_bins).any(axis=1).sum()

    if train_nan_count > 0 or test_nan_count > 0:
        print(f"  WARNING: {train_nan_count} train, {test_nan_count} test samples have NaN bins")
        print(f"  These will be filtered out")

        # Filter out NaN rows
        train_valid = ~np.isnan(y_train_bins_full).any(axis=1)
        test_valid = ~np.isnan(y_test_bins).any(axis=1)

        X_train_full = X_train_full[train_valid]
        y_train_bins_full = y_train_bins_full[train_valid]
        y_train_wa_full = y_train_wa_full[train_valid]

        X_test = X_test[test_valid]
        y_test_bins = y_test_bins[test_valid]
        y_test_wa = y_test_wa[test_valid]

        print(f"  After filtering: {X_train_full.shape[0]} train, {X_test.shape[0]} test")

    # Normalize bin probabilities to sum to 1 (they should already, but let's be safe)
    y_train_bins_full = y_train_bins_full / y_train_bins_full.sum(axis=1, keepdims=True)
    y_test_bins = y_test_bins / y_test_bins.sum(axis=1, keepdims=True)

    return X_train_full, X_test, y_train_bins_full, y_test_bins, y_train_wa_full, y_test_wa


def create_vector_regression_model(
    input_dim: int,
    hidden_units: list = [256, 128],
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.0005
):
    """
    Create neural network that outputs 10-dimensional probability vector.

    Architecture:
    - Input: PLM embedding (1280 or 2560 dim)
    - Hidden layers with dropout and L2 regularization
    - Output: 10-dimensional probability vector (softmax activation)

    Loss: Categorical Cross-Entropy (measures how well predicted distribution matches true distribution)
    """
    print("\nBuilding Vector Regression Model:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden units: {hidden_units}")
    print(f"  Dropout: {dropout_rate}")
    print(f"  L2 regularization: {l2_reg}")
    print(f"  Learning rate: {learning_rate}")

    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    # Hidden layers
    for units in hidden_units:
        model.add(tf.keras.layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # Output layer: 10 bins with softmax (probability distribution)
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Compile with categorical cross-entropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',  # KL divergence between predicted and true distributions
        metrics=['accuracy', 'categorical_crossentropy']
    )

    print(f"\nModel summary:")
    model.summary()

    return model


def convert_bins_to_wa(bin_probs: np.ndarray) -> np.ndarray:
    """
    Convert bin probability distribution to WA score.

    WA = weighted average = dot product of probabilities with bin numbers
    WA = probs @ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Parameters:
    -----------
    bin_probs : np.ndarray, shape (n_samples, 10)
        Probability distribution over 10 bins

    Returns:
    --------
    wa : np.ndarray, shape (n_samples,)
        Weighted average scores
    """
    bins = np.arange(1, 11)  # [1, 2, 3, ..., 10]
    wa = bin_probs @ bins
    return wa


def train_vector_regression(
    model_name: str,
    hidden_units: list = [256, 128],
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.0005,
    batch_size: int = 32,
    epochs: int = 100,
    val_split: float = 0.2
):
    """
    Train vector regression model on PLM embeddings.

    Returns:
    --------
    results : dict with model, metrics, predictions, history
    """
    print(f"\n{'='*70}")
    print(f"Vector Regression: {model_name}")
    print(f"{'='*70}")

    # Load data with bin distributions
    X_train_full, X_test, y_train_bins_full, y_test_bins, y_train_wa_full, y_test_wa = \
        load_data_with_bin_distributions(model_name)

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test)

    # Split training data for validation
    X_train, X_val, y_train_bins, y_val_bins, y_train_wa, y_val_wa = train_test_split(
        X_train_full, y_train_bins_full, y_train_wa_full,
        test_size=val_split,
        random_state=42
    )

    print(f"\nData splits:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Create model
    model = create_vector_regression_model(
        input_dim=X_train.shape[1],
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        learning_rate=learning_rate
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train_bins,
        validation_data=(X_val, y_val_bins),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Predict bin probabilities
    print("\nGenerating predictions...")
    y_train_bins_pred = model.predict(X_train, verbose=0)
    y_val_bins_pred = model.predict(X_val, verbose=0)
    y_test_bins_pred = model.predict(X_test, verbose=0)

    # Convert to WA scores
    y_train_wa_pred = convert_bins_to_wa(y_train_bins_pred)
    y_val_wa_pred = convert_bins_to_wa(y_val_bins_pred)
    y_test_wa_pred = convert_bins_to_wa(y_test_bins_pred)

    # Compute metrics on WA (for comparison with other methods)
    train_metrics = compute_regression_metrics(y_train_wa, y_train_wa_pred, prefix='train_')
    val_metrics = compute_regression_metrics(y_val_wa, y_val_wa_pred, prefix='val_')
    test_metrics = compute_regression_metrics(y_test_wa, y_test_wa_pred, prefix='test_')

    metrics = {**train_metrics, **val_metrics, **test_metrics}

    print_metrics_report(metrics, title=f"Vector Regression ({model_name}) - WA Metrics")

    # Create plots
    create_plots(
        model_name, history, y_test_wa, y_test_wa_pred, y_test_bins, y_test_bins_pred, metrics
    )

    return {
        'model_name': model_name,
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'history': history.history,
        'y_test_wa': y_test_wa,
        'y_test_wa_pred': y_test_wa_pred,
        'y_test_bins': y_test_bins,
        'y_test_bins_pred': y_test_bins_pred
    }


def create_plots(model_name, history, y_test_wa, y_test_wa_pred, y_test_bins, y_test_bins_pred, metrics):
    """Create visualization plots"""
    figures_dir = Path(__file__).parent.parent / 'figures'

    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history.history['loss'], label='Train', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Cross-Entropy Loss', fontweight='bold')
    ax.set_title(f'Training History: {model_name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history.history.get('accuracy', []), label='Train', linewidth=2)
    ax.plot(history.history.get('val_accuracy', []), label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title(f'Accuracy: {model_name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / f'vector_reg_{model_name}_training.png', dpi=300, bbox_inches='tight')
    plt.close()

    # WA Predictions vs Actual
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(y_test_wa, y_test_wa_pred, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    ax.plot([y_test_wa.min(), y_test_wa.max()], [y_test_wa.min(), y_test_wa.max()],
            'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual WA', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted WA', fontweight='bold', fontsize=12)
    ax.set_title(f'Vector Regression: {model_name}\nTest MSE = {metrics["test_mse"]:.4f}, R² = {metrics["test_r2"]:.4f}',
                 fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / f'vector_reg_{model_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Plots saved to figures/")


def main():
    """Run vector regression experiments"""
    print("\n" + "="*70)
    print("Vector Regression Architecture")
    print("="*70)
    print("\nKey differences from soft-label approach:")
    print("  - Output: 10-dimensional probability vector (softmax)")
    print("  - Loss: Categorical Cross-Entropy")
    print("  - WA computation: dot product with bin numbers [1..10]")
    print("\nAdvantages:")
    print("  - Simpler and matches problem structure better")
    print("  - Directly models experimental bin distributions")
    print("  - Enables lightweight fine-tuning for other datasets")
    print()

    # Available models
    model_names = ['esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M']

    # Hyperparameters (using same as soft-label for fair comparison)
    hyperparams = {
        'hidden_units': [256, 128],
        'dropout_rate': 0.3,  # Using VALIDATION-based choice (fixed earlier)
        'l2_reg': 0.001,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'epochs': 100,
        'val_split': 0.2
    }

    # Train on each embedding
    all_results = []
    for model_name in model_names:
        result = train_vector_regression(model_name, **hyperparams)
        all_results.append({
            'model': 'Vector Regression',
            'embedding_model': model_name,
            **result['metrics']
        })

    # Combine results
    results_df = pd.DataFrame(all_results)

    # Print comparison
    print("\n" + "="*70)
    print("SUMMARY: Vector Regression Performance")
    print("="*70 + "\n")
    print(results_df[['embedding_model', 'test_mse', 'test_r2', 'test_spearman']].to_string(index=False))

    # Find best
    best_idx = results_df['test_mse'].idxmin()
    best = results_df.iloc[best_idx]
    print(f"\n🎯 BEST MODEL: {best['embedding_model']}")
    print(f"   Test MSE: {best['test_mse']:.4f}")
    print(f"   Test R²: {best['test_r2']:.4f}")
    print(f"   Spearman ρ: {best['test_spearman']:.4f}")

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_df.to_csv(results_dir / 'vector_regression_results.csv', index=False)
    print(f"\n✓ Results saved to {results_dir / 'vector_regression_results.csv'}")

    # Compare to soft-label approach
    print("\n" + "="*70)
    print("Comparison: Vector Regression vs Soft-Label NN")
    print("="*70 + "\n")

    try:
        soft_label_results = pd.read_csv(results_dir / 'nn_classifier_results.csv')
        print("Soft-Label NN (previous approach):")
        print(soft_label_results[['embedding_model', 'test_mse', 'test_r2', 'test_spearman']].to_string(index=False))

        print("\nVector Regression (new approach):")
        print(results_df[['embedding_model', 'test_mse', 'test_r2', 'test_spearman']].to_string(index=False))

        print("\nImprovement (Vector - SoftLabel, negative is better for MSE):")
        for emb in model_names:
            soft_mse = soft_label_results[soft_label_results['embedding_model'] == emb]['test_mse'].values[0]
            vec_mse = results_df[results_df['embedding_model'] == emb]['test_mse'].values[0]
            improvement = vec_mse - soft_mse
            symbol = "✓" if improvement < 0 else "✗"
            print(f"  {emb:20s}: {improvement:+.4f} {symbol}")

    except FileNotFoundError:
        print("Could not find soft-label results for comparison")

    print("\n" + "="*70)
    print("Vector Regression Complete!")
    print("="*70)


if __name__ == '__main__':
    main()

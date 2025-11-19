#!/usr/bin/env python3
"""
Quick Test Script
=================

Validates that the entire setup is working correctly by:
1. Testing data loading
2. Training a small Random Forest model
3. Verifying evaluation utilities
4. Creating a test plot

Usage:
    python scripts/00_quick_test.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data_utils import load_plm_embeddings, get_model_info
from eval_utils import evaluate_model
from plot_utils import plot_predictions_vs_actual


def main():
    print("\n" + "="*70)
    print("Quick Test: Validating Setup")
    print("="*70 + "\n")

    # 1. Test data loading
    print("1. Testing data loading...")
    model_info = get_model_info()
    print(f"   Found {len(model_info)} PLM models: {', '.join(model_info.keys())}")

    X_train, X_test, y_train, y_test = load_plm_embeddings('esm2-650M')
    print(f"   ✓ Data loaded successfully")
    print()

    # 2. Train a small model
    print("2. Training small Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=10,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"   ✓ Model trained successfully")
    print()

    # 3. Evaluate
    print("3. Testing evaluation utilities...")
    metrics, y_train_pred, y_test_pred = evaluate_model(
        model, X_train, y_train, X_test, y_test,
        model_name="Quick Test RF"
    )
    print(f"   ✓ Evaluation completed")
    print()

    # 4. Create plot
    print("4. Creating test plot...")
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    plot_predictions_vs_actual(
        y_test, y_test_pred,
        title="Quick Test: Predictions vs Actual",
        save_path=figures_dir / "quick_test_predictions.png"
    )
    print(f"   ✓ Plot saved to {figures_dir / 'quick_test_predictions.png'}")
    print()

    # Summary
    print("="*70)
    print("Setup Validation Complete!")
    print("="*70)
    print("\nAll systems operational. You can now run the main experiments:")
    print("  - scripts/01_random_forest_baseline.py")
    print("  - scripts/02_neural_network_classifier.py")
    print("  - scripts/03_model_comparison.py")
    print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script 13: Cross-Dataset Fine-Tuning with Vector Model

Implements Professor Schrier's suggestion: pretrain the best vector regression
model on Grasso data, then fine-tune the output layer(s) on each external
dataset using 5-fold cross-validation.

Strategy:
  1. Pretrain 5-seed vector NN ensemble on full Grasso training data (ESM2-650M)
  2. For each external dataset:
     a. 5-fold CV (stratified for binary Wu dataset)
     b. Per fold: freeze feature layers, fine-tune last dense layers
     c. For Wu (binary): replace softmax(10) output with sigmoid(1)
     d. For continuous datasets: keep softmax(10) → convert to WA via dot product
  3. Report fold-level and aggregated metrics
  4. Compare against zero-shot baseline from Script 05

External datasets:
  - Wu:          81 SPs, binary (0/1), B. subtilis
  - Xue:        322 SPs, continuous (0–10437)
  - Zhang-P43:  114 SPs, continuous (0–193)
  - Zhang-PglVM: 114 SPs, continuous (0–327)
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from scipy import stats

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import SignalPeptideVectorNN, FocalLoss
from src.data_loading import load_plm_with_bins

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
SEEDS = [42, 123, 456, 789, 1024]
N_SEEDS = len(SEEDS)
N_FOLDS = 5
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'

# Best vector model config (from Script 10)
VECTOR_CONFIG = dict(
    hidden_layers=(256, 256, 128),
    dropout=0.35,
    learning_rate=5e-4,
    batch_size=32,
    epochs=200,
    loss='focal',
)

# Fine-tuning config
FT_EPOCHS = 100
FT_LR = 1e-4        # 5x lower than pretrain
FT_PATIENCE = 10     # stricter early stopping
FT_BATCH_SIZE = 32

# External datasets
EXTERNAL_DATASETS = {
    'Wu': {
        'file': 'wu_esm_embeddings.parquet',
        'is_binary': True,
        'description': '81 SPs, binary WA (functional/non-functional)',
    },
    'Xue': {
        'file': 'xue_esm_embeddings.parquet',
        'is_binary': False,
        'description': '322 SPs, WA 0–10437 (enzyme activity units)',
    },
    'Zhang-P43': {
        'file': 'zhang_p43_esm_embeddings.parquet',
        'is_binary': False,
        'description': '114 SPs, WA 0–193 (P43 promoter)',
    },
    'Zhang-PglVM': {
        'file': 'zhang_pglvm_esm_embeddings.parquet',
        'is_binary': False,
        'description': '114 SPs, WA 0–327 (PglVM promoter)',
    },
}

BIN_CENTERS = np.arange(1, 11, dtype=np.float64)  # [1, 2, ..., 10]


def load_external_dataset(filename):
    """Load an external ESM embedding parquet file."""
    path = DATA_DIR / filename
    df = pd.read_parquet(path)
    X = np.stack(df['embedding'].values)
    y = df['WA'].values.astype(np.float64)
    return X, y


def pretrain_vector_ensemble(X_train, y_train_bins, scaler):
    """Pretrain 5-seed vector NN ensemble on Grasso data."""
    X_train_scaled = scaler.transform(X_train)
    models = []

    for i, seed in enumerate(SEEDS):
        print(f"  Pretraining seed {seed} ({i+1}/{N_SEEDS})...", end=' ', flush=True)

        # 80/20 split for early stopping during pretrain
        rng = np.random.default_rng(seed)
        n = len(y_train_bins)
        indices = rng.permutation(n)
        n_val = int(n * 0.2)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        config = {**VECTOR_CONFIG, 'random_state': seed}
        model = SignalPeptideVectorNN(**config)
        model.fit(
            X_train_scaled[train_idx], y_train_bins[train_idx],
            X_val=X_train_scaled[val_idx], y_val_bins=y_train_bins[val_idx],
            verbose=0,
        )

        # Sanity: predict on val and compute WA MSE
        val_probs = model.predict(X_train_scaled[val_idx])
        val_wa_pred = val_probs @ BIN_CENTERS
        from src.data_loading import BIN_COLUMNS
        val_wa_true = y_train_bins[val_idx] @ BIN_CENTERS
        val_mse = float(np.mean((val_wa_true - val_wa_pred) ** 2))
        print(f"val MSE = {val_mse:.4f}")

        models.append(model)

    return models


def _build_ft_model(pretrained_model, output_activation, output_units, loss_fn,
                    metrics, seed):
    """
    Build a fine-tuning model by extracting pretrained feature layers and
    replacing the output head.

    Architecture of pretrained model (3 hidden blocks):
        layers[0]: Input
        layers[1]: Dense(256)   layers[2]: LeakyReLU  layers[3]: Dropout  (block 1)
        layers[4]: Dense(256)   layers[5]: LeakyReLU  layers[6]: Dropout  (block 2)
        layers[7]: Dense(128)   layers[8]: LeakyReLU  layers[9]: Dropout  (block 3)
        layers[10]: Dense(10, softmax)

    We freeze blocks 1-2 (layers 1-6) and keep block 3 + new output trainable.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    src_model = pretrained_model.model_

    # Get the output of the last hidden block (before the softmax output layer)
    # This is the output of the last Dropout layer (layer index -2, i.e. second-to-last)
    feature_output = src_model.layers[-2].output  # last Dropout output

    # Build new model: pretrained features → new output head
    new_output = layers.Dense(output_units, activation=output_activation,
                              name='ft_output')(feature_output)
    ft_model = keras.Model(inputs=src_model.input, outputs=new_output)

    # Freeze first 2 hidden blocks (6 layers: Dense+LeakyReLU+Dropout × 2)
    # layers[0] = Input (skip), layers[1:7] = blocks 1-2 (freeze)
    # layers[7:10] = block 3 (trainable), layers[10] = new output (trainable)
    for layer in ft_model.layers[1:7]:
        layer.trainable = False
    for layer in ft_model.layers[7:]:
        layer.trainable = True

    ft_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FT_LR),
        loss=loss_fn,
        metrics=metrics,
    )
    return ft_model


def clone_and_finetune_regression(pretrained_model, X_train, y_train,
                                  X_val, y_val, seed):
    """
    Fine-tune a pretrained vector model for continuous regression.

    Freezes first 2 hidden blocks, keeps last block trainable.
    Replaces the softmax(10) output with a linear(1) output.
    """
    ft_model = _build_ft_model(pretrained_model, 'linear', 1, 'mse', ['mae'], seed)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=FT_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    fit_kwargs = dict(
        x=X_train, y=y_train,
        batch_size=FT_BATCH_SIZE,
        epochs=FT_EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )
    if X_val is not None:
        fit_kwargs['validation_data'] = (X_val, y_val)

    ft_model.fit(**fit_kwargs)
    return ft_model


def clone_and_finetune_binary(pretrained_model, X_train, y_train,
                              X_val, y_val, seed):
    """
    Fine-tune a pretrained vector model for binary classification.

    Replaces softmax(10) output with sigmoid(1) for binary prediction.
    """
    ft_model = _build_ft_model(pretrained_model, 'sigmoid', 1,
                               'binary_crossentropy', ['accuracy'], seed)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=FT_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    fit_kwargs = dict(
        x=X_train, y=y_train,
        batch_size=FT_BATCH_SIZE,
        epochs=FT_EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )
    if X_val is not None:
        fit_kwargs['validation_data'] = (X_val, y_val)

    ft_model.fit(**fit_kwargs)
    return ft_model


def evaluate_fold(y_true, y_pred, is_binary):
    """Compute metrics for one fold."""
    result = {}

    sp_rho, sp_p = stats.spearmanr(y_true, y_pred)
    result['spearman_rho'] = float(sp_rho)
    result['spearman_p'] = float(sp_p)

    pe_r, pe_p = stats.pearsonr(y_true, y_pred)
    result['pearson_r'] = float(pe_r)
    result['pearson_p'] = float(pe_p)

    result['n_samples'] = len(y_true)

    if is_binary:
        try:
            auc = roc_auc_score(y_true, y_pred)
            result['auc_roc'] = float(auc)
        except ValueError:
            result['auc_roc'] = float('nan')

    return result


def run_finetuning_cv(pretrained_models, scaler_grasso, dataset_name, spec):
    """Run 5-fold CV fine-tuning for one external dataset."""
    X_ext, y_ext = load_external_dataset(spec['file'])
    is_binary = spec['is_binary']

    print(f"\n{'─'*60}")
    print(f"  {dataset_name}: {spec['description']}")
    print(f"  N={len(y_ext)}, binary={is_binary}")
    print(f"{'─'*60}")

    # --- Zero-shot baseline (no fine-tuning) ---
    X_ext_grasso_scaled = scaler_grasso.transform(X_ext)
    zs_preds = []
    for model in pretrained_models:
        probs = model.predict(X_ext_grasso_scaled)
        wa_pred = probs @ BIN_CENTERS
        zs_preds.append(wa_pred)
    zs_pred_avg = np.mean(zs_preds, axis=0)
    zs_metrics = evaluate_fold(y_ext, zs_pred_avg, is_binary)
    print(f"  Zero-shot Spearman: {zs_metrics['spearman_rho']:+.4f} "
          f"(p={zs_metrics['spearman_p']:.2e})")

    # --- Fine-tuned CV ---
    if is_binary:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X_ext, y_ext))
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X_ext))

    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, y_tr = X_ext[train_idx], y_ext[train_idx]
        X_te, y_te = X_ext[test_idx], y_ext[test_idx]

        # Scale using Grasso training statistics so frozen pretrained layers
        # receive inputs with the same distribution as during pretraining
        X_tr_scaled = scaler_grasso.transform(X_tr)
        X_te_scaled = scaler_grasso.transform(X_te)

        # Fine-tune each pretrained seed model
        fold_preds = []
        for seed_i, pt_model in enumerate(pretrained_models):
            seed = SEEDS[seed_i]

            if is_binary:
                ft_model = clone_and_finetune_binary(
                    pt_model, X_tr_scaled, y_tr,
                    X_val=None, y_val=None, seed=seed)
                pred = ft_model.predict(X_te_scaled, verbose=0).ravel()
            else:
                ft_model = clone_and_finetune_regression(
                    pt_model, X_tr_scaled, y_tr,
                    X_val=None, y_val=None, seed=seed)
                pred = ft_model.predict(X_te_scaled, verbose=0).ravel()

            fold_preds.append(pred)

            # Clean up to save memory
            del ft_model
            keras.backend.clear_session()

        # Ensemble average
        avg_pred = np.mean(fold_preds, axis=0)
        fold_metrics = evaluate_fold(y_te, avg_pred, is_binary)
        fold_metrics['fold'] = fold_i
        fold_metrics['n_train'] = len(train_idx)
        fold_metrics['n_test'] = len(test_idx)
        fold_results.append(fold_metrics)

        sp = fold_metrics['spearman_rho']
        extra = f"  AUC={fold_metrics.get('auc_roc', 0):.3f}" if is_binary else ""
        print(f"  Fold {fold_i+1}/{N_FOLDS}: Spearman={sp:+.4f}  n_test={len(test_idx)}{extra}")

    # Aggregate across folds
    agg = {}
    for key in ['spearman_rho', 'pearson_r']:
        vals = [f[key] for f in fold_results]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))
    if is_binary:
        auc_vals = [f['auc_roc'] for f in fold_results if not np.isnan(f.get('auc_roc', float('nan')))]
        if auc_vals:
            agg['auc_roc_mean'] = float(np.mean(auc_vals))
            agg['auc_roc_std'] = float(np.std(auc_vals))

    print(f"\n  Fine-tuned CV Spearman: {agg['spearman_rho_mean']:+.4f} "
          f"± {agg['spearman_rho_std']:.4f}")
    print(f"  Zero-shot Spearman:    {zs_metrics['spearman_rho']:+.4f}")
    improvement = agg['spearman_rho_mean'] - zs_metrics['spearman_rho']
    print(f"  Improvement:           {improvement:+.4f}")

    return {
        'dataset': dataset_name,
        'description': spec['description'],
        'is_binary': is_binary,
        'n_samples': len(y_ext),
        'n_folds': N_FOLDS,
        'n_seeds': N_SEEDS,
        'zero_shot': zs_metrics,
        'fine_tuned_folds': fold_results,
        'fine_tuned_aggregate': agg,
    }


def make_figure(all_results, save_path):
    """Generate comparison figure: zero-shot vs fine-tuned."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    datasets = [r['dataset'] for r in all_results]
    zs_spearman = [r['zero_shot']['spearman_rho'] for r in all_results]
    ft_spearman = [r['fine_tuned_aggregate']['spearman_rho_mean'] for r in all_results]
    ft_std = [r['fine_tuned_aggregate']['spearman_rho_std'] for r in all_results]

    # Panel A: Side-by-side bar chart
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.35

    bars_zs = ax.bar(x - width/2, zs_spearman, width, label='Zero-shot',
                     color='steelblue', alpha=0.85)
    bars_ft = ax.bar(x + width/2, ft_spearman, width, yerr=ft_std,
                     label='Fine-tuned (5-fold CV)', color='forestgreen',
                     alpha=0.85, capsize=4)

    ax.set_ylabel('Spearman ρ')
    ax.set_title('(A) Zero-Shot vs Fine-Tuned')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

    for bar in list(bars_zs) + list(bars_ft):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.02 if height >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{height:+.2f}', ha='center', va=va, fontsize=8)

    # Panel B: Improvement
    ax = axes[1]
    improvements = [ft - zs for ft, zs in zip(ft_spearman, zs_spearman)]
    colors = ['forestgreen' if imp > 0 else 'firebrick' for imp in improvements]
    bars = ax.bar(datasets, improvements, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Spearman ρ Improvement')
    ax.set_title('(B) Fine-Tuning Improvement')
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.02 if height >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{imp:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    ax.set_xticklabels(datasets, rotation=15, ha='right')

    fig.suptitle('Cross-Dataset Fine-Tuning: Vector Model Transfer Learning',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {save_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)

    # ── 1. Load Grasso training data with bins ────────────────────────────
    print("Loading Grasso ESM2-650M data with bin probabilities...")
    (X_train, X_test, y_train_wa, y_test_wa,
     y_train_bins, y_test_bins, meta) = load_plm_with_bins('esm2-650M')
    print(f"  Train: {meta['n_train']} samples ({meta['n_train_dropped_nan_bins']} dropped for NaN bins)")
    print(f"  Test:  {meta['n_test']} samples")

    # ── 2. Fit scaler on Grasso training data ─────────────────────────────
    scaler_grasso = StandardScaler()
    scaler_grasso.fit(X_train)

    # ── 3. Pretrain vector ensemble on Grasso ─────────────────────────────
    print(f"\nPretraining {N_SEEDS}-seed vector ensemble on Grasso...")
    pretrained_models = pretrain_vector_ensemble(X_train, y_train_bins, scaler_grasso)

    # Sanity check on Grasso test set
    X_test_scaled = scaler_grasso.transform(X_test)
    test_preds = []
    for m in pretrained_models:
        probs = m.predict(X_test_scaled)
        wa_pred = probs @ BIN_CENTERS
        test_preds.append(wa_pred)
    test_pred_avg = np.mean(test_preds, axis=0)
    test_mse = float(np.mean((y_test_wa - test_pred_avg) ** 2))
    test_sp, _ = stats.spearmanr(y_test_wa, test_pred_avg)
    print(f"\n  Grasso test MSE: {test_mse:.4f}, Spearman: {test_sp:.4f}")

    # ── 4. Fine-tune on each external dataset ─────────────────────────────
    print(f"\n{'='*60}")
    print("  Cross-Dataset Fine-Tuning (5-Fold CV)")
    print(f"{'='*60}")

    all_results = []
    for name, spec in EXTERNAL_DATASETS.items():
        result = run_finetuning_cv(pretrained_models, scaler_grasso, name, spec)
        all_results.append(result)

    # ── 5. Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15} {'N':>4}  {'Zero-Shot ρ':>12}  {'Fine-Tuned ρ':>14}  {'Δ':>8}")
    print(f"  {'-'*15} {'-'*4}  {'-'*12}  {'-'*14}  {'-'*8}")
    for r in all_results:
        zs = r['zero_shot']['spearman_rho']
        ft = r['fine_tuned_aggregate']['spearman_rho_mean']
        ft_s = r['fine_tuned_aggregate']['spearman_rho_std']
        delta = ft - zs
        print(f"  {r['dataset']:<15} {r['n_samples']:>4}  {zs:>+12.4f}  "
              f"{ft:>+.4f} ± {ft_s:.4f}  {delta:>+8.4f}")

    # ── 6. Figure ─────────────────────────────────────────────────────────
    fig_path = FIGURES_DIR / 'cross_dataset_finetuning.png'
    make_figure(all_results, fig_path)

    # ── 7. Save results ───────────────────────────────────────────────────
    results_out = {
        'config': {
            'vector_config': {k: list(v) if isinstance(v, tuple) else v
                              for k, v in VECTOR_CONFIG.items()},
            'finetune_epochs': FT_EPOCHS,
            'finetune_lr': FT_LR,
            'finetune_patience': FT_PATIENCE,
            'n_folds': N_FOLDS,
            'n_seeds': N_SEEDS,
            'seeds': SEEDS,
        },
        'grasso_pretrain': {
            'test_mse': test_mse,
            'test_spearman': float(test_sp),
        },
        'datasets': {r['dataset']: r for r in all_results},
    }

    out_path = RESULTS_DIR / 'cross_dataset_finetuning_results.json'
    with open(out_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

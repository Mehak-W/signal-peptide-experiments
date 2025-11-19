#!/usr/bin/env python3
"""
Grasso Design Task Evaluation
==============================

Grasso et al. describe a design task where they designed variants of signal peptides
and compared model predictions to experimental results. This is different from the
standard test set evaluation.

This script evaluates our models on the Grasso design library:
- WT sequences (6 total)
- Designed variants (11,637 total across multiple design strategies)
- Design strategies include: position scanning, charge modifications, hydrophobicity changes, etc.

Usage:
    python scripts/08_grasso_design_task_evaluation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_design_library():
    """Load Grasso design library with WT and variants"""
    print("="*70)
    print("Loading Grasso Design Library")
    print("="*70 + "\n")

    data_dir = Path(__file__).parent.parent / 'data'
    xlsx_path = data_dir / 'sb2c00328_si_011.xlsx'

    print(f"Loading from: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name='Library_w_Bins_and_WA')

    print(f"\nTotal sequences: {len(df)}")
    print(f"Libraries: {df['Library'].nunique()}")

    # Separate WT and designed
    wt = df[df['Library'] == 'WT'].copy()
    designed = df[df['Library'] != 'WT'].copy()

    print(f"\nWT sequences: {len(wt)}")
    print(f"Designed variants: {len(designed)}")

    # Remove NaN WA values
    wt = wt.dropna(subset=['WA'])
    designed = designed.dropna(subset=['WA'])

    print(f"\nAfter removing NaN WA:")
    print(f"WT sequences: {len(wt)}")
    print(f"Designed variants: {len(designed)}")

    return wt, designed, df


def load_model_predictions():
    """
    Load predictions from our trained models.
    For simplicity, we'll use the best model (Ginkgo AA0 NN)
    """
    print("\n" + "="*70)
    print("Loading Model Predictions")
    print("="*70 + "\n")

    # We need to generate embeddings for the design library and make predictions
    # For now, let's check if we have predictions saved
    results_dir = Path(__file__).parent.parent / 'results'

    # Since we don't have pre-computed predictions for all design variants,
    # we'll need to generate them on-the-fly using our trained model
    # This requires loading the model and computing embeddings

    print("NOTE: Design library predictions require:")
    print("  1. Computing PLM embeddings for all design variants")
    print("  2. Loading trained NN model")
    print("  3. Generating predictions")
    print("\nFor this initial implementation, we'll create a placeholder")
    print("that shows the analysis framework. Full implementation requires")
    print("embedding generation (compute-intensive).")

    return None


def evaluate_design_predictions(wt_df, designed_df, predictions=None):
    """
    Evaluate model predictions on design task.

    Key metrics:
    - Correlation between predicted and actual WA changes
    - Ability to rank variants correctly
    - Identification of beneficial vs detrimental mutations
    """
    print("\n" + "="*70)
    print("Design Task Evaluation Framework")
    print("="*70 + "\n")

    # Group designed variants by parent WT gene
    print("Design strategies in library:")
    print(designed_df['Library'].value_counts().head(20))

    # Analyze WT sequences
    print("\n\nWT sequences available for comparison:")
    print(wt_df[['ID', 'gene', 'WA', 'SP_aa']].to_string())

    # For each WT, find its variants
    print("\n\nVariants per WT gene:")
    for gene in wt_df['gene'].unique():
        wt_wa = wt_df[wt_df['gene'] == gene]['WA'].values[0]
        variants = designed_df[designed_df['gene'] == gene]
        if len(variants) > 0:
            print(f"\n{gene} (WT WA={wt_wa:.2f}):")
            print(f"  Variants: {len(variants)}")
            print(f"  WA range: {variants['WA'].min():.2f} - {variants['WA'].max():.2f}")
            print(f"  Libraries: {variants['Library'].nunique()}")

    # Create figure showing design task structure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, gene in enumerate(wt_df['gene'].unique()):
        if i >= 6:
            break

        ax = axes[i // 3, i % 3]

        wt_wa = wt_df[wt_df['gene'] == gene]['WA'].values[0]
        variants = designed_df[designed_df['gene'] == gene]

        if len(variants) > 0:
            # Histogram of variant performance
            ax.hist(variants['WA'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(wt_wa, color='red', linestyle='--', linewidth=2, label=f'WT = {wt_wa:.2f}')

            ax.set_xlabel('WA Score', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title(f'{gene} Design Library\n({len(variants)} variants)', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path('figures/design_task_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    return None


def main():
    """Main evaluation pipeline"""
    print("\n" + "="*70)
    print("Grasso Design Task Evaluation")
    print("="*70 + "\n")

    print("This analysis evaluates model performance on designed variants,")
    print("distinct from the standard train/test split evaluation.")
    print("")

    # Load data
    wt_df, designed_df, full_df = load_design_library()

    # Load predictions (placeholder for now)
    predictions = load_model_predictions()

    # Evaluate
    evaluate_design_predictions(wt_df, designed_df, predictions)

    print("\n" + "="*70)
    print("Design Task Analysis Complete")
    print("="*70)
    print("\nNEXT STEPS:")
    print("  1. Generate PLM embeddings for all design variants")
    print("  2. Load trained NN model and make predictions")
    print("  3. Compare predictions vs experimental results")
    print("  4. Analyze: Can model predict which mutations improve/worsen secretion?")
    print("\nThis framework shows the structure of the design task.")
    print("Full implementation requires embedding generation (compute-intensive).")


if __name__ == '__main__':
    main()

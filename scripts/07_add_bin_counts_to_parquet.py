#!/usr/bin/env python3
"""
Add Bin Count Columns to Parquet Files
=======================================

Professor Schrier noted: "I can't imagine that storing 10 floats x 1000 rows adds
more than a few kB. Most of these files are the representation vectors"

This script adds the 10 bin probability columns from the Excel file to the parquet files.

Usage:
    python scripts/07_add_bin_counts_to_parquet.py
"""

import pandas as pd
from pathlib import Path
import numpy as np

def add_bin_counts_to_parquet():
    """Add bin count columns to all parquet files"""

    print("="*70)
    print("Adding Bin Count Columns to Parquet Files")
    print("="*70 + "\n")

    # Load Excel file with bin counts
    data_dir = Path(__file__).parent.parent / 'data'
    xlsx_path = data_dir / 'sb2c00328_si_011.xlsx'

    print("Loading bin counts from Excel file...")
    df_bins = pd.read_excel(xlsx_path, sheet_name='Library_w_Bins_and_WA')

    # Get bin columns
    bin_cols = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]

    print(f"Found {len(bin_cols)} bin columns in Excel file")
    print(f"Total sequences in Excel: {len(df_bins)}")

    # Create mapping from sequence to bin probabilities
    seq_to_bins = {}
    for idx, row in df_bins.iterrows():
        seq = row['SP_aa']
        bins = row[bin_cols].values
        seq_to_bins[seq] = bins

    print(f"Created mapping for {len(seq_to_bins)} sequences\n")

    # Process each parquet file
    parquet_files = [
        'trainAA_esm2-650M.parquet',
        'trainAA_esm2-3B.parquet',
        'trainAA_ginkgo-AA0-650M.parquet',
        'testAA_esm2-650M.parquet',
        'testAA_esm2-3B.parquet',
        'testAA_ginkgo-AA0-650M.parquet'
    ]

    for parquet_file in parquet_files:
        print(f"Processing {parquet_file}...")
        parquet_path = data_dir / parquet_file

        # Load parquet file
        df = pd.read_parquet(parquet_path)
        initial_size = parquet_path.stat().st_size / (1024**2)  # MB

        print(f"  Loaded {len(df)} sequences")
        print(f"  Initial size: {initial_size:.2f} MB")
        print(f"  Columns: {list(df.columns)}")

        # Add bin columns
        bin_data = []
        matched = 0
        for idx, row in df.iterrows():
            seq = row['sequence']
            if seq in seq_to_bins:
                bin_data.append(seq_to_bins[seq])
                matched += 1
            else:
                # If no match, fill with NaN
                bin_data.append([np.nan] * 10)

        # Convert to DataFrame and add to original
        bin_df = pd.DataFrame(bin_data, columns=bin_cols, index=df.index)
        df_updated = pd.concat([df, bin_df], axis=1)

        print(f"  Matched {matched}/{len(df)} sequences with bin data")

        # Save updated parquet
        df_updated.to_parquet(parquet_path)

        final_size = parquet_path.stat().st_size / (1024**2)  # MB
        size_increase = final_size - initial_size

        print(f"  Updated columns: {list(df_updated.columns)}")
        print(f"  Final size: {final_size:.2f} MB")
        print(f"  Size increase: {size_increase:.2f} MB ({size_increase/initial_size*100:.1f}%)")
        print(f"  ✓ Saved updated file\n")

    print("="*70)
    print("Bin Counts Added to All Parquet Files!")
    print("="*70)
    print("\nAs Professor Schrier noted: adding 10 floats per row adds minimal size")
    print("compared to the embedding vectors (1280 or 2560 floats per row).")


if __name__ == '__main__':
    add_bin_counts_to_parquet()

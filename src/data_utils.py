"""
Data Loading and Processing Utilities
Shared functions for loading PLM embeddings and Grasso dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


def load_plm_embeddings(
    model_name: str,
    data_dir: Optional[str] = None,
    return_split: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-computed PLM embeddings from parquet files.

    Parameters
    ----------
    model_name : str
        Name of the PLM model ('esm2-650M', 'esm2-3B', or 'ginkgo-AA0-650M')
    data_dir : str, optional
        Directory containing the parquet files. If None, auto-detects relative to this file.
    return_split : bool
        If True, return train/test split. If False, return combined data.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Training and test embeddings and labels

    Notes
    -----
    The parquet files contain:
    - sequence: amino acid sequence
    - embedding: vector representation from PLM
    - WA: weighted average efficiency score (1=best, 10=worst)
    """
    # Auto-detect data directory if not provided
    if data_dir is None:
        # Try to find data directory relative to this file or cwd
        src_dir = Path(__file__).parent
        repo_root = src_dir.parent
        data_path = repo_root / "data"
        if not data_path.exists():
            # Try from current working directory
            data_path = Path.cwd() / "data"
        if not data_path.exists():
            data_path = Path.cwd() / "../data"
    else:
        data_path = Path(data_dir)

    # Load training and test data
    train_file = data_path / f"trainAA_{model_name}.parquet"
    test_file = data_path / f"testAA_{model_name}.parquet"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    # Extract embeddings and labels
    X_train = np.stack(train_df['embedding'].values)
    X_test = np.stack(test_df['embedding'].values)
    y_train = train_df['WA'].values
    y_test = test_df['WA'].values

    print(f"Loaded {model_name} embeddings:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")

    if return_split:
        return X_train, X_test, y_train, y_test
    else:
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        return X, y


def load_grasso_data(
    data_dir: Optional[str] = None,
    filename: str = "sb2c00328_si_011.xlsx"
) -> pd.DataFrame:
    """
    Load the original Grasso et al. dataset.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the data file. If None, auto-detects.
    filename : str
        Name of the Excel file

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
        - SP_nt: nucleotide sequence
        - SP_aa: amino acid sequence
        - WA: weighted average efficiency (1=best, 10=worst)
        - Set: train/test split indicator
    """
    # Auto-detect data directory if not provided
    if data_dir is None:
        src_dir = Path(__file__).parent
        repo_root = src_dir.parent
        data_path = repo_root / "data" / filename
        if not data_path.parent.exists():
            data_path = Path.cwd() / "data" / filename
        if not data_path.parent.exists():
            data_path = Path.cwd() / "../data" / filename
    else:
        data_path = Path(data_dir) / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_excel(data_path)
    print(f"Loaded Grasso dataset: {len(df)} sequences")

    return df


def get_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame based on 'Set' column from Grasso et al.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Set' column

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Training and test DataFrames
    """
    train_df = df[df['Set'] == 'Train'].copy()
    test_df = df[df['Set'] == 'Test'].copy()

    print(f"Train/Test split: {len(train_df)} / {len(test_df)} sequences")

    return train_df, test_df


def get_model_info() -> Dict[str, Dict[str, any]]:
    """
    Get information about available PLM models.

    Returns
    -------
    model_info : dict
        Dictionary with model names as keys and info as values
    """
    return {
        'esm2-650M': {
            'full_name': 'ESM-2 650M',
            'parameters': '650M',
            'embedding_dim': 1280,
            'description': 'ESM-2 with 650M parameters'
        },
        'esm2-3B': {
            'full_name': 'ESM-2 3B',
            'parameters': '3B',
            'embedding_dim': 2560,
            'description': 'ESM-2 with 3B parameters'
        },
        'ginkgo-AA0-650M': {
            'full_name': 'Ginkgo AA0',
            'parameters': '650M',
            'embedding_dim': 1280,
            'description': 'Ginkgo AA0 with 650M parameters'
        }
    }


def load_cross_dataset(
    dataset_name: str,
    data_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cross-dataset embeddings (Wu, Xue, Zhang, etc.).

    Parameters
    ----------
    dataset_name : str
        Name of dataset: 'wu', 'xue', 'zhang_p43', 'zhang_pglvm'
    data_dir : str, optional
        Directory containing the parquet files. If None, auto-detects.

    Returns
    -------
    X, y : np.ndarray
        Embeddings and labels

    Notes
    -----
    These datasets are used for cross-dataset evaluation to test generalization
    of models trained on the Grasso dataset.
    """
    # Auto-detect data directory
    if data_dir is None:
        src_dir = Path(__file__).parent
        repo_root = src_dir.parent
        data_path = repo_root / "data"
        if not data_path.exists():
            data_path = Path.cwd() / "data"
    else:
        data_path = Path(data_dir)

    # Load dataset
    file_path = data_path / f"{dataset_name}_esm_embeddings.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Cross-dataset file not found: {file_path}")

    df = pd.read_parquet(file_path)

    # Extract embeddings and labels
    X = np.stack(df['embedding'].values)
    y = df['WA'].values

    print(f"Loaded {dataset_name} dataset:")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  WA range: {y.min():.2f} to {y.max():.2f}")

    return X, y


def get_cross_datasets() -> Dict[str, str]:
    """
    Get information about available cross-datasets.

    Returns
    -------
    datasets : dict
        Dictionary with dataset names and descriptions
    """
    return {
        'wu': 'Wu et al. - B. subtilis secretion (81 sequences)',
        'xue': 'Xue et al. - S. cerevisiae secretion (322 sequences)',
        'zhang_p43': 'Zhang et al. - P43 promoter (443 sequences)',
        'zhang_pglvm': 'Zhang et al. - PglvM promoter (443 sequences)'
    }

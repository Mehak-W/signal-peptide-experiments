# Data Directory

This directory contains pre-computed Protein Language Model (PLM) embeddings for signal peptide sequences.

## Files

### PLM Embeddings (Parquet Format)

The following files contain pre-computed embeddings from three different PLM models:

**ESM-2 650M:**
- `trainAA_esm2-650M.parquet` - Training set embeddings (1280-dimensional)
- `testAA_esm2-650M.parquet` - Test set embeddings (1280-dimensional)

**ESM-2 3B:**
- `trainAA_esm2-3B.parquet` - Training set embeddings (2560-dimensional)
- `testAA_esm2-3B.parquet` - Test set embeddings (2560-dimensional)

**Ginkgo AA0 650M:**
- `trainAA_ginkgo-AA0-650M.parquet` - Training set embeddings (1280-dimensional)
- `testAA_ginkgo-AA0-650M.parquet` - Test set embeddings (1280-dimensional)

### Original Dataset

**`sb2c00328_si_011.xlsx`**

Supplementary Table S2 from [Grasso et al. (2023)](https://dx.doi.org/10.1021/acssynbio.2c00328).

Columns:
- `SP_nt`: Nucleotide sequence of signal peptide
- `SP_aa`: Amino acid sequence of signal peptide
- `WA`: Weighted average efficiency score (1=best, 10=worst)
- `Set`: Train/Test split indicator

Note: Only 83% of sequences have WA values (successful retrieval after screening).

## Data Format

### Parquet Files Structure

Each parquet file contains three columns:

1. **sequence** (string): Amino acid sequence of the signal peptide
2. **embedding** (array): Dense vector representation from the PLM
   - ESM-2 650M / Ginkgo AA0: 1280 dimensions
   - ESM-2 3B: 2560 dimensions
3. **WA** (float): Weighted average efficiency score
   - Range: 1 (best) to 10 (worst)
   - Based on screening results from Grasso et al.

### Dataset Statistics

- **Training set**: ~3,095 sequences
- **Test set**: ~1,326 sequences
- **Total**: 4,421 informative signal peptides
- **Feature dimensions**:
  - ESM-2 650M: 1280
  - ESM-2 3B: 2560
  - Ginkgo AA0: 1280

## Data Generation

The PLM embeddings were generated using:
- **ESM-2 models**: Facebook/Meta AI's ESM-2 (650M and 3B parameter versions)
- **Ginkgo AA0**: Ginkgo BioWorks' protein language model
- **API**: Ginkgo BioWorks API (<$0.10 USD total cost as of 03/2025)
- **Generation script**: See `signal_peptides-main/scripts/06_Retrieve_Peptide_Embeddings.wls`

## Loading Data

Use the provided utility functions:

```python
from src.data_utils import load_plm_embeddings

# Load a specific embedding
X_train, X_test, y_train, y_test = load_plm_embeddings('esm2-650M')

# Available models: 'esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M'
```

## References

1. Grasso, S., et al. (2023). "Signal Peptide Efficiency: From High-Throughput Data to Prediction and Explanation." *ACS Synthetic Biology*. DOI: 10.1021/acssynbio.2c00328

2. Lin, Z., et al. (2022). "Language models of protein sequences at the scale of evolution enable accurate structure prediction." *bioRxiv*. (ESM-2 paper)

## License

The original dataset is from Grasso et al. (2023) and is reproduced here for research purposes. Please cite the original paper if you use this data.

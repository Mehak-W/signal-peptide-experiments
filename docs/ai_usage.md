# AI Tool Usage Documentation

## Model Version

* **Model**: Claude (Anthropic), Opus 4.5 family
* **Access**: Web interface (claude.ai)
* **Period of use**: January--February 2026

## How AI Was Used

Claude was used as a coding assistant during three phases of development:

1. **Debugging** -- Resolving Python errors in TensorFlow/Keras model
   building, pandas data loading, and matplotlib figure generation.
2. **Experiment structure** -- Brainstorming how to organize the
   architecture search (Script 09) and ensemble optimization (Script 10)
   into phased sweeps, and discussing which hyperparameter ranges were
   reasonable for the dataset size.
3. **Verification** -- Cross-checking reported metrics in summary tables
   against the raw JSON result files. This pass caught several rounding
   discrepancies that were corrected before submission.

All code was written by I, Mehak Wadhwa, building iteratively from the
baseline reproduction (Script 01) through the final ensemble optimization
(Script 10). Decisions on model architecture, loss function selection,
evaluation methodology, and interpretation of results were made by the
me.


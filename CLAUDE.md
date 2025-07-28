# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multilingual scaling laws research application that implements and fits scaling laws for machine learning models. The codebase creates an intuitive, aesthetically-pleasing web application for machine learning practitioners to understand and interpret scaling laws. It enables them to do the following:

* [1] Given a compute budget, visualize the Scaling Laws from notable papers, and recommend an optimal model size (N*) and optimal number of training tokens (D*).
* [2] Automatically fit their own scaling laws given a csv of (N,D,Loss) data.
* [3] For both of the above, read guidance and recommendations for their particular scaling law goals and setup.

For [1] here are the inputs a user can provide:

* Required: Training compute budget C (in FLOPs) OR a target loss value L.
* Optional: Inference compute budget in test-time tokens D_{test}
* Optional: Language (defaults to English, we will provide a few other options)
* Optional: The total tokens in the training set, in case there are data constraints
* Optional: Model FLOP Utilization at Inference---this is specific for when we are also using the inference compute budget (MFU defined in this paper: https://arxiv.org/pdf/2401.00448)

For [1] the outputs to the user would be:

* From relevant scaling laws, a table of the optimal N* and D*
* Plot: Optimal scaling line as training time D (x-axis) and model size N (y-axis) change for each applicable scaling law model. It should show the compute budget constraint curve, and the various lines for each scaling law that pertain to optimal N and D.
* Plot: Tokens per parameter ratio (y-axis) over compute budget CT (x-axis) for each applicable scaling law model.
* Recommendations to user: Point them to the papers from which the shown/plotted scaling laws were derived above, and the extent to which they are relevant to their work: ie is the user's compute budget in range of what the paper showed? Is it using similar architecture, data, or other assumptions?


For [2], we don't have all the fitting code ready yet, but here is what the interface should expect as input:

* Required: File with [L, N, D] triples. Must be between a reasonable min and max length (to be able to obtain a git, and not be too long for us to process efficiently).
* Required: Select the type of scaling law they would like to fit from drop-down list.
* Optional: May need to allow the csv to have optional other columns, eg total unique tokens U, in case they are relevant for fitting the scaling law. The scaling law class itself should check that it has them before fitting.

For [2] the outputs to the user would be:

* Full formula, with all fitted values.
* Evaluation metrics on fit quality (R-squared, etc, dropping-out random rows).
* Plot: Optimal scaling line as training time D (x-axis) and model size N (y-axis) change.
* Plot: Tokens per parameter ratio (y-axis) over compute budget CT (x-axis)
* Recommendations to user? (TODO)




## Key Architecture

### Core Classes

- `ScalingLaw` (src/scaling_law_classes/scaling_law.py): Abstract base class defining the API for all scaling law implementations
  - Key methods: `loss()`, `N_to_D()`, `DL_to_N()`, `compute_optimal_allocation()`, `compute_optimal_allocation_inference()`
  - Handles both training and inference compute optimization
  - Note: The actual file paths have changed from src/scaling_law.py to src/scaling_law_classes/scaling_law.py

- `LawParams` dataclass: Stores scaling law parameters (A, B, irreducible, alpha, beta, extras)

- `ScalingLawWrapper` (src/scaling_law_classes/scaling_law.py): Stores all scaling law metadata including paper URL, architecture, languages, compute budget range

- `BasicScalingLaw` (src/scaling_law_classes/basic_scaling_law.py): Implements the standard Chinchilla scaling law
  - Loss formula: L = E + A/N^α + B/D^β
  - Provides analytic solutions for optimal allocation

- `DataConstrainedScalingLaw` (src/scaling_law_classes/data_constrained_scaling_law.py): Implements scaling laws with data repetition constraints
  - Additional variable U (unique tokens)
  - Handles repetition factors for both model and data



### Key Modules

- `src/scaling_laws.py`: Registry of pre-defined scaling laws (Chinchilla, Chinchilla Replication, etc.) stored as `ALL_SCALING_LAWS` dictionary
- `src/helpers/plotting.py`: Visualization utilities for scaling law plots
- `src/fitters.py`: Fitting utilities (currently empty, to be implemented)

### Key Capabilities

1. **Fitting**: Supports both PyTorch and NumPy implementations for fitting scaling laws to data (not yet implemented)
2. **Loss Prediction**: Compute loss given model size (N) and data size (D)
3. **Optimal Allocation**: Find optimal N,D given compute budget
4. **Inference Optimization**: Compute optimal training tokens considering inference costs
5. **Iso-loss Curves**: Generate curves of constant loss across N,D space

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (using pytest from requirements)
pytest

# Start Jupyter for interactive analysis
jupyter lab

# Run Streamlit app (if exists)
streamlit run app.py
```

## Development Notes

- The codebase uses both NumPy and PyTorch for numerical computations
- Plotting utilities are in src/helpers/plotting.py
- The project appears to be set up for both library use and interactive analysis (Jupyter, Streamlit)
- Main dependencies include: torch, numpy, scipy, pandas, matplotlib, scikit-learn
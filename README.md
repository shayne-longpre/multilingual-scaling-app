# Scaling Laws App

A library and web app for implementing, fitting, and visualizing scaling laws for machine learning models.

## Overview

With this, researchers can: 

1. **Visualize scaling laws** from prior papers and get optimal model size (N\*) and training tokens (D\*) recommendations for a given compute budget
2. **Fit custom scaling laws** to their own (N, D, Loss) data
3. **Optimize compute allocation** for both training and inference

## Installation

```bash
pip install -r requirements.txt
```

## Scaling Law Classes

All scaling law classes inherit from `ScalingLaw` and implement a consistent API.

### Available Classes

| Class | Description |
|-------|-------------|
| `ChinchillaScalingLaw` | Standard Chinchilla scaling law with sympy-based formula parsing |
| `BasicScalingLaw` | Simple hard-coded implementation: L = E + A/N^α + B/D^β to be replaced by `ChinchillaScalingLaw` |
| `DataConstrainedScalingLaw` | Handles data repetition constraints with unique tokens (U) |
| `GeneralScalingLaw` | Flexible scaling law with custom formula strings |

### Basic Usage

```python
from src.scaling_law_classes.basic_scaling_law import BasicScalingLaw

# Initialize with parameters
law = BasicScalingLaw(params={
    'A': 406.4,
    'B': 410.7,
    'irreducible': 1.69,
    'alpha': 0.34,
    'beta': 0.28
})

# Predict loss for given N and D
loss = law.loss(N=1e9, D=1e10)

# Find optimal allocation for compute budget
result = law.compute_optimal_allocation(C=1e20)
print(f"Optimal N: {result['model']:.2e}, Optimal D: {result['data']:.2e}")
```

### Fitting a Scaling Law

```python
from src.scaling_law_classes.chinchilla_scaling_law import ChinchillaScalingLaw

law = ChinchillaScalingLaw()
loss, params = law.fit("path/to/data.csv")  # CSV with N, D, Loss columns
print(f"Fitted parameters: {params}")
```

## Loss Function API

All scaling law classes implement `torch_loss` and `numpy_loss` with consistent signatures:

```python
def torch_loss(
    self,
    params_list: torch.Tensor,      # optimizer parameters
    form_exp_parts: Callable,        # evaluates log-sum-exp components
    inp: Dict[str, torch.Tensor],   # {'N': ..., 'D': ..., 'Loss': ...}
    tie_indices: List[List[int]] = [],
    loss_kwargs: Dict = {'loss_func': 'log_huber', 'delta': 1e-3},
) -> torch.Tensor
```

Supported loss functions: `log_huber`, `huber`, `log_mae`, `log_mse`

## Key Methods

| Method | Description |
|--------|-------------|
| `loss(N, D, **vars)` | Compute loss for given parameters |
| `compute_optimal_allocation(C)` | Find optimal N, D for compute budget C |
| `compute_optimal_allocation_inference(C, D_inference)` | Optimize considering inference costs |
| `N_to_D(N, target_loss)` | Find D needed to achieve target loss at given N |
| `DL_to_N(D, L)` | Find minimum N needed to achieve loss L with D tokens |
| `fit(data)` | Fit scaling law parameters to data |

## Testing

```bash
# Run all tests
pytest

# Run signature tests only
pytest tests/test_scaling_law_signatures.py -v
```

## Running the Web App

Run this command to launch the backend:
```
uv run uvicorn api.main:app --port 8000
```

And this one to launch the frontend:
```
cd frontend
npm run dev
```

## Project Structure

```
src/
  scaling_law_classes/
    scaling_law.py              # Base class and ScalingLawWrapper
    chinchilla_scaling_law.py   # Chinchilla implementation
    basic_scaling_law.py        # Simple scaling law
    data_constrained_scaling_law.py  # Data-constrained variant
    general_scaling_law.py      # Flexible formula-based law
  scaling_laws.py               # Registry of pre-defined laws (ALL_SCALING_LAWS)
  helpers/
    plotting.py                 # Visualization utilities
tests/
  test_scaling_law_signatures.py  # API consistency tests
```

## TODOs

- test GeneralScalingLaw more thoroughly
- improve auto-generation of log form for GeneralScalingLaw
- modify BasicScalingLaw so it has hardcoded parameters and can't be fit again
- improve web app by adding a table showing optimal N, D values and the predicted Loss for a particular Compute budget
- improve web app plots by adding a line to each plot showing the Loss vs. the optimal N, D, and C with optimal N, D allotment
- sanitize csv input. Possibly move to a textbox instead, which looks ugly, but does at least work 
- consider moving the custom GeneralScalingLaw fitting to a new tab in the web app
- add text to the web app with instructions, references, etc.

## References

- [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Scaling Data-Constrained Language Models (Muennighoff et al., 2023)](https://arxiv.org/abs/2305.16264)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2401.00448)

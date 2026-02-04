import sys

import numpy as np

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.basic_scaling_law import BasicScalingLaw
from src.scaling_law_classes.chinchilla_scaling_law import ChinchillaScalingLaw
from src.scaling_law_classes.data_constrained_scaling_law import (
    DataConstrainedScalingLaw,
)
from src.scaling_law_classes.scaling_law import ScalingLawWrapper


ALL_SCALING_LAWS = {}

# -----------------------------------------------------------------------------
# Option 1: BasicScalingLaw with Chinchilla initialization
# -----------------------------------------------------------------------------
ALL_SCALING_LAWS["Chinchilla (Hoffmann 2022)"] = ScalingLawWrapper(
    name="Chinchilla (Hoffmann 2022)",
    scaling_law=BasicScalingLaw(
        params={
            "A": 406.4,
            "B": 410.7,
            "alpha": 0.3392,
            "beta": 0.2849,
            "E": 1.69,
        }
    ),
    use_init_params=True,
    paper="https://arxiv.org/pdf/2203.15556",
    publication_date="2022-03-29",
    model_architecture="Transformer (decoder-only)",
    training_data="MassiveText (1.4T tokens)",
    languages=["English"],
    compute_budget_range=(int(6e18), int(5e23)),
    extra_args=[],
    notes="Uses Chinchilla paper params as initial guess, then BFGS optimization",
)

# -----------------------------------------------------------------------------
# Option 2: BasicScalingLaw with Porian et al initialization
# -----------------------------------------------------------------------------
ALL_SCALING_LAWS["Chinchilla Replication (Besiroglu 2024)"] = ScalingLawWrapper(
    name="Chinchilla Replication (Besiroglu 2024)",
    scaling_law=BasicScalingLaw(
        params={
            "A": 482.01,
            "B": 2085.43,
            "alpha": 0.3478,
            "beta": 0.3658,
            "E": 1.8172,
        }
    ),
    use_init_params=True,
    paper="https://www.arxiv.org/pdf/2404.10102",
    publication_date="2024-04-15",
    model_architecture="Transformer (decoder-only, RMSNorm)",
    training_data="FineWeb (15T tokens)",
    languages=["English"],
    compute_budget_range=(int(1e19), int(1e24)),
    extra_args=[],
    notes="Uses replication study params as initial guess, then BFGS optimization",
)

# -----------------------------------------------------------------------------
# Option 3: BasicScalingLaw with grid search
# -----------------------------------------------------------------------------
ALL_SCALING_LAWS["Basic Scaling Law (Grid Search)"] = ScalingLawWrapper(
    name="Basic Scaling Law (Grid Search)",
    scaling_law=BasicScalingLaw(params={}),
    use_init_params=False,
    paper="https://arxiv.org/pdf/2203.15556",
    publication_date="2022-03-29",
    model_architecture="Transformer (decoder-only)",
    training_data="Various",
    languages=["English"],
    compute_budget_range=(int(1e18), int(1e24)),
    extra_args=[],
    notes="Grid search over parameter space, then BFGS from each point",
)

# -----------------------------------------------------------------------------
# Option 4: ChinchillaScalingLaw with grid search
# -----------------------------------------------------------------------------
ALL_SCALING_LAWS["Chinchilla Scaling Law (Grid Search)"] = ScalingLawWrapper(
    name="Chinchilla Scaling Law (Grid Search)",
    scaling_law=ChinchillaScalingLaw(params={}),
    use_init_params=False,
    paper="https://arxiv.org/pdf/2203.15556",
    publication_date="2022-03-29",
    model_architecture="Transformer (decoder-only)",
    training_data="Various",
    languages=["English"],
    compute_budget_range=(int(1e18), int(1e24)),
    extra_args=[],
    notes="ChinchillaScalingLaw class with grid search initialization",
)

# -----------------------------------------------------------------------------
# Option 5: Data-Constrained Scaling Law
# -----------------------------------------------------------------------------
ALL_SCALING_LAWS["Data-Constrained Scaling Law"] = ScalingLawWrapper(
    name="Data-Constrained Scaling Law",
    scaling_law=DataConstrainedScalingLaw(
        params={
            "A": np.exp(6.255414),
            "B": np.exp(7.3049974),
            "alpha": 0.3526596,
            "beta": 0.3526596,
            "E": np.exp(0.6254804),
            "rd_star": 15.387756,
            "rn_star": 5.309743,
        }
    ),
    use_init_params=False,
    paper="https://arxiv.org/pdf/2305.16264",
    publication_date="2023-12-10",
    model_architecture="Transformer (decoder-only)",
    training_data="C4 & OSCAR with varying unique tokens",
    languages=["English"],
    compute_budget_range=(int(1e18), int(1e23)),
    extra_args=["U"],
    notes="For data repetition scenarios; requires U (unique tokens) column",
)

import sys

import numpy as np

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.basic_scaling_law import BasicScalingLaw
from src.scaling_law_classes.data_constrained_scaling_law import (
    DataConstrainedScalingLaw,
)
from src.scaling_law_classes.scaling_law import ScalingLawWrapper


ALL_SCALING_LAWS = {}

# -----------------------------------------------------------------------------
# Original Chinchilla Scaling Law (2022)
# -----------------------------------------------------------------------------
ALL_SCALING_LAWS["Chinchilla"] = ScalingLawWrapper(
    name="Chinchilla",
    scaling_law=BasicScalingLaw(
        params={
            "A": 406.4,
            "B": 410.7,
            "alpha": 0.3392,
            "beta": 0.2849,
            "E": 1.69,
        }
    ),
    paper="https://arxiv.org/pdf/2203.15556",
    publication_date="2022-03-29",
    model_architecture="Transformer (decoder-only)",
    training_data="MassiveText (1.4T tokens)",
    languages=["English"],
    compute_budget_range=(int(6e18), int(5e23)),  # ~6 × 10^18 to 5 × 10^23 FLOPs
    extra_args=[],
    notes="Original Chinchilla paper by DeepMind. Established compute-optimal training with L = E + A/N^α + B/D^β",
)

# -----------------------------------------------------------------------------
# Chinchilla Replication Study (2024)
# -----------------------------------------------------------------------------
ALL_SCALING_LAWS["Chinchilla Replication"] = ScalingLawWrapper(
    name="Chinchilla Replication",
    scaling_law=BasicScalingLaw(
        params={
            "A": 482.01,
            "B": 2085.43,
            "alpha": 0.3478,
            "beta": 0.3658,
            "E": 1.8172,
        }
    ),
    paper="https://www.arxiv.org/pdf/2404.10102",
    publication_date="2024-04-15",
    model_architecture="Transformer (decoder-only, RMSNorm)",
    training_data="FineWeb (15T tokens)",
    languages=["English"],
    compute_budget_range=(int(1e19), int(1e24)),  # ~10^19 to 10^24 FLOPs
    extra_args=[],
    notes="Replication study showing updated parameters with modern training practices and larger datasets",
)

# -----------------------------------------------------------------------------
# Data Constrained Scaling Laws (2023)
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
    paper="https://arxiv.org/pdf/2305.16264",
    publication_date="2023-12-10",
    model_architecture="Transformer (decoder-only)",
    training_data="C4 & OSCAR with varying unique tokens",
    languages=["English"],
    compute_budget_range=(int(1e18), int(1e23)),  # ~10^18 to 10^23 FLOPs
    extra_args=["U"],  # Requires unique tokens parameter
    notes="Extends basic scaling laws to account for data repetition effects when training on limited unique tokens",
)

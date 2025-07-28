import sys

import numpy as np

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.basic_scaling_law import BasicScalingLaw
from src.scaling_law_classes.data_constrained_scaling_law import (
    DataConstrainedScalingLaw,
)
from src.scaling_law_classes.scaling_law import LawParams, ScalingLawWrapper


ALL_SCALING_LAWS = {}

# -----------------------------------------------------------------------------
# Original Chinchilla Scaling Law (2022)
# -----------------------------------------------------------------------------
ALL_SCALING_LAWS["Chinchilla"] = ScalingLawWrapper(
    name="Chinchilla",
    scaling_law=BasicScalingLaw(
        LawParams(
            A=406.4,
            B=410.7,
            alpha=0.3392,
            beta=0.2849,
            irreducible=1.69,
        )
    ),
    paper="https://arxiv.org/pdf/2203.15556",
    publication_date="",  # TODO
    model_architecture="",
    training_data="",
    languages="",
    compute_budget_range=(0, 0),
    extra_args=[],
    notes="",
)

# -----------------------------------------------------------------------------
# Chinchilla Replication Study (2024)
# -----------------------------------------------------------------------------
ALL_SCALING_LAWS["Chinchilla Replication"] = ScalingLawWrapper(
    name="Chinchilla Replication",
    scaling_law=BasicScalingLaw(
        LawParams(
            A=482.01,
            B=2085.43,
            alpha=0.3478,
            beta=0.3658,
            irreducible=1.8172,
        )
    ),
    paper="https://www.arxiv.org/pdf/2404.10102",
    publication_date="",  # TODO
    model_architecture="",
    training_data="",
    languages="",
    compute_budget_range=(0, 0),
    extra_args=[],
    notes="",
)

# -----------------------------------------------------------------------------
# Data Constrained Scaling Laws (2023)
# -----------------------------------------------------------------------------
# ALL_SCALING_LAWS["Data-Constrained Scaling Law"] = ScalingLawWrapper(
#     name="Data-Constrained Scaling Law",
#     scaling_law=DataConstrainedScalingLaw(
#         LawParams(
#             A=np.exp(6.255414),
#             B=np.exp(7.3049974),
#             alpha=0.3526596,
#             beta=0.3526596,
#             irreducible=np.exp(0.6254804),
#             extras={"rd_star": 15.387756, "rn_star": 5.309743},
#         )
#     ),
#     paper="https://proceedings.neurips.cc/paper_files/paper/2023/file/"
#     "9d89448b63ce1e2e8dc7af72c984c196-Paper-Conference.pdf",
#     publication_date="",  # TODO
#     model_architecture="",
#     training_data="",
#     languages="",
#     compute_budget_range=(0, 0),
#     extra_args=[],
#     notes="",
# )

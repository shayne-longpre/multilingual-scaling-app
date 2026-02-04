import sys
from typing import Dict

import numpy as np
import torch

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.general_scaling_law import GeneralScalingLaw, minimize_scl_loss


class ChinchillaScalingLaw(GeneralScalingLaw):
    """
    Implementation of the Chinchilla scaling law.

    Loss formula: L = E + A/N^α + B/D^β

    This class inherits from GeneralScalingLaw and provides Chinchilla-specific defaults
    plus analytic overrides for methods like N_to_D and DL_to_N.
    """

    def __init__(
        self,
        params: Dict[str, float] = {},
        is_fit_already: bool = False,
    ):
        super().__init__(
            params=params,
            form_str="E + A / N**alpha + B / D**beta",
            params_str="A B E alpha beta",
            vars_str="N D",
            form_exp_parts_str=["logA - alpha * logN", "logB - beta * logD", "logE"],
            is_fit_already=is_fit_already,
        )

    def N_to_D(self, N: float, target_loss: float, **other_vars) -> float:
        """
        Analytic solution for D given N and target_loss.

        From L = E + A/N^α + B/D^β, solving for D:
        D = (B / (L - E - A/N^α))^(1/β)
        """
        L_eff = target_loss - self.params["E"]
        if L_eff <= 0:
            raise ValueError("target_loss must exceed irreducible loss")
        denom = L_eff - self.params["A"] / N**self.params["alpha"]
        if denom <= 0:
            raise ValueError(
                "No finite D can satisfy the loss at this N (denominator ≤ 0)"
            )
        D = (self.params["B"] / denom) ** (1.0 / self.params["beta"])
        return D

    def DL_to_N(self, D: float, L: float, **other_vars) -> float:
        """
        Analytic solution for N given D and target loss L.

        From L = E + A/N^α + B/D^β, solving for N:
        N = (A / (L - E - B/D^β))^(1/α)
        """
        L_eff = L - self.params["E"]

        if L_eff <= 0:
            raise ValueError(
                f"Target loss {L} must exceed irreducible loss {self.params['E']}"
            )

        denominator = L_eff - self.params["B"] / (D**self.params["beta"])

        if denominator <= 0:
            raise ValueError(
                f"Cannot achieve loss {L} with {D} tokens - need more data"
            )

        partial_result = self.params["A"] / denominator
        return partial_result ** (1 / self.params["alpha"])

    def compute_optimal_train_tokens(self, x: float, T: float, L: float) -> float:
        """
        Equation (12) in https://arxiv.org/pdf/2401.00448

        Find the optimal number of tokens (D) to train on for a model
        of quality L (pre-training loss L) and run inference for T tokens.
        This method is used by a solver (e.g. Newton's method) to find root (optimal D).
        We cannot use a formula because there is no analytical formula when T > 0.

        The equation is:
        (β·B/α + B)·D^(-β) + (T·β·B)/(3·α)·D^(-β-1) + E - L = 0
        """
        p = self.params

        coeff_1 = (p["beta"] * p["B"]) / p["alpha"] + p["B"]
        coeff_2 = (T * p["beta"] * p["B"]) / (3 * p["alpha"])
        loss_diff = p["E"] - L

        return (
            coeff_1 * x ** (-1 * p["beta"])
            + coeff_2 * x ** ((-1 * p["beta"]) - 1)
            + loss_diff
        )

    def fit(self, data, init_params: Dict[str, float] | None = None, *args, **kwargs):
        """
        Fit the Chinchilla scaling law to data.

        Uses the inherited GeneralScalingLaw.fit() method with Chinchilla-specific
        parameter conversion.
        """
        loss, fit_params = super().fit(data, init_params, *args, **kwargs)

        # Ensure we return all expected params with correct names
        result_params = {
            "A": fit_params.get("A"),
            "B": fit_params.get("B"),
            "E": fit_params.get("E"),
            "alpha": fit_params.get("alpha"),
            "beta": fit_params.get("beta"),
        }

        return loss, result_params

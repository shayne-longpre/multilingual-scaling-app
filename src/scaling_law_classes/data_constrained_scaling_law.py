import sys

import numpy as np
import torch
from scipy.optimize import brentq

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.scaling_law import ScalingLaw


class DataConstrainedScalingLaw(ScalingLaw):
    variables = ("N", "D", "U")
    default_vars = {"N": 1.0, "D": 1.0, "U": 1.0}

    # --- NumPy loss ------------------------------------------------------
    def loss_expr(self, *, N: float, D: float, U: float, **kwargs):
        p = self.params

        RD = np.maximum((D / U) - 1, 0)
        UN = np.minimum(N, self.D_to_N(U))
        RN = np.maximum((N / UN) - 1, 0)
        model_denom = UN + UN * p.extras["rn_star"] * (
            1 - np.exp(-1 * RN / p.extras["rn_star"])
        )
        data_denom = U + U * p.extras["rd_star"] * (
            1 - np.exp(-1 * RD / (p.extras["rd_star"]))
        )

        loss = (
            p.irreducible
            + (p.A / (model_denom**p.alpha))
            + (p.B / (data_denom**p.beta))
        )
        return loss

    def D_to_N(self, D):
        return (D * self.G) ** (self.params.beta / self.params.alpha) * self.G

    def DL_to_N(self, D, L):
        return 0.0  # TODO

    def compute_optimal_train_tokens(self, x, T, L):
        return 0.0 # TODO

    # --- Numeric N → D using root‑finder --------------------------------
    def N_to_D(self, N: float, target_loss: float, **other_vars) -> float:
        U = other_vars.get("U")
        if U is None:
            raise ValueError("DataConstrainedScalingLaw.N_to_D requires keyword 'U'")

        # Quick feasibility check – use current loss at some D to bracket root
        def loss_minus_L(D):
            return self.loss(N=N, D=D, U=U) - target_loss

        # Attempt to find a sign change automatically.
        D_low, D_high = 1e-6, 1e6
        try:
            # Expand upper bound until the function becomes positive
            while loss_minus_L(D_high) > 0 and D_high < 1e12:
                D_high *= 10
            # Ensure lower bound is below target
            while loss_minus_L(D_low) < 0 and D_low > 1e-12:
                D_low /= 10
            root = brentq(loss_minus_L, D_low, D_high, maxiter=256)
            return root
        except ValueError as e:
            raise ValueError("Unable to bracket iso‑loss root for given N.") from e

    @staticmethod
    def torch_loss(inp: torch.Tensor, theta: torch.Tensor):
        a, b, e, alpha, beta, ep_star, n_star = theta
        tm = inp[:, 0] + inp[:, 0] * n_star * (1 - torch.exp(-inp[:, 3] / n_star))
        td = inp[:, 1] + inp[:, 1] * ep_star * (1 - torch.exp(-inp[:, 2] / ep_star))
        pre = torch.stack(
            [
                a - alpha * torch.log(tm),
                b - beta * torch.log(td),
                e.expand(inp.shape[0]),
            ]
        )
        post = torch.logsumexp(pre, dim=0)
        return torch.nn.functional.huber_loss(
            post, torch.log(inp[:, 4]), delta=1e-3, reduction="none"
        ).sum()

    @staticmethod
    def numpy_loss(inp: np.ndarray, params: np.ndarray) -> np.ndarray:
        a, b, e, alpha, beta, ep_star, n_star = params
        UN, U, RD, RN = inp[:, 0], inp[:, 1], inp[:, 2], inp[:, 3]
        tm = UN + UN * n_star * (1 - np.exp(-RN / n_star))
        td = U + U * ep_star * (1 - np.exp(-RD / ep_star))
        return np.exp(e) + np.exp(a) / tm**alpha + np.exp(b) / td**beta

    def iso_loss_function(self, target_loss: float, **other_vars):
        if "U" not in other_vars:
            raise ValueError("iso_loss_function requires keyword argument 'U'")
        return super().iso_loss_function(target_loss, **other_vars)

    def compute_optimal_allocation(self, C, *, U, **kw):
        return super().compute_optimal_allocation(C, U=U, **kw)

    # @classmethod
    # def fit(cls, *args, **kw):
    #     from src.scaling_fitters import fit_data_constrained_scaling
    #     return fit_data_constrained_scaling(cls, *args, **kw)

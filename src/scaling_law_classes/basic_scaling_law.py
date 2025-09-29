import sys
import math
import numpy as np
import torch

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.scaling_law import ScalingLaw, LawParams


class BasicScalingLaw(ScalingLaw):
    variables = ("N", "D")
    default_vars = {"N": 1.0, "D": 1.0}

    # --- NumPy loss ------------------------------------------------------
    def loss_expr(self, *, N: float, D: float, **kwargs):
        p = self.params
        return p.irreducible + p.A / N**p.alpha + p.B / D**p.beta

    # --- Torch loss (same as before) ------------------------------------
    @staticmethod
    def torch_loss(inp: torch.Tensor, theta: torch.Tensor, *, tie: bool = True):
        if tie:
            a, b, e, alpha, beta = theta[0], theta[1], theta[2], theta[3], theta[3]
        else:
            a, b, e, alpha, beta = theta
        pre = torch.stack(
            [
                a - alpha * torch.log(inp[:, 0]),
                b - beta * torch.log(inp[:, 1]),
                e.expand(inp.shape[0]),
            ]
        )
        post = torch.logsumexp(pre, dim=0)
        return torch.nn.functional.huber_loss(
            post, torch.log(inp[:, 2]), delta=1e-3
        ).sum()

    # --- NumPy pred for curve_fit --------------------------------------
    @staticmethod
    def numpy_loss(
        inp: np.ndarray, params: np.ndarray, *, tie: bool = True
    ) -> np.ndarray:
        if tie:
            a, b, e, alpha, beta = params[0], params[1], params[2], params[3], params[3]
        else:
            a, b, e, alpha, beta = params
        N, D = inp[:, 0], inp[:, 1]
        return np.exp(e) + np.exp(a) / N**alpha + np.exp(b) / D**beta

    # --- Analytic N → D on iso‑loss ------------------------------------
    def N_to_D(self, N: float, target_loss: float, **other_vars) -> float:
        p = self.params
        L_eff = target_loss - p.irreducible
        if L_eff <= 0:
            raise ValueError("target_loss must exceed irreducible loss")
        denom = L_eff - p.A / N**p.alpha
        if denom <= 0:
            raise ValueError(
                "No finite D can satisfy the loss at this N (denominator ≤ 0)"
            )
        D = (p.B / denom) ** (1.0 / p.beta)
        return D

    def DL_to_N(self, D, L):
        """
        Minimum number of model params needed to reach L model loss after D tokens.

        This is the regular Chinchilla equation solved for N.
        """
        p = self.params
        L_eff = L - p.irreducible

        if L_eff <= 0:
            raise ValueError(
                f"Target loss {L} must exceed irreducible loss {p.irreducible}"
            )

        denominator = L_eff - p.B / (D**p.beta)

        if denominator <= 0:
            raise ValueError(
                f"Cannot achieve loss {L} with {D} tokens - need more data"
            )

        partial_result = p.A / denominator
        return partial_result ** (1 / p.alpha)

    def compute_optimal_train_tokens(self, x, T, L):
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

        coeff_1 = (p.beta * p.B) / p.alpha + p.B
        coeff_2 = (T * p.beta * p.B) / (3 * p.alpha)
        loss_diff = p.irreducible - L

        return (
            coeff_1 * x ** (-1 * p.beta)
            + coeff_2 * x ** ((-1 * p.beta) - 1)
            + loss_diff
        )

    @classmethod
    def fit(cls, data, *args, **kw):
        N  = data["N"].values.astype(float)
        D  = data["D"].values.astype(float)
        y    = data["Loss"].values.astype(float)

        # unique_tokens = data["U"].iloc[0]
        # min_epochs = round(data["D"].min() / unique_tokens,2)
        # max_epochs = round(data["D"].max() / unique_tokens,2)
        # print(f"Data points: {len(data)}. Ranging from {min_epochs} to {max_epochs} epochs.")

        grid = {
                'a': torch.arange(start=0, end=25, step=5),
                'b': torch.arange(start=0, end=25, step=5),
                'e': torch.arange(start=-1, end=1, step=0.5),
                'alpha': torch.arange(start=0, end=2, step=0.5),
                'beta': torch.arange(start=0, end=2, step=0.5)
            }
        loss, theta, pq = minimize_scl_loss(
            init_params=None,  # ignored because grid_specs is provided
            grid_specs=grid,
            torch_loss    = cls.torch_loss,
            inp_torch     = torch.tensor(np.c_[N, D, y], dtype=torch.float32),
            loss_kwargs   = {"tie": args.tie},
        )

        if args.tie:
            A, B, E, alpha = theta ; beta = alpha
        else:
            A, B, E, alpha, beta = theta
        params = LawParams(A=np.exp(A), B=np.exp(B), irreducible=np.exp(E), alpha=alpha, beta=beta)
        return loss, cls(params)

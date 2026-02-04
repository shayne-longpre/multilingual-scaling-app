import sys
import math
from typing import Callable, Dict, List
import numpy as np
import torch

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.scaling_law import ScalingLaw
from src.scaling_law_classes.general_scaling_law import minimize_scl_loss


class BasicScalingLaw(ScalingLaw):

    # --- NumPy loss ------------------------------------------------------
    def loss_expr(self, *, N: float, D: float, **kwargs):
        p = self.params
        return p['E'] + p['A'] / N**p['alpha'] + p['B'] / D**p['beta']

    def apply_form_exp_parts(self, params_list: torch.Tensor, inp: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Evaluate the log-sum-exp parts using the optimizer's current parameter values.

        Args:
            params_list: tensor of parameter values from optimizer [logA, logB, logE, alpha, beta]
            inp: dict with 'N', 'D', 'Loss' tensors
        """
        logA, logB, logE, alpha, beta = params_list[0], params_list[1], params_list[2], params_list[3], params_list[4]
        N = inp['N']
        D = inp['D']
        return [
            logA - alpha * torch.log(N),
            logB - beta * torch.log(D),
            logE.expand(N.shape[0]),
        ]

    def torch_loss(
        self,
        params_list: torch.Tensor,
        form_exp_parts: Callable,
        inp: Dict[str, torch.Tensor],
        tie_indices: List[List[int]] = [],
        loss_kwargs: Dict = {'loss_func': 'log_huber', 'delta': 1e-3},
    ) -> torch.Tensor:
        loss_func = loss_kwargs.get('loss_func', 'log_huber')
        delta = loss_kwargs.get('delta', 1e-3)

        for tie_params in tie_indices:
            tie_source = params_list[tie_params[0]]
            for i in tie_params[1:]:
                params_list[i] = tie_source

        pre = torch.stack(form_exp_parts(params_list, inp))
        post = torch.logsumexp(pre, dim=0)

        if loss_func == 'log_huber':
            return torch.nn.functional.huber_loss(
                post, torch.log(inp["Loss"]), delta=delta, reduction="none"
            ).sum()
        elif loss_func == 'huber':
            return torch.nn.functional.huber_loss(
                torch.exp(post), inp["Loss"], delta=delta, reduction="none"
            ).sum()
        elif loss_func == 'log_mae':
            return torch.abs(torch.log(inp["Loss"]) - post).sum()
        elif loss_func == 'log_mse':
            return ((torch.log(inp["Loss"]) - post) ** 2).sum()
        else:
            raise NotImplementedError(f"Loss function {loss_func} not implemented.")

    def numpy_loss(
        self,
        params_list: np.ndarray,
        form_exp_parts: Callable,
        inp: Dict[str, np.ndarray],
        tie_indices: List[List[int]] = [],
        loss_kwargs: Dict = {'loss_func': 'log_huber', 'delta': 1e-3},
    ) -> np.ndarray:
        loss_func = loss_kwargs.get('loss_func', 'log_huber')
        delta = loss_kwargs.get('delta', 1e-3)

        for tie_params in tie_indices:
            tie_source = params_list[tie_params[0]]
            for i in tie_params[1:]:
                params_list[i] = tie_source

        pre = np.stack(form_exp_parts(params_list, inp))
        post = np.logaddexp.reduce(pre, axis=0)

        if loss_func == 'log_huber':
            return np.sum(
                np.where(
                    np.abs(np.log(inp["Loss"]) - post) <= delta,
                    0.5 * (np.log(inp["Loss"]) - post)**2,
                    delta * (np.abs(np.log(inp["Loss"]) - post) - 0.5 * delta)))
        elif loss_func == 'huber':
            return np.sum(
                np.where(
                    np.abs(inp["Loss"] - np.exp(post)) <= delta,
                    0.5 * (inp["Loss"] - np.exp(post))**2,
                    delta * (np.abs(inp["Loss"] - np.exp(post)) - 0.5 * delta)))
        elif loss_func == 'log_mae':
            return np.abs(np.log(inp["Loss"]) - post).sum()
        elif loss_func == 'log_mse':
            return ((np.log(inp["Loss"]) - post) ** 2).sum()
        else:
            raise NotImplementedError(f"Loss function {loss_func} not implemented.")

    # --- Analytic N → D on iso‑loss ------------------------------------
    def N_to_D(self, N: float, target_loss: float, **other_vars) -> float:
        p = self.params
        L_eff = target_loss - p['E']
        if L_eff <= 0:
            raise ValueError("target_loss must exceed irreducible loss")
        denom = L_eff - p['A'] / N**p['alpha']
        if denom <= 0:
            raise ValueError(
                "No finite D can satisfy the loss at this N (denominator ≤ 0)"
            )
        D = (p['B'] / denom) ** (1.0 / p['beta'])
        return D

    def DL_to_N(self, D, L):
        """
        Minimum number of model params needed to reach L model loss after D tokens.

        This is the regular Chinchilla equation solved for N.
        """
        p = self.params
        L_eff = L - p['E']

        if L_eff <= 0:
            raise ValueError(
                f"Target loss {L} must exceed irreducible loss {p['E']}"
            )

        denominator = L_eff - p['B'] / (D**p['beta'])

        if denominator <= 0:
            raise ValueError(
                f"Cannot achieve loss {L} with {D} tokens - need more data"
            )

        partial_result = p['A'] / denominator
        return partial_result ** (1 / p['alpha'])

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

        coeff_1 = (p['beta'] * p['B']) / p['alpha'] + p['B']
        coeff_2 = (T * p['beta'] * p['B']) / (3 * p['alpha'])
        loss_diff = p['E'] - L

        return (
            coeff_1 * x ** (-1 * p['beta'])
            + coeff_2 * x ** ((-1 * p['beta']) - 1)
            + loss_diff
        )

    # Param name mappings for optimizer
    optim_params_names = ['logA', 'logB', 'logE', 'alpha', 'beta']

    def fit(self, data, init_params: dict | None = None, *args, **kwargs):
        N = data["N"].values.astype(float)
        D = data["D"].values.astype(float)
        L = data["Loss"].values.astype(float)

        if init_params is not None:
            # Use provided params as single starting point
            grid = {
                'logA': torch.tensor([np.log(init_params['A'])], dtype=torch.float32),
                'logB': torch.tensor([np.log(init_params['B'])], dtype=torch.float32),
                'logE': torch.tensor([np.log(init_params['E'])], dtype=torch.float32),
                'alpha': torch.tensor([init_params['alpha']], dtype=torch.float32),
                'beta': torch.tensor([init_params['beta']], dtype=torch.float32)
            }
        else:
            # Use hardcoded grid (current behavior)
            grid = {
                'logA': torch.arange(start=0, end=25+5, step=5.0),
                'logB': torch.arange(start=0, end=25+5, step=5.0),
                'logE': torch.arange(start=-1, end=1+0.5, step=0.5),
                'alpha': torch.arange(start=0, end=2+0.5, step=0.5),
                'beta': torch.arange(start=0, end=2+0.5, step=0.5)
            }

        # Create input dict with tensors
        inp_torch = {
            "N": torch.tensor(N, dtype=torch.float32),
            "D": torch.tensor(D, dtype=torch.float32),
            "Loss": torch.tensor(L, dtype=torch.float32),
        }

        loss, theta, _pq = minimize_scl_loss(
            init_params     = None,  # ignored because grid_specs is provided
            grid_specs      = grid,
            torch_loss      = self.torch_loss,
            form_exp_parts  = self.apply_form_exp_parts,
            inp_torch       = inp_torch,
            loss_kwargs     = {"tie_groups": kwargs.get('tie', []), "delta": 1e-3, "loss_func": "log_huber"},
            param_names     = self.optim_params_names,
        )

        # theta is a dict with optim_params_names keys (logA, logB, logE, alpha, beta)
        # Convert to fit_params with original param names (A, B, E, alpha, beta)
        fit_params = {
            "A": np.exp(theta['logA']),
            "B": np.exp(theta['logB']),
            "E": np.exp(theta['logE']),
            "alpha": theta['alpha'],
            "beta": theta['beta']
        }
        return loss, fit_params
